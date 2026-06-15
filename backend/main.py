import logging
import os
import re
import sys
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gitbo")


# ── Request schema with input validation ──────────────────────────────────────

# FIX #9: Only accept valid GitHub HTTPS URLs to prevent SSRF / local-path abuse
_GITHUB_HTTPS_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(\.git)?/?$"
)
_MAX_ISSUE_LEN = 8_000  # characters — prevents oversized prompts


class AnalyzeRequest(BaseModel):
    repo_url: str
    github_token: str
    issue_description: str

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        v = v.strip()
        if not _GITHUB_HTTPS_RE.match(v):
            raise ValueError(
                "repo_url must be a valid GitHub HTTPS URL "
                "(e.g. https://github.com/owner/repo)"
            )
        return v

    @field_validator("github_token")
    @classmethod
    def validate_token(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("github_token must not be empty.")
        return v

    @field_validator("issue_description")
    @classmethod
    def validate_issue(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("issue_description must not be empty.")
        if len(v) > _MAX_ISSUE_LEN:
            raise ValueError(f"issue_description must be ≤ {_MAX_ISSUE_LEN} characters.")
        return v


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Gitbo – CodeOnboard Developer API")

# FIX #3: Restrict CORS to known local origins only.
# For production deployments, set the ALLOWED_ORIGINS env var to the
# frontend's actual domain (e.g. "https://yourapp.example.com").
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# FIX #7: Fail fast — if the graph fails to compile, log the full traceback
# and keep app_graph = None so the health check can report it clearly.
try:
    from backend.agents import app_graph
    logger.info("LangGraph pipeline compiled successfully.")
except Exception:
    logger.critical(
        "FATAL: LangGraph pipeline failed to compile:\n%s", traceback.format_exc()
    )
    app_graph = None


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Simple health check — also reports graph compilation status."""
    return {
        "status": "ok",
        "graph_ready": app_graph is not None,
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if app_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline unavailable: LangGraph failed to compile. Check server logs.",
        )

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Add it to your .env file.",
        )

    initial_state = {
        "repo_url": request.repo_url,
        "github_token": request.github_token,
        "issue_description": request.issue_description,
        "issue_analysis": {},
        "retrieved_context": [],
        "implementation_plan": {},
        "pr_result": {},
        "git_diff": "",
        "temp_dir": "",
        "errors": [],
    }

    try:
        # FIX #6: The pipeline performs blocking I/O (git clone, embedding,
        # LLM calls).  Running it directly in an async handler blocks the
        # entire event loop.  run_in_threadpool offloads it to a thread-pool
        # worker so other requests are not stalled.
        result = await run_in_threadpool(app_graph.invoke, initial_state)
        return result
    except Exception as e:
        logger.exception("Pipeline execution error")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


# ── Static frontend ───────────────────────────────────────────────────────────

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s. Only serving API.", frontend_dir)
