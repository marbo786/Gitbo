import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure the root of the workspace is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load local .env file
load_dotenv()

# Declare schema
class AnalyzeRequest(BaseModel):
    repo_url: str
    github_token: str
    issue_description: str

app = FastAPI(title="Gitbo - CodeOnboard Developer API")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import compiled LangGraph workflow
try:
    from backend.agents import app_graph
except Exception as e:
    print(f"Error loading agents/compiled graph: {str(e)}")
    app_graph = None

@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if not app_graph:
        raise HTTPException(
            status_code=500, 
            detail="LangGraph flow failed to compile. Please check server logs."
        )
        
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not configured. Please set it in a .env file."
        )
        
    try:
        # Initialize LangGraph state
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
            "errors": []
        }
        
        # Execute the Graph
        result = app_graph.invoke(initial_state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

# Mount static frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_dir}. FastAPI will only serve API.")
