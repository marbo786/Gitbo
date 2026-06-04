import json
import os
import re
import shutil
import tempfile
import time
import uuid

import chromadb
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from backend.utils.chunker import chunk_file
from backend.utils.git_handler import (
    clone_repo,
    create_branch_and_commit,
    get_git_diff,
    push_branch,
    create_pull_request,
)
from backend.utils.parser import apply_search_replace


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    repo_url: str
    github_token: str
    issue_description: str
    issue_analysis: dict        # Filled by Agent 1
    retrieved_context: list     # Filled by Agent 2
    implementation_plan: dict   # Filled by Agent 3
    pr_result: dict             # Filled by Agent 4
    git_diff: str               # Filled by Agent 4
    temp_dir: str               # Set by Agent 2, used by Agent 4
    errors: List[str]


# ── Shared embedding model (lazy singleton) ───────────────────────────────────

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedding_model


# ── JSON helpers ──────────────────────────────────────────────────────────────

def clean_invalid_sets(raw_json: str) -> str:
    """Convert curly-brace sets { "a", "b" } → valid JSON arrays ["a", "b"]."""
    chars = list(raw_json)
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] == "{":
            brace_count, j, in_quote, has_colon, has_comma = 1, i + 1, False, False, False
            while j < n:
                c = chars[j]
                if c == '"' and (j == 0 or chars[j - 1] != "\\"):
                    in_quote = not in_quote
                elif not in_quote:
                    if c == "{":
                        brace_count += 1
                    elif c == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    elif c == ":":
                        has_colon = True
                    elif c == ",":
                        has_comma = True
                j += 1
            if brace_count == 0 and has_comma and not has_colon:
                chars[i] = "["
                chars[j] = "]"
            i = j
        i += 1
    return "".join(chars)


def extract_json(text: str) -> dict:
    """
    Extract and parse the first JSON object in an LLM response.

    FIX #10: Instead of a greedy .* match (which spans multiple JSON blocks
    and can grab braces from prose), we use json.JSONDecoder.raw_decode to
    find the first valid JSON object starting from the first '{'.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:500]}")

    decoder = json.JSONDecoder(strict=False)
    # Try parsing from each '{' position until one succeeds
    pos = start
    while pos < len(text):
        if text[pos] != "{":
            pos += 1
            continue
        try:
            obj, _ = decoder.raw_decode(text, pos)
            return obj
        except json.JSONDecodeError:
            pos += 1

    # Fallback: try with set-cleaning and comment stripping on the substring
    raw_json = text[start:]
    try:
        cleaned = clean_invalid_sets(raw_json)
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        cleaned = re.sub(r"^\s*//.*$", "", cleaned, flags=re.MULTILINE)
        return json.loads(cleaned, strict=False)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}\nRaw: {raw_json[:500]}")


# ── LLM helpers ───────────────────────────────────────────────────────────────

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    from groq import Groq
    return Groq(api_key=api_key)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ── Misc helpers ──────────────────────────────────────────────────────────────

def extract_file_paths_from_text(text: str) -> list[str]:
    pattern = r"\b[a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]{2,4}\b"
    paths = []
    for c in re.findall(pattern, text):
        c = c.strip().strip("'\"` ()")
        if re.match(r"^\d+\.\d+$", c) or not c or c.startswith("."):
            continue
        _, ext = os.path.splitext(c.lower())
        if ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".html",
                   ".java", ".cpp", ".h", ".go", ".yml", ".yaml", ".toml"):
            paths.append(c)
    return list(set(paths))


def slugify(text: str) -> str:
    """
    FIX #11: Added a non-empty fallback so branch names are always valid.
    Also strips trailing hyphens that could result from truncation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    slug = text.strip("-")[:30].rstrip("-")
    return slug if slug else "fix"


def _cleanup_temp(temp_dir: str) -> None:
    """FIX #5: Safely remove the cloned temp directory."""
    if temp_dir and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as exc:
            print(f"[Cleanup] Warning: could not remove {temp_dir}: {exc}")


# ── Agent nodes ───────────────────────────────────────────────────────────────

def issue_understanding_node(state: AgentState) -> dict:
    print("[Agent 1] Analyzing issue description...")
    errors = state.get("errors", [])
    try:
        system_prompt = (
            "You are an expert issue triage agent. Analyze the GitHub issue and extract its core meaning.\n"
            "Return strict JSON only — no prose, no markdown wrappers.\n"
            "Schema:\n"
            "{\n"
            '  "problem": "Brief summary of the core bug or feature request.",\n'
            '  "category": "Bug" | "Feature" | "Refactor",\n'
            '  "difficulty": "Easy" | "Medium" | "Hard",\n'
            '  "skills": ["Language1", "Framework1"],\n'
            '  "search_queries": [\n'
            '    "distinct semantic query 1",\n'
            '    "distinct semantic query 2",\n'
            '    "distinct semantic query 3"\n'
            "  ]\n"
            "}\n"
            "Make search_queries target specific function names, variable names, or error messages."
        )
        response_text = call_llm(system_prompt, f"Issue:\n{state['issue_description']}")
        analysis = extract_json(response_text)
        return {"issue_analysis": analysis}
    except Exception as e:
        msg = f"Issue Understanding Agent failed: {e}"
        print(f"[ERROR] {msg}")
        return {"errors": errors + [msg], "issue_analysis": {}}


def codebase_search_node(state: AgentState) -> dict:
    print("[Agent 2] Searching codebase for relevant files...")
    errors = state.get("errors", [])
    if not state.get("issue_analysis"):
        return {"errors": errors + ["Cannot proceed to codebase search: Issue analysis is empty."]}

    repo_url = state["repo_url"]
    token = state["github_token"]
    queries = state["issue_analysis"].get("search_queries", [])

    temp_dir = tempfile.mkdtemp(prefix="gitbo_clone_")
    try:
        print(f"[Agent 2] Cloning repo to {temp_dir}")
        clone_repo(repo_url, token, temp_dir)

        # FIX #12: Broadened file extension list to include config/project files
        code_extensions = (
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".h", ".go",
            ".css", ".html",
            ".yml", ".yaml", ".toml", ".json",
        )
        exclude_dirs = {
            ".git", "node_modules", "venv", ".venv", "__pycache__",
            "dist", "build", ".next", ".nuxt",
        }
        # Cap JSON/config files at a reasonable size to avoid embedding huge lock files
        _LARGE_FILE_LIMIT = 100_000  # bytes

        all_chunks = []
        for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext not in code_extensions:
                    continue
                full_path = os.path.join(root, file)
                # Skip very large config files (e.g. package-lock.json)
                if os.path.getsize(full_path) > _LARGE_FILE_LIMIT:
                    continue
                rel_path = os.path.relpath(full_path, temp_dir).replace("\\", "/")
                for chunk in chunk_file(full_path):
                    chunk["file_path"] = rel_path
                    all_chunks.append(chunk)

        if not all_chunks:
            # FIX #5: Clean up even when returning early with empty context
            _cleanup_temp(temp_dir)
            return {"temp_dir": "", "retrieved_context": [],
                    "errors": errors + ["No code chunks found in repository."]}

        print(f"[Agent 2] Generating embeddings for {len(all_chunks)} chunks...")
        model = get_embedding_model()
        contents = [c["content"] for c in all_chunks]
        embeddings = model.encode(contents, show_progress_bar=False).tolist()

        client = chromadb.EphemeralClient()
        collection = client.create_collection(name=f"repo_{uuid.uuid4().hex[:16]}")
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            embeddings=embeddings,
            documents=contents,
            metadatas=[{
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "name": c["name"],
                "type": c["type"],
            } for c in all_chunks],
        )

        retrieved: dict = {}
        for q in queries:
            print(f"[Agent 2] Query: '{q}'")
            q_emb = model.encode([q]).tolist()[0]
            results = collection.query(query_embeddings=[q_emb], n_results=3)
            if results and results["documents"]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    key = (meta["file_path"], meta["start_line"], meta["end_line"])
                    if key not in retrieved:
                        retrieved[key] = {
                            "file_path": meta["file_path"],
                            "start_line": meta["start_line"],
                            "end_line": meta["end_line"],
                            "name": meta["name"],
                            "type": meta["type"],
                            "content": doc,
                        }

        return {"temp_dir": temp_dir, "retrieved_context": list(retrieved.values())}

    except Exception as e:
        msg = f"Codebase Search Agent failed: {e}"
        print(f"[ERROR] {msg}")
        # FIX #5: Clean up on failure
        _cleanup_temp(temp_dir)
        return {"errors": errors + [msg], "temp_dir": "", "retrieved_context": []}


def solution_architect_node(state: AgentState) -> dict:
    print("[Agent 3] Formulating code modification plan...")
    errors = state.get("errors", [])
    if not state.get("retrieved_context"):
        return {"errors": errors + ["Cannot proceed to Solution Architect: Retrieved context is empty."]}

    try:
        system_prompt = (
            "You are a Software Architect writing a code-modification plan.\n"
            "Return strict JSON only — no prose, no markdown wrappers.\n"
            "Schema:\n"
            "{\n"
            '  "root_cause": "Detailed analysis referencing specific files and line numbers.",\n'
            '  "recommended_changes": "Step-by-step modifications. Specify exact files and logical changes.",\n'
            '  "estimated_effort": "e.g. Easy (15 mins)"\n'
            "}"
        )

        # Token budget: cap each chunk to avoid exceeding Groq's 6 000 TPM limit.
        # Strategy: truncate each chunk to 600 chars (~150 tokens) and send at most 6 chunks.
        MAX_CHUNK_CHARS = 600
        MAX_CHUNKS = 6

        context_lines = []
        for chunk in state["retrieved_context"][:MAX_CHUNKS]:
            code = chunk["content"]
            if len(code) > MAX_CHUNK_CHARS:
                code = code[:MAX_CHUNK_CHARS] + "\n... [truncated]"
            context_lines.append(
                f"File: {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})\n"
                f"Block: {chunk['name']} ({chunk['type']})\n"
                f"Code:\n{code}\n"
                "---"
            )
        context_str = "\n".join(context_lines)

        user_prompt = (
            f"Issue Analysis:\n{json.dumps(state['issue_analysis'], indent=2)}\n\n"
            f"Retrieved Code:\n{context_str}"
        )

        response_text = call_llm(system_prompt, user_prompt)
        plan = extract_json(response_text)
        return {"implementation_plan": plan}

    except Exception as e:
        msg = f"Solution Architect Agent failed: {e}"
        print(f"[ERROR] {msg}")
        return {"errors": errors + [msg], "implementation_plan": {}}


def pr_creator_node(state: AgentState) -> dict:
    print("[Agent 4] Applying changes and opening PR...")
    errors = state.get("errors", [])
    if not state.get("implementation_plan"):
        return {"errors": errors + ["Cannot proceed to PR Creator: Implementation plan is empty."]}

    temp_dir = state.get("temp_dir", "")
    if not temp_dir or not os.path.exists(temp_dir):
        return {"errors": errors + ["Cloned repository directory not found."]}

    try:
        plan_text = json.dumps(state["implementation_plan"])
        mentioned_files = extract_file_paths_from_text(plan_text)
        mentioned_files = [
            f for f in mentioned_files
            if not any(f.endswith(ext) for ext in (".lock", ".svg", ".png", ".jpg", ".ico"))
        ]

        retrieved_files = [c["file_path"] for c in state["retrieved_context"]]
        candidate_files = mentioned_files if mentioned_files else retrieved_files

        file_contents: dict[str, str] = {}
        new_files: list[str] = []

        for f_path in candidate_files:
            f_norm = f_path.replace("\\", "/").strip()
            if any(f_norm.endswith(ext) for ext in (".git", ".md", ".log", ".env", ".png", ".jpg")):
                continue

            full_path = os.path.join(temp_dir, f_norm)

            # FIX #13: Only use basename fallback when exactly one match exists
            if not os.path.exists(full_path):
                basename = os.path.basename(f_norm)
                matches = []
                for root, _, files in os.walk(temp_dir):
                    for fname in files:
                        if fname == basename:
                            candidate = os.path.join(root, fname)
                            rel = os.path.relpath(candidate, temp_dir)
                            excluded = (".git", "node_modules", "venv", "__pycache__", "dist", "build")
                            if not any(p in rel.split(os.sep) for p in excluded):
                                matches.append(candidate)
                if len(matches) == 1:
                    full_path = matches[0]
                    f_norm = os.path.relpath(full_path, temp_dir).replace("\\", "/")
                elif len(matches) > 1:
                    print(f"[Agent 4] Warning: ambiguous basename '{basename}' — skipping.")
                    continue

            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_contents[f_norm] = f.read()
                except Exception:
                    pass
            else:
                new_files.append(f_norm)

        files_context = ""
        for path, content in file_contents.items():
            files_context += f"\n--- EXISTING FILE: {path} ---\n{content}\n---\n"
        for path in new_files:
            files_context += f"\n--- NEW FILE (CREATE): {path} ---\n[Empty]\n---\n"

        system_prompt = (
            "You are an expert developer. Output code modifications as SEARCH/REPLACE blocks.\n"
            "ONLY output blocks for files that need changes.\n"
            "New file format:\n"
            "FILE: path/to/file\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "[full new file content]\n"
            ">>>>>>> REPLACE\n\n"
            "Edit format:\n"
            "FILE: path/to/file\n"
            "<<<<<<< SEARCH\n"
            "[exact lines to replace — character-perfect including indentation]\n"
            "=======\n"
            "[replacement lines]\n"
            ">>>>>>> REPLACE\n\n"
            "Rules:\n"
            "1. SEARCH must match the target file exactly (indentation, comments, all).\n"
            "2. Make minimal edits only.\n"
            "3. REPLACE block must be valid code — no prose or explanation inside it.\n"
            "4. Output nothing except these blocks."
        )

        user_prompt = (
            f"Issue: {state['issue_description']}\n\n"
            f"Implementation Plan:\n{json.dumps(state['implementation_plan'], indent=2)}\n\n"
            f"Target Files:\n{files_context}"
        )

        max_retries = 2
        result = {"modified_files": [], "errors": []}
        
        for attempt in range(max_retries + 1):
            llm_edits = call_llm(system_prompt, user_prompt)
            print(f"[Agent 4] Applying SEARCH/REPLACE blocks (Attempt {attempt + 1})...")

            result = apply_search_replace(temp_dir, llm_edits)
            
            if result["modified_files"]:
                if result["errors"]:
                    print(f"[Agent 4] Search/replace warnings: {result['errors']}")
                    errors.extend(result["errors"])
                break  # Success!
            else:
                print(f"[Agent 4] Attempt {attempt + 1} failed. Errors: {result['errors']}")
                if attempt < max_retries:
                    # Append feedback for the next attempt
                    user_prompt += (
                        f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                        f"Your previous output resulted in NO modified files.\n"
                        f"Parser errors:\n{json.dumps(result['errors'], indent=2)}\n"
                        "Check your SEARCH blocks for exact matches and ensure the FILE: header is correct. Try again."
                    )

        if not result["modified_files"]:
            raise ValueError(f"No files were successfully modified after {max_retries + 1} attempts. Errors: {result['errors']}")

        # FIX #11: Branch name always has a safe slug + short timestamp suffix
        prob_summary = state["issue_analysis"].get("problem", "code-fix")
        slug = slugify(prob_summary)
        # Short timestamp suffix prevents collisions on repeated runs
        timestamp_suffix = str(int(time.time()))[-5:]
        branch_name = f"fix/{slug}-{timestamp_suffix}"
        commit_message = f"fix: {prob_summary}\n\nGenerated by Gitbo/CodeOnboard."

        print(f"[Agent 4] Creating branch '{branch_name}' and committing...")
        create_branch_and_commit(temp_dir, branch_name, commit_message)

        git_diff = get_git_diff(temp_dir)

        print(f"[Agent 4] Pushing branch '{branch_name}'...")
        push_branch(temp_dir, state["repo_url"], state["github_token"], branch_name)

        print("[Agent 4] Opening GitHub Pull Request...")
        pr_title = f"Fix: {prob_summary}"
        pr_body = (
            f"This PR was opened automatically by **Gitbo** to address:\n\n"
            f"### Issue Description\n{state['issue_description']}\n\n"
            f"### Root Cause Analysis\n{state['implementation_plan'].get('root_cause', 'N/A')}\n\n"
            f"### Recommended Changes\n{state['implementation_plan'].get('recommended_changes', 'N/A')}"
        )
        pr_url = create_pull_request(
            repo_url=state["repo_url"],
            token=state["github_token"],
            branch_name=branch_name,
            title=pr_title,
            body=pr_body,
        )

        print(f"[Agent 4] PR opened: {pr_url}")

        return {
            "pr_result": {"success": True, "pr_url": pr_url, "branch": branch_name},
            "git_diff": git_diff,
            "errors": errors,
        }

    except Exception as e:
        msg = f"PR Creator Agent failed: {e}"
        print(f"[ERROR] {msg}")
        return {"errors": errors + [msg], "git_diff": "", "pr_result": {"success": False}}


def cleanup_node(state: AgentState) -> dict:
    """
    FIX #5 + #8: Dedicated cleanup node that always runs when the pipeline
    exits early due to errors.  Ensures the temp clone is removed even when
    pr_creator_node is never reached.
    """
    _cleanup_temp(state.get("temp_dir", ""))
    return {}


# ── Conditional routing ───────────────────────────────────────────────────────

def _has_errors(state: AgentState) -> bool:
    return bool(state.get("errors"))


def route_after_search(state: AgentState) -> str:
    """FIX #8: Skip architect and PR nodes if search failed."""
    if _has_errors(state) or not state.get("retrieved_context"):
        return "cleanup"
    return "solution_architect"


def route_after_architect(state: AgentState) -> str:
    """FIX #8: Skip PR node if the architect failed."""
    if _has_errors(state) or not state.get("implementation_plan"):
        return "cleanup"
    return "pr_creator"


# ── Graph compilation ─────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("issue_understanding", issue_understanding_node)
    workflow.add_node("codebase_search", codebase_search_node)
    workflow.add_node("solution_architect", solution_architect_node)
    workflow.add_node("pr_creator", pr_creator_node)
    workflow.add_node("cleanup", cleanup_node)

    workflow.set_entry_point("issue_understanding")

    # Issue understanding → codebase search (always)
    workflow.add_edge("issue_understanding", "codebase_search")

    # FIX #8: Conditional edges — route to cleanup on errors, else continue
    workflow.add_conditional_edges(
        "codebase_search",
        route_after_search,
        {"solution_architect": "solution_architect", "cleanup": "cleanup"},
    )
    workflow.add_conditional_edges(
        "solution_architect",
        route_after_architect,
        {"pr_creator": "pr_creator", "cleanup": "cleanup"},
    )

    # PR creator handles its own cleanup, then ends
    workflow.add_edge("pr_creator", END)
    # Cleanup node (early-exit path) ends
    workflow.add_edge("cleanup", END)

    return workflow.compile()


app_graph = build_graph()
