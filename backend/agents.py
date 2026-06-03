import os
import json
import re
import shutil
import tempfile
import uuid
import chromadb
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END

# Import utilities
from backend.utils.chunker import chunk_file
from backend.utils.git_handler import (
    clone_repo,
    create_branch_and_commit,
    get_git_diff,
    push_branch,
    create_pull_request
)
from backend.utils.parser import apply_search_replace

class AgentState(TypedDict):
    repo_url: str
    github_token: str
    issue_description: str
    issue_analysis: dict         # Filled by Agent 1
    retrieved_context: list      # Filled by Agent 2
    implementation_plan: dict    # Filled by Agent 3
    pr_result: dict              # Filled by Agent 4
    git_diff: str                # Filled by Agent 4
    temp_dir: str                # Filled by Agent 2, used by Agent 4
    errors: list[str]            # Track errors in execution

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # Load local model
        _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedding_model

def clean_invalid_sets(raw_json: str) -> str:
    """Detect and convert curly-brace sets { \"a\", \"b\" } to valid JSON arrays [ \"a\", \"b\" ]."""
    chars = list(raw_json)
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] == '{':
            brace_count = 1
            j = i + 1
            in_quote = False
            has_colon = False
            has_comma = False
            
            while j < n:
                char = chars[j]
                if char == '"' and (j == 0 or chars[j-1] != '\\'):
                    in_quote = not in_quote
                elif not in_quote:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    elif char == ':':
                        has_colon = True
                    elif char == ',':
                        has_comma = True
                j += 1
                
            if brace_count == 0:
                if has_comma and not has_colon:
                    chars[i] = '['
                    chars[j] = ']'
            i = j
        i += 1
    return "".join(chars)

def extract_json(text: str) -> dict:
    """Extract and load JSON from LLM output string, robust against control characters, sets, and comments."""
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text}")
        
    raw_json = match.group(1)
    
    try:
        return json.loads(raw_json, strict=False)
    except Exception as e:
        try:
            # 1. Convert set-like structures e.g. { "a", "b" } to [ "a", "b" ]
            cleaned_json = clean_invalid_sets(raw_json)
            # 2. Remove trailing commas e.g. , } or , ]
            cleaned_json = re.sub(r',\s*([\]}])', r'\1', cleaned_json)
            # 3. Remove single-line JavaScript/JSON comments
            cleaned_json = re.sub(r'^\s*//.*$', '', cleaned_json, flags=re.MULTILINE)
            return json.loads(cleaned_json, strict=False)
        except Exception:
            raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}\nRaw block was: {raw_json}")

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
    from groq import Groq
    return Groq(api_key=api_key)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def extract_file_paths_from_text(text: str) -> list[str]:
    # Match filenames with extension
    pattern = r'\b[a-zA-Z0-9_.\-/]+\.[a-zA-Z0-9]{2,4}\b'
    candidates = re.findall(pattern, text)
    paths = []
    for c in candidates:
        c_clean = c.strip().strip("'\"`()")
        if not re.match(r'^\d+\.\d+$', c_clean) and '.' in c_clean and not c_clean.startswith('.'):
            _, ext = os.path.splitext(c_clean.lower())
            if ext in ('.py', '.js', '.ts', '.tsx', '.css', '.html', '.java', '.cpp', '.h', '.go'):
                paths.append(c_clean)
    return list(set(paths))

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text)
    return text.strip('-')[:30]

# --- AGENT NODES ---

def issue_understanding_node(state: AgentState) -> dict:
    print("[Agent 1] Analyzing issue description...")
    errors = state.get("errors", [])
    try:
        system_prompt = (
            "You are an expert issue triage agent. Your task is to analyze a raw GitHub issue description and extract its core semantic meaning.\n"
            "You must return your analysis in strict JSON format with no additional conversation, explanation, or markdown wrappers. The JSON must match the following schema:\n"
            "{\n"
            '  "problem": "Brief summary of the core bug or feature request.",\n'
            '  "category": "Bug" | "Feature" | "Refactor",\n'
            '  "difficulty": "Easy" | "Medium" | "Hard",\n'
            '  "skills": ["Language1", "Framework1", ...],\n'
            '  "search_queries": [\n'
            '    "distinct semantic query 1 to search in codebase",\n'
            '    "distinct semantic query 2 to search in codebase",\n'
            '    "distinct semantic query 3 to search in codebase"\n'
            "  ]\n"
            "}\n"
            "Ensure the search queries target specific file names, variable names, function signatures, or error messages related to the issue."
        )
        user_prompt = f"Issue Description:\n{state['issue_description']}"
        
        response_text = call_llm(system_prompt, user_prompt)
        analysis = extract_json(response_text)
        return {"issue_analysis": analysis}
    except Exception as e:
        error_msg = f"Issue Understanding Agent failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"errors": errors + [error_msg], "issue_analysis": {}}

def codebase_search_node(state: AgentState) -> dict:
    print("[Agent 2] Searching codebase for relevant files...")
    errors = state.get("errors", [])
    if not state.get("issue_analysis"):
        return {"errors": errors + ["Cannot proceed to codebase search: Issue analysis is empty."]}
        
    repo_url = state["repo_url"]
    token = state["github_token"]
    queries = state["issue_analysis"].get("search_queries", [])
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="gitbo_clone_")
        print(f"[Agent 2] Cloned repo to temporary directory: {temp_dir}")
        clone_repo(repo_url, token, temp_dir)
        
        # Scan and chunk files
        code_extensions = ('.py', '.js', '.ts', '.java', '.cpp', '.h', '.go')
        exclude_dirs = ('.git', 'node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build')
        
        all_chunks = []
        for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext in code_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    chunks = chunk_file(full_path)
                    for chunk in chunks:
                        chunk["file_path"] = rel_path.replace("\\", "/") # normalize to forward slashes
                        all_chunks.append(chunk)
                        
        if not all_chunks:
            return {"temp_dir": temp_dir, "retrieved_context": []}
            
        print(f"[Agent 2] Generating embeddings for {len(all_chunks)} chunks...")
        model = get_embedding_model()
        contents = [c["content"] for c in all_chunks]
        embeddings = model.encode(contents, show_progress_bar=False).tolist()
        
        # Write chunks to ephemeral ChromaDB
        client = chromadb.EphemeralClient()
        collection_name = f"repo_{uuid.uuid4().hex[:16]}"
        collection = client.create_collection(name=collection_name)
        
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        metadatas = [{
            "file_path": c["file_path"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "name": c["name"],
            "type": c["type"]
        } for c in all_chunks]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        # Search collections for each query
        retrieved = {}
        for q in queries:
            print(f"[Agent 2] Executing semantic query: '{q}'")
            q_emb = model.encode([q]).tolist()[0]
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=4
            )
            if results and results['documents']:
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                for doc, meta in zip(docs, metas):
                    key = (meta["file_path"], meta["start_line"], meta["end_line"])
                    if key not in retrieved:
                        retrieved[key] = {
                            "file_path": meta["file_path"],
                            "start_line": meta["start_line"],
                            "end_line": meta["end_line"],
                            "name": meta["name"],
                            "type": meta["type"],
                            "content": doc
                        }
                        
        return {"temp_dir": temp_dir, "retrieved_context": list(retrieved.values())}
    except Exception as e:
        error_msg = f"Codebase Search Agent failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"errors": errors + [error_msg], "retrieved_context": []}

def solution_architect_node(state: AgentState) -> dict:
    print("[Agent 3] Formulating code modification plan...")
    errors = state.get("errors", [])
    if not state.get("retrieved_context"):
        return {"errors": errors + ["Cannot proceed to Solution Architect: Retrieved context is empty."]}
        
    try:
        system_prompt = (
            "You are a Software Architect designing a code modification plan. You will receive an issue analysis and relevant code snippets from the codebase.\n"
            "Analyze the context and write a clear implementation plan.\n"
            "You must return your design in strict JSON format with no additional conversation, explanation, or markdown wrappers. The JSON must match the following schema:\n"
            "{\n"
            '  "root_cause": "Detailed analysis of why the bug is happening, referencing specific files and line numbers.",\n'
            '  "recommended_changes": "Step-by-step description of modifications to make. Specify the exact files to edit and the exact logical changes.",\n'
            '  "estimated_effort": "Timelines/difficulty rating, e.g. Easy (15 mins)"\n'
            "}"
        )
        
        # Format code context for user prompt
        context_lines = []
        for chunk in state["retrieved_context"]:
            context_lines.append(
                f"File: {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})\n"
                f"Block Name: {chunk['name']} ({chunk['type']})\n"
                f"Code:\n{chunk['content']}\n"
                "----------------------------------------"
            )
        context_str = "\n".join(context_lines)
        
        user_prompt = (
            f"Issue Analysis:\n{json.dumps(state['issue_analysis'], indent=2)}\n\n"
            f"Retrieved Code Context:\n{context_str}"
        )
        
        response_text = call_llm(system_prompt, user_prompt)
        plan = extract_json(response_text)
        return {"implementation_plan": plan}
    except Exception as e:
        error_msg = f"Solution Architect Agent failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"errors": errors + [error_msg], "implementation_plan": {}}

def pr_creator_node(state: AgentState) -> dict:
    print("[Agent 4] Applying changes and opening PR...")
    errors = state.get("errors", [])
    if not state.get("implementation_plan"):
        return {"errors": errors + ["Cannot proceed to PR Creator: Implementation plan is empty."]}
        
    temp_dir = state.get("temp_dir")
    if not temp_dir or not os.path.exists(temp_dir):
        return {"errors": errors + ["Cloned repository directory not found."]}
        
    try:
        # 1. Read full file content of files in retrieved context and implementation plan
        plan_text = json.dumps(state["implementation_plan"])
        mentioned_files = extract_file_paths_from_text(plan_text)
        
        # Filter out non-code or heavy formats (like package-lock.json or assets)
        mentioned_files = [f for f in mentioned_files if not any(f.endswith(ext) for ext in ('.json', '.lock', '.svg', '.png', '.jpg', '.jpeg', '.md', '.ico'))]
        
        retrieved_files = [c["file_path"] for c in state["retrieved_context"]]
        all_candidate_files = mentioned_files if mentioned_files else retrieved_files
        
        file_contents = {}
        new_files = []
        
        for f_path in all_candidate_files:
            f_path_norm = f_path.replace("\\", "/").strip()
            if any(f_path_norm.endswith(ext) for ext in ('.git', '.md', '.log', '.env', '.png', '.jpg', '.json')):
                continue
                
            full_path = os.path.join(temp_dir, f_path_norm)
            
            # If not found directly, try to search recursively for the basename
            if not os.path.exists(full_path):
                found = False
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file == os.path.basename(f_path_norm):
                            possible_path = os.path.join(root, file)
                            possible_rel = os.path.relpath(possible_path, temp_dir)
                            if not any(part in possible_rel.split(os.sep) for part in ('.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build')):
                                full_path = possible_path
                                f_path_norm = possible_rel.replace("\\", "/")
                                found = True
                                break
                    if found:
                        break
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_contents[f_path_norm] = f.read()
                except Exception:
                    pass
            else:
                new_files.append(f_path_norm)
                
        files_context = ""
        for path, content in file_contents.items():
            files_context += f"\n--- EXISTING FILE: {path} ---\n{content}\n---------------------\n"
        for path in new_files:
            files_context += f"\n--- NEW FILE (DOES NOT EXIST YET, CREATE IT): {path} ---\n[Empty File]\n---------------------\n"
            
        # 2. Call LLM to output Search/Replace blocks
        system_prompt = (
            "You are an expert developer assistant. Your task is to output code modifications to resolve the issue using SEARCH/REPLACE blocks.\n"
            "You will be given:\n"
            "1. The issue analysis and description.\n"
            "2. The architect's recommended changes.\n"
            "3. The full content of the target files.\n\n"
            "ONLY output blocks for files that actually require changes. DO NOT output any blocks for files that do not require modifications. If a file does not need edits, DO NOT list it.\n"
            "DO NOT output placeholder search blocks like '/* existing styles */' or comments saying 'No changes needed'.\n"
            "If a new file needs to be created, write an empty SEARCH block, like this:\n\n"
            "FILE: path/to/new_file.css\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "[new file content]\n"
            ">>>>>>> REPLACE\n\n"
            "Format your response exactly like this:\n\n"
            "FILE: path/to/file.py\n"
            "<<<<<<< SEARCH\n"
            "[exact lines of code to modify]\n"
            "=======\n"
            "[modified lines of code]\n"
            ">>>>>>> REPLACE\n\n"
            "Rules:\n"
            "1. Output the relative file path after 'FILE: ' (e.g. FILE: src/main.py).\n"
            "2. The SEARCH block must match the lines of code in the target file EXACTLY, character-for-character including indentation and comments.\n"
            "3. Make minimal, clean edits to resolve the issue. Do not rewrite the whole file.\n"
            "4. The REPLACE block must contain ONLY the raw modified code lines. DO NOT write explanations, prose, conversational notes, markdown wrappers, or instructions inside the REPLACE block. Any content inside the REPLACE block will be directly written into the source code, so it must be valid syntax.\n"
            "5. Output nothing else but these blocks. Do not add conversational text."
        )
        
        user_prompt = (
            f"Issue: {state['issue_description']}\n\n"
            f"Implementation Plan:\n{json.dumps(state['implementation_plan'], indent=2)}\n\n"
            f"Target Files Content:\n{files_context}"
        )
        
        llm_edits = call_llm(system_prompt, user_prompt)
        print(f"[Agent 4] Applying SEARCH/REPLACE blocks output...")
        
        # 3. Apply search/replace on local cloned files
        result = apply_search_replace(temp_dir, llm_edits)
        if result["errors"]:
            print(f"[Agent 4] Warning search/replace errors: {result['errors']}")
            errors.extend(result["errors"])
            
        if not result["modified_files"]:
            raise ValueError(f"No files were successfully modified. Errors: {', '.join(result['errors'])}")
            
        # 4. Check out new branch
        prob_summary = state["issue_analysis"].get("problem", "code-fix")
        branch_name = f"fix/{slugify(prob_summary)}"
        commit_message = f"fix: {prob_summary}\n\nGenerated by CodeOnboard."
        
        print(f"[Agent 4] Creating branch '{branch_name}' and committing changes...")
        create_branch_and_commit(temp_dir, branch_name, commit_message)
        
        # Capture diff
        git_diff = get_git_diff(temp_dir)
        
        # 5. Push branch
        print(f"[Agent 4] Pushing branch '{branch_name}' to remote...")
        push_branch(temp_dir, state["repo_url"], state["github_token"], branch_name)
        
        # 6. Create PR via PyGithub
        print(f"[Agent 4] Creating GitHub Pull Request...")
        pr_title = f"Fix: {prob_summary}"
        pr_body = (
            f"This PR was opened automatically by **Gitbo** to address the following issue:\n\n"
            f"### Issue Description\n{state['issue_description']}\n\n"
            f"### Root Cause Analysis\n{state['implementation_plan'].get('root_cause', 'N/A')}\n\n"
            f"### Recommended Changes\n{state['implementation_plan'].get('recommended_changes', 'N/A')}"
        )
        pr_url = create_pull_request(
            repo_url=state["repo_url"],
            token=state["github_token"],
            branch_name=branch_name,
            title=pr_title,
            body=pr_body
        )
        
        print(f"[Agent 4] Success! PR opened at: {pr_url}")
        
        # Cleanup cloned directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        return {
            "pr_result": {
                "success": True,
                "pr_url": pr_url,
                "branch": branch_name
            },
            "git_diff": git_diff,
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"PR Creator Agent failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
                
        return {"errors": errors + [error_msg], "git_diff": "", "pr_result": {"success": False}}

# --- LANGGRAPH GRAPH COMPILATION ---

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    workflow.add_node("issue_understanding", issue_understanding_node)
    workflow.add_node("codebase_search", codebase_search_node)
    workflow.add_node("solution_architect", solution_architect_node)
    workflow.add_node("pr_creator", pr_creator_node)
    
    workflow.set_entry_point("issue_understanding")
    workflow.add_edge("issue_understanding", "codebase_search")
    workflow.add_edge("codebase_search", "solution_architect")
    workflow.add_edge("solution_architect", "pr_creator")
    workflow.add_edge("pr_creator", END)
    
    return workflow.compile()

app_graph = build_graph()
