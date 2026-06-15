<div align="center">

# 🤖 Gitbo

### Agentic AI Coding Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-8A2BE2?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq_API-LLM_Inference-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

*Drop a GitHub issue. Get a Pull Request. Done.*

[Overview](#-overview) · [Architecture](#️-architecture) · [Tech Stack](#️-tech-stack) · [Project Structure](#-project-structure) · [Getting Started](#-getting-started) · [Usage](#-usage)

</div>

---

## 📌 Overview

**Gitbo** is an autonomous, multi-agent coding assistant that closes the loop between a GitHub issue and a verified Pull Request — entirely without human intervention in between.

Given a natural language issue description and a target repository, Gitbo:

1. **Triages** the issue to diagnose root cause and affected components
2. **Semantically searches** the codebase using vector embeddings to surface the most relevant code contexts
3. **Architects** a precise, targeted fix plan grounded in real code
4. **Applies** structured `SEARCH/REPLACE` patches, commits to a new branch, and **opens a PR** on GitHub

Built with a glassmorphic dark-themed developer dashboard that streams the agent workflow in real-time.

---

## 🏗️ Architecture

Gitbo orchestrates a deterministic 4-agent pipeline via **LangGraph**:

```
╔══════════════════════════════════════════════════════════════════════╗
║                        GITBO AGENT PIPELINE                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   [GitHub Issue]                                                     ║
║        │                                                             ║
║        ▼                                                             ║
║  ┌─────────────────────┐                                             ║
║  │  1. Issue Triage    │  ← Diagnoses root cause, identifies         ║
║  │     Agent           │    affected dirs/components                 ║
║  └──────────┬──────────┘                                             ║
║             │                                                        ║
║             ▼                                                        ║
║  ┌─────────────────────┐                                             ║
║  │  2. Semantic Embed  │  ← AST/brace chunking → sentence-           ║
║  │     & Search Agent  │    transformers → ChromaDB vector search    ║
║  └──────────┬──────────┘                                             ║
║             │                                                        ║
║             ▼                                                        ║
║  ┌─────────────────────┐                                             ║
║  │  3. Solution        │  ← Formulates detailed technical fix plan   ║
║  │     Architect Agent │    grounded in retrieved code snippets      ║
║  └──────────┬──────────┘                                             ║
║             │                                                        ║
║             ▼                                                        ║
║  ┌─────────────────────┐                                             ║
║  │  4. PR Creator &    │  ← Applies SEARCH/REPLACE patches,          ║
║  │     Editor Agent    │    commits to branch, opens GitHub PR       ║
║  └──────────┬──────────┘                                             ║
║             │                                                        ║
║             ▼                                                        ║
║   [Verified GitHub Pull Request]                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Agent Responsibilities

| # | Agent | Role |
|---|-------|------|
| 1 | **Issue Triage Agent** | Parses the issue description, identifies the likely root cause, and pinpoints which directories/modules are relevant |
| 2 | **Semantic Embed & Search Agent** | Chunks source code via AST/brace parsing, generates `BAAI/bge-small-en-v1.5` embeddings, indexes in an ephemeral ChromaDB instance, and queries for top-K relevant contexts |
| 3 | **Solution Architect Agent** | Synthesizes retrieved code snippets into a concrete, step-by-step technical fix plan |
| 4 | **PR Creator & Editor Agent** | Applies `<<<<<<< SEARCH / >>>>>>> REPLACE` diffs, commits to a new branch via GitPython/PyGithub, and opens a verified Pull Request |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI · Uvicorn · Starlette |
| **Agentic Orchestration** | LangGraph · LangChain Core |
| **Vector Search** | ChromaDB · Sentence-Transformers (`BAAI/bge-small-en-v1.5`) |
| **GitHub Integration** | GitPython · PyGithub |
| **LLM Inference** | Groq API (Llama / Qwen models) |
| **Frontend** | Vanilla HTML5 · CSS3 (Custom Properties, Glassmorphism) · Vanilla JS |
| **Config & Validation** | Pydantic v2 · python-dotenv |

---

## 📂 Project Structure

```
Gitbo/
├── backend/
│   ├── main.py              # FastAPI app server, route definitions, CORS config
│   ├── agents.py            # LangGraph workflow: node definitions & edge logic
│   └── utils/
│       ├── chunker.py       # Multi-language AST/sliding-window code chunker
│       ├── parser.py        # SEARCH/REPLACE diff parser with fuzzy matching
│       └── git_handler.py   # Git operations & PyGithub PR interface
├── frontend/
│   ├── index.html           # Single-page agentic dashboard
│   ├── style.css            # CSS variables, glassmorphism, responsive layout
│   └── app.js               # API controller, real-time terminal renderer
├── tests/                   # Test suite
├── .gitignore               # Excludes venvs, .env, build caches
├── requirements.txt         # Pinned Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Git** — installed and on your PATH
- **Groq API Key** — free at [console.groq.com](https://console.groq.com)
- **GitHub Personal Access Token (PAT)** — needs `repo` scope (or fine-grained `Contents` + `Pull Requests` write access)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/marbo786/Gitbo.git
cd Gitbo

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env   # or create manually
```

Add the following to your `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

> **CORS Note:** By default, the API allows only `http://localhost:8000` and `http://127.0.0.1:8000`. To expose the frontend on a different origin, set `ALLOWED_ORIGINS=https://yourapp.example.com` in `.env`.

### Running Locally

```bash
# Start the FastAPI backend
python -m uvicorn backend.main:app --port 8000
```

Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🧑‍💻 Usage

1. Open the dashboard at `http://localhost:8000`
2. Enter the **target GitHub repository URL** (e.g. `https://github.com/owner/repo`)
3. Paste the **issue description** you want resolved
4. Provide your **GitHub PAT** (used only to open the PR — never stored)
5. Hit **Run** and watch the 4-agent pipeline execute step-by-step in the real-time terminal

The pipeline will emit each agent's reasoning, the retrieved code contexts, the proposed diff, and — on success — a direct link to the newly opened Pull Request.

---

## 🔐 Security

- Your GitHub PAT is passed per-request and never persisted server-side
- All LLM calls are routed through Groq's API; no source code leaves your machine except what is sent to Groq for inference
- CORS is locked to `localhost` by default; open only what you need via `ALLOWED_ORIGINS`

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change. Please ensure tests pass before submitting.

---

## 👤 Author

**Mohsin** — [@marbo786](https://github.com/marbo786) · [LinkedIn](https://linkedin.com/in/marbo123)

---

<div align="center">

*Built with LangGraph · FastAPI · ChromaDB · Groq*

</div>
