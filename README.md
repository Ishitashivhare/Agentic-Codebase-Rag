# Codebase RAG Assistant (Agentic)

A Retrieval-Augmented Generation assistant for exploring and asking questions
about a codebase, using a local Llama model (via Ollama), FAISS vector search,
and an agentic retrieval pipeline: a **Planner Agent** classifies your
question, an **Agent Orchestrator** runs an iterative retrieval loop against
a **Tool Registry** (semantic code search, file reading, file lookup,
repository structure), checks whether the retrieved evidence actually covers
your question, performs additional targeted retrieval when it doesn't, and
returns an answer with deterministic **file/function/class/line citations**.
The agent also supports exact symbol lookup, AST-based relationship analysis,
bounded caller/callee traversal, concrete method resolution when a method is
unambiguous, evidence sufficiency checks, and citation validation.

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.10 or 3.11** (tested range; 3.12 likely works but is untested) |
| [Ollama](https://ollama.ai/) | Latest — installed and running locally |
| Git | Any recent version (only needed for the "GitHub Repository" tab) |
| A Llama model pulled into Ollama | `llama3` (default) — see step 2 below |

You do **not** need a GPU. Embeddings run on CPU via `sentence-transformers`;
`faiss-cpu` is used for vector search.

---

## 2. Install Ollama and pull a model

Ollama runs the LLM locally — it is a separate install from the Python
dependencies below.

### Windows
1. Download and run the installer from **https://ollama.com/download/windows**.
2. Open a new **Command Prompt** or **PowerShell** window (so Ollama's PATH
   changes take effect) and confirm it's installed:
   ```
   ollama --version
   ```
3. Pull the default model this project expects:
   ```
   ollama pull llama3
   ```
4. Ollama runs as a background service on Windows automatically after
   install. Verify it's reachable:
   ```
   curl http://localhost:11434
   ```
   (If `curl` isn't available, just skip this — the Streamlit app will tell
   you if it can't reach Ollama.)

### Linux / macOS
1. Install:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   (macOS users can alternatively download the `.dmg` from
   https://ollama.com/download/mac.)
2. Start the Ollama server (if it isn't already running as a service):
   ```bash
   ollama serve
   ```
   Leave this running in its own terminal, or let your OS's service manager
   handle it (the Linux installer usually sets this up as a systemd service).
3. In a **second terminal**, pull the model:
   ```bash
   ollama pull llama3
   ```

You can substitute another Ollama-supported model (e.g. `llama3.1`,
`mistral`) — just enter its name in the app's sidebar instead of `llama3`.

---

## 3. Set up the Python environment

### Windows (PowerShell or Command Prompt)
```powershell
cd path\to\codebase-rag-assistant
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
cd path/to/codebase-rag-assistant
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first `pip install` will download `sentence-transformers`' underlying
model weights the *first time you actually use the app* (not during `pip
install` itself) — this requires an internet connection on first run only.

---

## 4. Environment variables

**None are required.** The Ollama server URL (default
`http://localhost:11434`) and model name (default `llama3`) are configured
directly in the Streamlit app's sidebar at runtime, not via environment
variables or a `.env` file. There is nothing to configure before first run.

---

## 5. Run the application

With your virtual environment activated (see step 3):

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501` in your browser. If that port is
already in use, Streamlit selects the next available port and prints the URL.

---

## 6. Load / index a repository

The app has three tabs:

- **Local Repository** — click "Index Repository" to index the codebase
  found at `data/repo/` inside this project folder. Copy or symlink the
  code you want indexed into `data/repo/` first.
- **GitHub Repository** — paste a GitHub URL (e.g.
  `https://github.com/owner/repo`) and click "Clone & Index from GitHub".
  Requires `git` to be installed and on your PATH.
- **Manage Cached Index** — view or clear previously indexed repositories.

Indexing does three things: loads every `.py` file, chunks it along
function/class boundaries (via Python's `ast` module, so citations can point
at exact function/class/line ranges), embeds each chunk, and builds a FAISS
index. This can take anywhere from a few seconds to a few minutes depending
on repository size — there is no GPU requirement, but larger repos will be
noticeably slower on CPU-only embedding.

---

## 7. Example queries

Once a repository is indexed, try questions like:

- *"Where is authentication implemented?"*
- *"What does UserService do?"*
- *"Explain how authentication works in this repository and identify where JWT tokens are generated."*
- *"Explain the repository structure."*
- *"How does module A interact with module B?"*
- *"Find all code related to JWT."*

For compound questions (like the third example above), the orchestrator will
automatically perform more than one retrieval pass if the first pass doesn't
cover every part of the question — you'll see this reflected in longer
response times for multi-part questions, and in a fuller `Source:` citation
list at the end of the answer.

---

## 8. Known Limitations

- **Only Python (`.py`) files are indexed.** Other languages aren't chunked
  or searched.
- **The GitHub tab requires `git`** to be installed and on your system PATH.
  Private repositories aren't supported (no credential/token flow).
- **Coverage checking is heuristic, not semantic.** The orchestrator requires
  every meaningful term in each detected information need to have a matching
  retrieved term, but this is still keyword/root matching rather than true
  language understanding.
- **AST analysis is Python-only.** Exact symbols, imports, callers, callees,
  and class methods are built from parseable `.py` files. Files with syntax
  errors are skipped by the AST index but remain available to text chunking.
- **Dynamic dispatch is limited.** Calls whose concrete receiver cannot be
  identified statically remain unresolved; method resolution succeeds only
  when the available source evidence identifies an unambiguous implementation.
- **Citation validation is structural.** It verifies that cited files and line
  ranges exist and that the source was included in retrieved evidence. It does
  not prove full semantic entailment of every generated claim.
- **No conversation memory.** Each question is answered independently; the
  assistant does not remember earlier questions in the same session. This
  is a deliberate scope boundary, not a bug.
- **Existing indexes built before this citation system will cite by
  filename only** (no function name, class name, or line numbers) until
  re-indexed, since that metadata didn't exist in older indexes.
- **`langchain`/`langchain-community` versions are pinned for compatibility**
  with the current text-splitting and community embedding/FAISS integrations.
  If you upgrade either package, rerun the regression suite and the FAISS
  index/query smoke test before changing import paths.
- **First-time model download**: the embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`) and any Ollama model you pull
  are both downloaded on first use, not during `pip install` — make sure
  you have internet access the first time you index a repo and the first
  time you run `ollama pull`.

---

## 9. Project Structure

```
app.py                          # Streamlit UI
requirements.txt
README.md
src/
├── loader.py                   # Loads .py files, preserving path per file
├── chunker.py                  # AST-aware chunking (function/class boundaries)
├── embeddings.py                # Embedding model loading
├── vectordb.py                  # FAISS build/save/load, with citation metadata
├── citations.py                 # Citation model + deterministic "Source:" formatting
├── github_loader.py             # Clone/cache GitHub repositories
├── rag_pipeline.py              # ask_codebase() - the Streamlit-facing entry point
├── code_index.py                # AST symbols, imports, and call relationships
├── agent/
│   ├── planner.py                # PlannerAgent: classifies queries, builds initial plans
│   └── orchestrator.py           # AgentOrchestrator: the iterative retrieval loop
└── tools/
    ├── base_tool.py               # BaseTool interface + ToolResult
    ├── tool_registry.py           # Tool registration/dispatch
    ├── search_code_tool.py        # Semantic search (wraps FAISS)
    ├── search_symbol_tool.py      # Exact AST symbol lookup
    ├── code_relationship_tool.py  # AST relationships and call traversal
    ├── citation_validator.py      # File, range, and evidence validation
    ├── read_file_tool.py          # Read a file (with path-traversal protection)
    ├── find_file_tool.py          # Locate files by name
    └── repository_structure_tool.py  # Directory/file tree
data/
└── repo/                        # Put the local repository you want to index here
tests/
├── test_code_relationships.py   # AST, traversal, edge-case, and citation tests
├── test_agent_workflow.py       # Planner/orchestrator/answer-generation tests
└── README.md                    # Deterministic vs. Ollama integration testing
```

## 10. Run tests

From the project root, run the deterministic suite:

```powershell
python -m unittest discover -s tests -v
```

The suite uses temporary repositories and mocked LLM/vector-store components;
it does not require Ollama or network access. Manual Ollama integration
requires `ollama serve`, a pulled model, and the Streamlit app running. See
`tests/README.md` for the separate integration setup.

## License

MIT License.
