# Tests

Run the deterministic regression suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

These tests use temporary repositories, deterministic AST fixtures, a fake
vector store, and a mocked LLM. They do not require Ollama, network access,
or model downloads.

Optional manual integration verification requires Ollama to be running with
the configured model (normally `llama3`):

```powershell
ollama serve
ollama pull llama3
streamlit run app.py
```

Then index a repository in the Streamlit UI and run a question that requires
semantic retrieval plus AST relationship traversal. Ollama integration is
kept separate from the normal regression suite because it depends on a local
service and model availability.
