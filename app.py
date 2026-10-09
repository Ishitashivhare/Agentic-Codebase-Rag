import streamlit as st
import os

from src.loader import load_repo
from src.chunker import chunk_code
from src.embeddings import load_embeddings
from src.vectordb import create_vector_db, load_vector_db
from src.rag_pipeline import ask_codebase
from src.github_loader import clone_github_repo, get_repo_info, list_cached_repos, remove_github_cache

# Cache embeddings model in Streamlit
@st.cache_resource
def get_embeddings():
    return load_embeddings()

st.title("Codebase RAG Assistant with Llama")

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "llama3"
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"
if "current_repo" not in st.session_state:
    st.session_state.current_repo = None
if "repo_root" not in st.session_state:
    st.session_state.repo_root = None

# Ollama configuration
with st.sidebar:
    st.header("Configuration")
    st.session_state.ollama_url = st.text_input(
        "Ollama Server URL",
        value=st.session_state.ollama_url,
        help="Default is http://localhost:11434"
    )
    st.session_state.ollama_model = st.text_input(
        "Ollama Model Name",
        value=st.session_state.ollama_model,
        help="Available models: llama3, llama2, mistral, neural-chat, etc."
    )
    st.info("Make sure Ollama is running on your system")

# Use absolute path
repo_path = os.path.join(os.path.dirname(__file__), "data", "repo")

# Create tabs for different indexing options
tab1, tab2, tab3 = st.tabs([" Local Repository", " GitHub Repository", " Manage Cached Index"])

# ==================== TAB 1: Local Repository ====================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        if st.button(" Index Repository"):
            try:
                st.write(f"Loading from: {repo_path}")
                st.write(f"Directory exists: {os.path.exists(repo_path)}")

                # List files in the directory
                if os.path.exists(repo_path):
                    all_files = os.listdir(repo_path)
                    st.write(f"Files in directory: {all_files}")

                files = load_repo(repo_path)
                st.write(f"Files loaded: {len(files)}")

                if not files:
                    st.error(f"No Python files found in {repo_path}")
                else:
                    with st.spinner(f"Processing {len(files)} file(s)..."):
                        chunks = chunk_code(files)
                        st.write(f"Chunks created: {len(chunks)}")

                        if not chunks:
                            st.error("No code chunks created from files.")
                        else:
                            with st.spinner("Loading embeddings..."):
                                embeddings = get_embeddings()

                            with st.spinner("Creating/Loading vector database..."):
                                db = create_vector_db(chunks, embeddings)

                            st.session_state.db = db
                            st.session_state.current_repo = "Local Repository"
                            st.session_state.repo_root = repo_path
                            st.success(f"Repository Indexed! ({len(chunks)} chunks created) ✓")
            except Exception as e:
                st.error(f"Error indexing repository: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        try:
            with st.spinner("Loading cached index..."):
                embeddings = get_embeddings()
                db = load_vector_db(embeddings)

                if db:
                    st.session_state.db = db
                    st.success("Loaded existing vector database! (Much faster) ✓")
                else:
                    st.warning("No cached index found. Please index the repository first.")
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")

# ==================== TAB 2: GitHub Repository ====================
with tab2:
    st.markdown("### Analyze Code from GitHub")
    st.markdown("Enter a GitHub repository URL to clone and analyze its code.")

    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="e.g., https://github.com/torvalds/linux",
        help="Full GitHub URL (e.g., https://github.com/owner/repo)"
    )

    col_github1, col_github2 = st.columns(2)

    with col_github1:
        if st.button("Clone & Index from GitHub"):
            if not github_url:
                st.error("Please enter a GitHub URL")
            else:
                try:
                    with st.spinner("Cloning repository from GitHub..."):
                        repo_path_github = clone_github_repo(github_url)

                    repo_info = get_repo_info(github_url)
                    st.info(f"Repository: {repo_info['owner']}/{repo_info['name']}")

                    with st.spinner(f"Loading Python files from {repo_info['name']}..."):
                        files = load_repo(repo_path_github)
                        st.write(f"Files loaded: {len(files)}")

                    if not files:
                        st.error(f"No Python files found in the repository")
                    else:
                        with st.spinner(f"Processing {len(files)} file(s)..."):
                            chunks = chunk_code(files)
                            st.write(f"Chunks created: {len(chunks)}")

                            if not chunks:
                                st.error("No code chunks created from files.")
                            else:
                                with st.spinner("Loading embeddings..."):
                                    embeddings = get_embeddings()

                                with st.spinner("Creating vector database..."):
                                    db = create_vector_db(chunks, embeddings, use_cache=False)

                                st.session_state.db = db
                                st.session_state.current_repo = repo_info['name']
                                st.session_state.repo_root = repo_path_github
                                st.success(f"Repository Indexed! ({len(chunks)} chunks created) ✓")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# ==================== TAB 3: Manage Cache ====================
with tab3:
    st.markdown("### Manage Cached Repositories")

    cached_repos = list_cached_repos()

    if cached_repos:
        st.write(f"Cached repositories ({len(cached_repos)}):")
        for repo in cached_repos:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{repo}")
            with col2:
                if st.button("Delete", key=f"delete_{repo}"):
                    try:
                        remove_github_cache(repo)
                        st.success(f"Removed {repo}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing {repo}: {str(e)}")
    else:
        st.info("No cached repositories found")

    st.markdown("---")
    if st.button(" Clear All Cache"):
        try:
            remove_github_cache()
            st.success("All cached repositories cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
with st.sidebar:
    st.markdown("---")
    st.subheader(" Advanced Settings")
    # Higher default k for better accuracy on complex codebases
    k_results = st.slider("Number of code chunks to retrieve", 1, 15, 8, help="More chunks = better context (recommended 8-10 for GitHub repos)")

    # Show current loaded repo info
    if "current_repo" in st.session_state and st.session_state.current_repo:
        st.markdown("---")
        st.info(f"📍 Loaded: {st.session_state.current_repo}")

st.markdown("---")
st.subheader("💬 Query Your Code")
query = st.text_input("Ask about the code", placeholder="e.g., 'How does the authentication work?'")

if query:
    if st.session_state.db is None:
        st.warning("⚠️ Please index the repository first by clicking the 'Index Repository' button.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                answer = ask_codebase(
                    query,
                    st.session_state.db,
                    st.session_state.ollama_model,
                    st.session_state.ollama_url,
                    k=k_results,
                    repo_root=st.session_state.repo_root,
                )
                st.markdown("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running with the specified model. Install from https://ollama.ai")
