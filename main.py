import os
import streamlit as st

from ingest import clone_repo, build_index, CHROMA_DIR
from git import GitCommandError
from query import load_index

st.set_page_config(page_title="AI Codebase Chat", page_icon="🤖", layout="wide")
st.title("🤖 AI Codebase Chat")
st.caption("Paste a GitHub repo, then ask questions about how the code works.")

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Load a Repository")
    repo_url = st.text_input(
        "GitHub repo URL", placeholder="https://github.com/user/repo"
    )
    ingest_clicked = st.button("Ingest Repository", type="primary")

    if ingest_clicked and repo_url:
        with st.spinner("Cloning and indexing... this can take a minute for larger repos"):
            try:
                clone_repo(repo_url)
                index = build_index("cloned_repo")
                st.session_state.query_engine = index.as_query_engine(similarity_top_k=8)
                st.session_state.messages = []
                st.success("Repository indexed! Ask away below.")
            except GitCommandError as e:
                error_text = str(e).lower()
                if "authentication" in error_text or "could not read username" in error_text:
                    st.error(
                        "🔒 This looks like a **private repository**. "
                        "This app currently only supports public GitHub repos. "
                        "Make the repo public, or use a different public repo to test."
                    )
                elif "not found" in error_text or "repository not found" in error_text:
                    st.error(
                        "❌ Repository not found. Double-check the URL is correct "
                        "and the repo actually exists."
                    )
                else:
                    st.error(f"❌ Git error while cloning: {e}")
            except Exception as e:
                st.error(f"⚠️ Something went wrong while indexing: {e}")

    st.divider()

    if st.button("Load previously indexed repo"):
        if os.path.exists(CHROMA_DIR):
            with st.spinner("Loading saved index..."):
                index = load_index()
                st.session_state.query_engine = index.as_query_engine(similarity_top_k=8)
                st.success("Loaded saved index.")
        else:
            st.warning("No saved index found yet — ingest a repo first.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.query_engine is None:
    st.info("👈 Enter a GitHub repo URL in the sidebar and click **Ingest Repository** to get started.")
else:
    question = st.chat_input("Ask something about the codebase...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(question)
                st.markdown(str(response))

                # Show which files this answer was actually grounded in
                if response.source_nodes:
                    with st.expander("📄 Sources"):
                        seen_files = set()
                        for node in response.source_nodes:
                            file_path = node.metadata.get("file_path", "Unknown file")
                            if file_path not in seen_files:
                                seen_files.add(file_path)
                                # Show just the filename, not the full clunky local path
                                short_name = os.path.relpath(file_path, "cloned_repo")
                                st.markdown(f"- `{short_name}`")

        st.session_state.messages.append({"role": "assistant", "content": str(response)})