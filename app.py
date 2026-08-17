import os
import uuid
import logging
import requests
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from rag.ingest import extract_pages, chunk_pages
from rag.embed_store import (
    get_chroma_collection,
    add_documents_to_store,
    clear_collection,
    get_collection_count,
)
from rag.agent import retrieve_with_fallback
from rag.llm import get_llm_client, generate_answer, DEFAULT_MODEL

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── st.set_page_config must be the very first Streamlit call ──────────────
st.set_page_config(
    page_title="DocMind — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHROMA_PERSIST_DIR = "./chroma_store"


def get_available_models(api_key: str):
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if r.status_code == 200:
            models = [
                x["id"] for x in r.json().get("data", [])
                if "whisper" not in x["id"]
            ]
            if models:
                return sorted(models)
    except Exception:
        pass
    return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
        }
        [data-testid="stExpander"] summary {
            font-size: 0.85rem;
            color: #666;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "collection" not in st.session_state:
        st.session_state.collection = None


init_session_state()


@st.cache_resource(show_spinner=False)
def _cached_llm_client(api_key: str) -> object:
    return get_llm_client(api_key)


with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.caption("Upload PDFs · Ask questions · Get cited answers")
    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "**API key not configured.**\n\n"
            "Create a `.env` file with:\n"
            "```\nGROQ_API_KEY=your_key_here\n```\n\n"
            "Or add it to Streamlit secrets when deploying to the cloud.",
        )
        st.stop()

    st.subheader("⚙️ Settings")

    available_models = get_available_models(api_key)

    selected_model = st.selectbox(
        "Model",
        options=available_models,
        index=0,
        help="Models are fetched live from Groq — always up to date.",
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=10, value=5,
        help="How many document chunks to fetch per question. More = broader context.",
    )

    distance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.50, max_value=1.00, value=0.85, step=0.05,
        help=(
            "Cosine distance cutoff. Chunks more distant than this are ignored. "
            "Lower values are stricter (fewer but more relevant chunks)."
        ),
    )

    st.divider()

    st.subheader("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="You can upload multiple PDFs at once.",
    )

    if uploaded_files:
        if st.session_state.collection is None:
            collection_name = f"docs_{st.session_state.session_id}"
            with st.spinner("Initialising vector store…"):
                st.session_state.collection = get_chroma_collection(
                    persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=collection_name,
                )

        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]

        for uploaded_file in new_files:
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                try:
                    pages = extract_pages(uploaded_file)
                    chunks = chunk_pages(pages)
                    stored_count = add_documents_to_store(
                        collection=st.session_state.collection,
                        documents=chunks,
                    )
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(
                        f"✅ **{uploaded_file.name}**  \n"
                        f"{len(pages)} pages → {stored_count} chunks indexed."
                    )
                except ValueError as e:
                    st.error(f"❌ **{uploaded_file.name}**: {e}")
                except Exception as e:
                    st.error(
                        f"❌ Unexpected error processing **{uploaded_file.name}**:  \n{e}"
                    )

    if st.session_state.processed_files:
        st.divider()
        st.subheader("📑 Indexed Documents")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(f"- {fname}")

        if st.session_state.collection:
            chunk_count = get_collection_count(st.session_state.collection)
            st.caption(f"Vector store: **{chunk_count}** chunks total")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Erase chat history"):
            st.session_state.chat_history = []
            st.rerun()

    with col_b:
        if st.button(
            "📂 Clear Docs",
            use_container_width=True,
            help="Remove all uploaded documents and reset the vector store",
        ):
            if st.session_state.collection is not None:
                clear_collection(
                    persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=f"docs_{st.session_state.session_id}",
                )
                st.session_state.collection = None
            st.session_state.processed_files = set()
            st.session_state.chat_history = []
            st.rerun()


st.markdown("## 📄 DocMind — Document Q&A")
st.caption(
    "Ask anything about your uploaded documents. "
    "Answers are sourced exclusively from the document content."
)

if not st.session_state.processed_files:
    st.info(
        "👈  **Upload one or more PDF files** from the sidebar to get started.",
        icon="📂",
    )
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View sources", expanded=False):
                for src in message["sources"]:
                    similarity = round((1 - src["distance"]) * 100, 1)
                    st.markdown(
                        f"**{src['source']}** — Page {src['page']} "
                        f"&nbsp;·&nbsp; *{similarity}% similarity*"
                    )
                    st.caption(
                        src["text"][:320] + ("…" if len(src["text"]) > 320 else "")
                    )
                    st.divider()

user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.empty()
        output = st.empty()

        try:
            llm_client = _cached_llm_client(api_key)

            status.caption("🔍 Searching your documents…")

            chunks, was_reformulated, reformulated_query = retrieve_with_fallback(
                collection=st.session_state.collection,
                client=llm_client,
                query=user_input,
                top_k=top_k,
                distance_threshold=distance_threshold,
                model=selected_model,
            )

            reformulation_note = ""
            if was_reformulated and reformulated_query:
                reformulation_note = (
                    f"\n\n---\n> 🔄 *Search was retried with reformulated query:  \n"
                    f"> \"{reformulated_query}\"*"
                )

            if not chunks:
                status.empty()
                response_text = (
                    "I wasn't able to find relevant information in your uploaded "
                    "documents to answer this question.\n\n"
                    "**Suggestions:**\n"
                    "- Try rephrasing your question with different terminology.\n"
                    "- Lower the *Relevance threshold* in the sidebar for broader matching.\n"
                    "- Confirm that the topic is actually covered in the uploaded PDFs."
                )
                if reformulation_note:
                    response_text += reformulation_note

                output.warning(response_text)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response_text, "sources": []}
                )

            else:
                status.caption("💬 Generating answer…")
                answer = generate_answer(
                    client=llm_client,
                    question=user_input,
                    context_chunks=chunks,
                    model=selected_model,
                )
                status.empty()

                full_response = answer + reformulation_note
                output.markdown(full_response)

                with st.expander("📚 View sources", expanded=False):
                    for src in chunks:
                        similarity = round((1 - src["distance"]) * 100, 1)
                        st.markdown(
                            f"**{src['source']}** — Page {src['page']} "
                            f"&nbsp;·&nbsp; *{similarity}% similarity*"
                        )
                        st.caption(
                            src["text"][:320] + ("…" if len(src["text"]) > 320 else "")
                        )
                        st.divider()

                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": full_response,
                    "sources": chunks,
                })

        except RuntimeError as e:
            status.empty()
            err_msg = f"⚠️ {e}"
            output.error(err_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": err_msg, "sources": []}
            )

        except Exception as e:
            status.empty()
            err_msg = f"⚠️ An unexpected error occurred: {e}"
            output.error(err_msg)
            logger.exception("Unhandled error in chat response loop.")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": err_msg, "sources": []}
            )
