"""
app.py — DocMind: RAG-Powered PDF Analyzer
============================================
Main Streamlit application. This file wires together all the pipeline
modules defined in the rag/ package and provides the interactive UI.

Full pipeline flow:
    [User uploads PDF]
        → ingest.extract_pages()        # extract text per page
        → ingest.chunk_pages()          # split into overlapping chunks
        → embed_store.add_documents()   # embed + store in ChromaDB
    
    [User asks a question]
        → agent.retrieve_with_fallback()  # semantic search + retry
        → llm.generate_answer()           # grounded LLM response
        → display answer + source cites

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    - Push to GitHub, connect the repo in share.streamlit.io
    - Add OPENROUTER_API_KEY to the app's Secrets settings
"""

import os
import uuid
import logging
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# RAG pipeline modules
from rag.ingest import extract_pages, chunk_pages
from rag.embed_store import (
    get_chroma_collection,
    add_documents_to_store,
    clear_collection,
    get_collection_count,
)
from rag.agent import retrieve_with_fallback
from rag.llm import get_llm_client, generate_answer, DEFAULT_MODEL

# Load .env file when running locally (no-op on Streamlit Cloud)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_store"

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ═══════════════════════════════════════════════════════════════════════════
# Page config — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocMind — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS — keeps the UI clean without a heavy framework
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
        /* Reduce top padding on main content area */
        .block-container { padding-top: 1.5rem; }

        /* Soften the chat message bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.25rem 0.5rem;
        }

        /* Make the source expanders less prominent */
        [data-testid="stExpander"] summary {
            font-size: 0.85rem;
            color: #666;
        }

        /* Sidebar dividers */
        [data-testid="stSidebar"] hr {
            margin: 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# API key helper
# ═══════════════════════════════════════════════════════════════════════════
def get_api_key() -> Optional[str]:
    """
    Read the Groq API key from Streamlit secrets (cloud deployment)
    or from the GROQ_API_KEY environment variable (local development).

    Streamlit secrets are configured in the app dashboard on
    share.streamlit.io and are accessible via st.secrets at runtime.
    """
    # Streamlit Cloud path: secrets configured in the dashboard
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Local development path: key in .env file loaded by python-dotenv above
    return os.getenv("GROQ_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════
def init_session_state() -> None:
    """
    Set up all st.session_state keys on the first Streamlit run.

    session_state persists values across reruns within the same browser
    session, allowing us to keep chat history and the list of processed
    files without re-running the full ingestion pipeline on every click.
    """
    if "session_id" not in st.session_state:
        # Unique ID per browser session → used as the ChromaDB collection name
        # so different users on Streamlit Cloud don't share a vector store.
        st.session_state.session_id = uuid.uuid4().hex[:12]

    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "sources": list}
        st.session_state.chat_history = []

    if "processed_files" not in st.session_state:
        # Track which filenames have already been embedded this session
        # so we don't re-embed the same PDF if the user re-uploads it.
        st.session_state.processed_files = set()

    if "collection" not in st.session_state:
        # ChromaDB Collection object — lazily initialised on first upload
        st.session_state.collection = None


init_session_state()


# ═══════════════════════════════════════════════════════════════════════════
# Cached resource loaders
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _cached_llm_client(api_key: str) -> object:
    """
    Cache the OpenAI/Groq client across reruns.
    @st.cache_resource persists the object for the lifetime of the app server.
    """
    return get_llm_client(api_key)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.caption("Upload PDFs · Ask questions · Get cited answers")
    st.divider()

    # ── API key validation ────────────────────────────────────────────────
    api_key = get_api_key()
    if not api_key:
        st.error(
            "**API key not configured.**\n\n"
            "Create a `.env` file with:\n"
            "```\nGROQ_API_KEY=your_key_here\n```\n\n"
            "Or add it to Streamlit secrets when deploying to the cloud.",
        )
        st.stop()

    # ── Settings ──────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    selected_model = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=0,
        help="All listed models are available on Groq's free tier.",
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=10, value=5,
        help="How many document chunks to fetch per question. More = broader context.",
    )

    distance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.10, max_value=1.00, value=0.60, step=0.05,
        help=(
            "Cosine distance cutoff. Chunks more distant than this are ignored. "
            "Lower values are stricter (fewer but more relevant chunks)."
        ),
    )

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────────
    st.subheader("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="You can upload multiple PDFs at once.",
    )

    # ── Process newly uploaded files ──────────────────────────────────────
    if uploaded_files:
        # Lazily initialise the ChromaDB collection on first upload
        if st.session_state.collection is None:
            collection_name = f"docs_{st.session_state.session_id}"
            with st.spinner("Initialising vector store…"):
                st.session_state.collection = get_chroma_collection(
                    persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=collection_name,
                )

        # Only process files that haven't been indexed in this session
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]

        for uploaded_file in new_files:
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                try:
                    # ── RAG ingestion pipeline (Steps 1-3) ─────────────────
                    # Step 1: Extract text from each page of the PDF
                    pages = extract_pages(uploaded_file)

                    # Step 2: Split page text into overlapping chunks
                    chunks = chunk_pages(pages)

                    # Step 3: Embed chunks and store in ChromaDB
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
                    # Known user-facing errors: empty file, image-only PDF, etc.
                    st.error(f"❌ **{uploaded_file.name}**: {e}")
                except Exception as e:
                    st.error(
                        f"❌ Unexpected error processing **{uploaded_file.name}**:  \n{e}"
                    )

    # ── Indexed document list ─────────────────────────────────────────────
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📑 Indexed Documents")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(f"- {fname}")

        if st.session_state.collection:
            chunk_count = get_collection_count(st.session_state.collection)
            st.caption(f"Vector store: **{chunk_count}** chunks total")

    # ── Action buttons ─────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# Main chat area
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## 📄 DocMind — Document Q&A")
st.caption(
    "Ask anything about your uploaded documents. "
    "Answers are sourced exclusively from the document content."
)

# Gate: no documents uploaded yet
if not st.session_state.processed_files:
    st.info(
        "👈  **Upload one or more PDF files** from the sidebar to get started.",
        icon="📂",
    )
    st.stop()

# ── Render existing chat history ───────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # For assistant messages, show a collapsible source panel
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

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    # Store and display the user's message immediately
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        status = st.empty()   # status indicator (replaced once we have an answer)
        output = st.empty()   # answer placeholder

        try:
            llm_client = _cached_llm_client(api_key)

            # ── Step 4: Agentic retrieval ──────────────────────────────────
            # retrieve_with_fallback first tries the original query.
            # If that returns nothing relevant, it reformulates the query
            # using the LLM and retries once.
            status.caption("🔍 Searching your documents…")

            chunks, was_reformulated, reformulated_query = retrieve_with_fallback(
                collection=st.session_state.collection,
                client=llm_client,
                query=user_input,
                top_k=top_k,
                distance_threshold=distance_threshold,
                model=selected_model,
            )

            # Build a note to show if the query was rewritten by the agent
            reformulation_note = ""
            if was_reformulated and reformulated_query:
                reformulation_note = (
                    f"\n\n---\n> 🔄 *Search was retried with reformulated query:  \n"
                    f"> \"{reformulated_query}\"*"
                )

            # ── No relevant context found ──────────────────────────────────
            # Rather than letting the LLM hallucinate from thin air, we
            # explicitly refuse to answer and tell the user why.
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
                # ── Step 5: Generate a grounded answer ────────────────────
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

                # ── Step 6: Source attribution ─────────────────────────────
                # Show which chunks were used, with similarity scores and
                # a preview of the text — so the user can verify the answer.
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
            # Known, user-facing errors from llm.py (rate limits, network, etc.)
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
