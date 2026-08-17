# DocMind — RAG-Powered PDF Analyzer

🌍 **Live Demo:** https://rag-pdf-analyzer-y95dmjwr7j64pc84236kda.streamlit.app/

> **About this project:** I built this as a basic foundational project during my first year of college. It allowed me to apply the core concepts of Artificial Intelligence that I've been learning — including Large Language Models (LLMs), neural networks, embeddings, API integrations, and Retrieval-Augmented Generation (RAG) pipelines.

A full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and ask natural-language questions about them. Answers are sourced exclusively from the uploaded content, with cited page references for every response.

---

## Features

- **Multi-PDF support** — upload and query across several documents in one session
- **Local embeddings** — sentence-transformers runs fully offline; no embedding API cost
- **Relevance filtering** — if the document doesn't contain the answer, the app says so instead of hallucinating
- **Agentic retry** — on low-relevance retrieval, the query is automatically reformulated and retried once
- **Source attribution** — every answer shows which document and page number it came from, with a similarity score
- **Chat history** — full conversation memory within a browser session
- **Free LLM** — powered by Groq's free-tier models (no credit card needed for basic usage)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│   Sidebar: PDF upload, settings    Main: chat interface      │
├─────────────────────────────────────────────────────────────┤
│  rag/ingest.py      Extract text (pdfplumber → PyPDF2)       │
│                     Chunk text (RecursiveCharacterSplitter)  │
├─────────────────────────────────────────────────────────────┤
│  rag/embed_store.py Embed chunks (all-MiniLM-L6-v2, local)   │
│                     Store vectors in ChromaDB (on disk)      │
├─────────────────────────────────────────────────────────────┤
│  rag/retriever.py   Semantic similarity search                │
│                     Cosine distance threshold filtering       │
├─────────────────────────────────────────────────────────────┤
│  rag/agent.py       Query reformulation + retry loop          │
├─────────────────────────────────────────────────────────────┤
│  rag/llm.py         Groq API call (OpenAI SDK format)         │
│                     Grounded answer generation                │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline — step by step

| Step             | Module                      | What happens                                                                   |
| ----------------- | --------------------------- | ------------------------------------------------------------------------------ |
| 1. Extract        | `ingest.py`                 | PDF pages → raw text (pdfplumber, PyPDF2 fallback)                             |
| 2. Chunk          | `ingest.py`                 | Text → overlapping segments (~700 chars, 100 overlap)                          |
| 3. Embed & store  | `embed_store.py`            | Chunks → 384-dim vectors → ChromaDB on disk                                    |
| 4. Retrieve       | `retriever.py` + `agent.py` | Question → nearest vectors → relevance filter → optional query rewrite + retry |
| 5. Generate       | `llm.py`                    | Retrieved chunks + question → LLM → grounded answer                            |
| 6. Cite           | `app.py`                    | Display answer + source filename, page number, similarity %                    |

---

## Project Structure

```
rag-pdf-analyzer/
├── app.py                  # Streamlit UI + pipeline orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
├── README.md
└── rag/
    ├── __init__.py
    ├── ingest.py           # PDF extraction + text chunking
    ├── embed_store.py      # Embedding + ChromaDB vector store
    ├── retriever.py        # Semantic search + relevance filtering
    ├── llm.py              # Groq API wrapper + query reformulation
    └── agent.py            # Retry loop (reflect + reformulate)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A [Groq](https://console.groq.com) account (free — no credit card required for free-tier models)

### 1. Get a Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up for a free account.
2. Navigate to **API Keys** → **Create API Key**.
3. Copy the key — it starts with `gsk_...`.

Free-tier models have generous rate limits for personal use.

### 2. Clone and set up

```
git clone https://github.com/varunc12502-oss/Rag-Pdf-Analyzer.git
cd Rag-Pdf-Analyzer

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure the API key

```
# Copy the example and fill in your key
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your-actual-key-here
```

### 4. Run the app

```
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

1. Push your project to a **public GitHub repository** (the `.env` file is in `.gitignore` — it will not be committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Click **New app**, select the repo and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add:

```
GROQ_API_KEY = "gsk_your-actual-key-here"
```

5. Click **Deploy**. The app will install dependencies from `requirements.txt` automatically.

> **Note:** The `chroma_store/` folder is excluded from git via `.gitignore`.
> On Streamlit Cloud the vector store is rebuilt from uploads during each session, which is the expected behaviour.

---

## Configuration

All settings are accessible in the sidebar at runtime:

| Setting              | Default                  | Description                                        |
| --------------------- | ------------------------- | ---------------------------------------------------- |
| Model                | `llama3-8b-8192`   | LLM for answer generation and query reformulation   |
| Chunks to retrieve   | 5                          | How many document segments to fetch per question    |
| Relevance threshold  | 0.85                       | Cosine distance cutoff — higher = more permissive    |

### Relevance threshold guide

| Distance   | Similarity | Meaning           |
| ----------- | ----------- | ------------------ |
| < 0.3      | > 70%       | Highly relevant    |
| 0.3 – 0.6  | 40–70%      | Somewhat relevant  |
| > 0.6      | < 40%       | Likely off-topic   |

---

## Key Design Decisions

**Why pdfplumber over PyMuPDF or PyPDF2?**
pdfplumber handles multi-column and styled PDFs better. PyPDF2 is kept as a fallback for edge cases pdfplumber can't open.

**Why sentence-transformers locally?**
Running `all-MiniLM-L6-v2` locally eliminates embedding API costs and latency. The model is 90 MB, downloads once, and is cached by the library.

**Why ChromaDB?**
It's the simplest vector database to set up locally — no Docker, no server. The `PersistentClient` writes to disk automatically and supports the full ANN search API we need.

**Why cosine distance over L2?**
Cosine distance is length-invariant, meaning short and long chunks about the same topic still score as similar. L2 distance penalises length differences, which is undesirable here.

**Why refuse to answer on low relevance?**
RAG systems that always answer tend to hallucinate when context is weak. The explicit relevance gate here prevents that by returning "I don't know" when no chunk clears the threshold.

---

## Limitations

- **Scanned / image PDFs** are not supported (no OCR). Only digitally-created PDFs with embedded text work.
- **Very large PDFs** (500+ pages) may be slow to ingest on first upload, depending on hardware.
- **Free-tier LLM rate limits** on Groq are generous but finite — if you hit them, wait a minute and retry.
- **Session isolation**: each browser session gets its own vector store collection. Refreshing the page starts a new session.

---

## License

MIT
