# Lawgic AI (TrustRAG)

**A trustworthy, multi-domain Retrieval-Augmented Generation (RAG) system for Indian legal Q&A.**

Lawgic AI answers legal questions only from retrieved source text, checks every citation it generates against what was actually retrieved, and refuses to answer when the retrieved context doesn't support one. It covers three legal domains and can also answer questions about any PDF you attach.

| Domain | Source dataset | Indexed chunks |
|---|---|---|
| Consumer Grievances | [GrahakNyay](https://github.com/ShreyGanatra/GrahakNyay) | 1,073 |
| Bail Adjudication | IndianBailJudgments-1200 (Hugging Face) | 2,021 |
| Matrimonial Litigation | IMLJD (Hugging Face, metadata-only) | 3,423 |

---

## Key Features

- **Hybrid retrieval**: BM25 keyword search plus semantic search with `law-ai/InLegalBERT` embeddings (ChromaDB), fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reorders candidates before generation.
- **Citation-enforced generation**: Groq API (`openai/gpt-oss-20b`), prompted to answer only from retrieved context.
- **Citation verification**: every generated citation is checked against the retrieved chunks, and unsupported claims are flagged.
- **Attached-PDF Q&A (page-aware)**: attach any text-based PDF and ask for a specific page ("explain X from page 4"), a full-document summary, or a general question. Sessions are held in memory and never touch the permanent domain index.
- **LoRA query rewriter**: a `flan-t5-small` adapter that rewrites casual phrasing into legal terminology before retrieval (optional, toggled per request).
- **Observability dashboard**: Streamlit dashboard showing latency, grounding rate, and query volume from a live interaction log.
- **Evaluation benchmark**: scripted gold-question set scoring retrieval hit rate and citation grounding.
- **Isolated per-domain chat UI**: each domain has its own thread and its own attached document.

---

## Architecture

```
User query (optionally with an attached PDF)
        |
        v
[optional] LoRA query rewriter (casual -> legal phrasing)
        |
        v
Hybrid retrieval: BM25 + InLegalBERT semantic search (RRF fusion)
        |
        v
Cross-encoder reranking
        |
        v
Citation-enforced generation (Groq, openai/gpt-oss-20b)
        |
        v
Citation verification
        |
        v
Answer + citations + grounding status  ->  interaction_log.csv  ->  dashboard
```

Attached PDFs follow a parallel path (`/process-pdf` then `/ask-doc`): page-aware extraction, in-memory per-document index, and three auto-detected query modes (page-specific, summarize, general).

---

## Project Structure

```
trustrag_multi/
├── backend/
│   └── main.py                  # FastAPI app: retrieval, generation, verification, PDF Q&A
├── frontend/
│   └── index.html               # Single-file chat UI (three isolated case files)
├── data/
│   ├── chroma_db/               # Persistent ChromaDB vector store (generated, not committed)
│   ├── interaction_log.csv      # Per-query telemetry (generated, not committed)
│   └── rewrite_training_data.json  # 48 (casual -> legal) pairs for LoRA training
├── models/
│   └── legal_query_rewriter_lora/  # Trained LoRA adapter (generated)
├── ingest.py                    # Multi-domain ingestion into ChromaDB
├── evaluate.py                  # Evaluation benchmark
├── dashboard.py                 # Streamlit observability dashboard
├── finetune_lora.py             # LoRA training script for the query rewriter
├── test_rewriter.py             # Standalone rewriter sanity check
└── TrustRAG_multidomain_ingestion.ipynb  # Notebook version of ingestion
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (developed on 3.13)
- A [Groq](https://console.groq.com) API key
- ~2 GB free disk for model downloads (InLegalBERT, cross-encoder, flan-t5-small)

### 1. Install

```bash
git clone <your-repo-url>
cd trustrag_multi

python3 -m venv venv
source venv/bin/activate

pip install fastapi uvicorn python-multipart chromadb sentence-transformers rank_bm25 \
            groq pymupdf numpy pandas torch transformers peft accelerate datasets streamlit
```

### 2. Set your API key

```bash
export GROQ_API_KEY="your-key-here"
```

> **Never commit your API key.** Keep it in an environment variable (or a `.env` file listed in `.gitignore`), not in source files.

### 3. Build the index

`ingest.py` loads the three datasets and writes them to `data/chroma_db`. The Consumer domain reads from a local clone of GrahakNyay:

```bash
git clone https://github.com/ShreyGanatra/GrahakNyay.git data/raw_consumer
python ingest.py
```

The Bail and Matrimonial datasets are pulled from Hugging Face automatically. The first run downloads the embedding model.

### 4. (Optional) Train the query rewriter

```bash
python finetune_lora.py      # trains and saves the adapter to models/legal_query_rewriter_lora
python test_rewriter.py      # prints sample rewrites
```

If no adapter is present, the backend starts normally and falls back to raw queries.

### 5. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Wait for `Application startup complete` and the per-domain chunk counts. Check it:

```bash
curl http://localhost:8000/health
```

### 6. Open the frontend

Open `frontend/index.html` in Chrome. It talks to `http://localhost:8000` (change `API_BASE` in the script block to point elsewhere).

### 7. (Optional) Dashboard and evaluation

```bash
streamlit run dashboard.py   # run from the project root
python evaluate.py           # run from the project root
```

> `evaluate.py` defines `CHROMA_DIR` near the top. Update it if your project lives somewhere other than the path it was written with.

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Index status and per-domain chunk counts |
| `POST` | `/ask` | Domain Q&A. Body: `{"query", "domain", "use_rewriter"}` |
| `POST` | `/upload` | Permanently index a PDF into a domain (multipart: `domain`, `file`) |
| `POST` | `/process-pdf` | Load a PDF into an in-memory session, returns `doc_id` |
| `POST` | `/ask-doc` | Ask about an attached PDF. Body: `{"doc_id", "query"}` |
| `POST` | `/ask-pdf` | One-shot: upload a PDF and ask in a single request |

`domain` is one of `consumer`, `bail`, `matrimonial`.

Example:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "Can I get a refund if the product was delivered late?", "domain": "consumer"}'
```

Response shape:

```json
{
  "query": "...",
  "domain": "consumer",
  "answer": "... [rag_qa]",
  "sources": ["rag_qa", "..."],
  "verification": {
    "total_citations_in_answer": 1,
    "verified": ["rag_qa"],
    "unverified_hallucinated": [],
    "is_fully_grounded": true
  },
  "rewritten_query": null
}
```

---

## Results

**Live telemetry** (103 logged queries):

| Metric | Value |
|---|---|
| Average latency | 0.35 s |
| Citation grounding rate | 98% |
| Average sources per answer | 6.5 |

**Evaluation benchmark** (`evaluate.py`, 8 gold questions across all three domains):

| Metric | Result |
|---|---|
| Retrieval hit rate | 75% (6/8) |
| Citation grounding rate | 100% (8/8) |

The two retrieval misses were both in the Consumer domain and came from the benchmark's keyword check being too strict, not from retrieval returning irrelevant text.

**Attached-PDF Q&A** held a 100% grounding rate across 80 logged queries on three documents, including the Consumer Protection Act 2019 (40 pages) and a BNSS bail document.

**Query rewriter**: trained on 48 pairs (344,064 trainable parameters, 0.45% of the base model). Training loss fell from about 5.9 to about 2.9 after masking padding tokens in the labels. On a Consumer query the rewrite produced a deeper, better-cited answer. On a Bail query it drifted off-topic. Both outcomes are reported rather than only the favourable one.

---

## Known Limitations

- **Matrimonial domain is metadata-only.** The source dataset has no judgment text, so answers there are shallow.
- **Scanned PDFs are not supported.** Extraction uses the PDF text layer; image-only documents are rejected with an error message.
- **Attached-PDF sessions are in memory.** Restarting the backend clears them and you must re-attach the file.
- **The rewriter is small.** 48 training pairs gives mixed results, especially for Bail phrasing.
- **Not production-ready.** No authentication, no scaling work, and the CORS policy is open for local development.
- **Not legal advice.** The system returns information grounded in retrieved documents and is not a substitute for a qualified legal professional.

---

## Suggested `.gitignore`

```
venv/
__pycache__/
.env
data/chroma_db/
data/raw_*/
data/interaction_log.csv
models/
*.zip
```

Datasets keep their own licences, so check each source before redistributing any data.

---

## Tech Stack

Python · FastAPI · ChromaDB · Sentence-Transformers (InLegalBERT) · rank_bm25 · Hugging Face Transformers + PEFT (LoRA) · PyMuPDF · Groq API · Streamlit · vanilla HTML/CSS/JS

## Acknowledgements

Built as a capstone project at the Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS, Mumbai. Thanks to the authors of GrahakNyay, IndianBailJudgments-1200, IMLJD, and InLegalBERT.
