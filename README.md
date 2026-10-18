# Lawgic AI — Multi-Domain Indian Legal RAG System

Three case files, one shared pipeline: **Consumer Grievances**, **Bail Adjudication**, **Matrimonial Litigation**.

## Project structure

```
trustrag_multi/
├── TrustRAG_multidomain_ingestion.ipynb   # Run this first, in Jupyter Lab
├── backend/
│   └── main.py                            # FastAPI server
├── frontend/
│   └── index.html                         # Web UI (open directly in a browser)
├── data/
│   ├── raw_consumer/                      # Cloned Grahak-Nyay repo lands here
│   └── chroma_db/                         # Vector store, built by the notebook
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install fastapi uvicorn pydantic sentence-transformers rank-bm25 chromadb openai datasets huggingface_hub gitpython
```

### 2. Set your API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the ingestion notebook
Open `TrustRAG_multidomain_ingestion.ipynb` in Jupyter Lab and run all cells. This:
- Clones/downloads the three datasets (Consumer, Bail, Matrimonial)
- Chunks and tags each with its `domain`
- Builds one shared ChromaDB collection at `data/chroma_db`

**Before running Section 3 (Bail):** check whether the IBPS dataset (`arxiv.org/abs/2508.07592`) has been publicly released yet — the paper said it would be released "soon" as of publication. If it isn't available, the notebook falls back to HLDC; verify the exact Hugging Face repo ID for HLDC before running, since it isn't hardcoded correctly in the placeholder.

### 4. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Check it's alive: open `http://localhost:8000/health` — it should report how many chunks are indexed per domain.

### 5. Open the frontend
Just open `frontend/index.html` directly in a browser (double-click it, or `open frontend/index.html`). It talks to the backend at `http://localhost:8000`.

## What's implemented (the ~50–70% milestone)

- Multi-domain ingestion with domain-tagged chunking
- Hybrid retrieval (BM25 + InLegalBERT embeddings) filtered per domain
- Cross-encoder reranking
- Citation-enforced generation
- Citation verification (hallucination guardrail)
- Full web UI with domain switching, live citations, and a verification indicator

## What's left for the remaining milestone

- Evaluation benchmark (per domain, with legally verified gold answers)
- Domain-embedding comparison (InLegalBERT vs. generic baseline)
- LoRA/QLoRA fine-tuning for query rewriting
- Observability dashboard (latency, cost, citation accuracy over time)
- Deployment polish (currently local-only; CORS is wide open in `main.py` — tighten before sharing publicly)
