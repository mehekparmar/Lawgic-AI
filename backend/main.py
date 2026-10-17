"""
TrustRAG backend — FastAPI service wrapping the multi-domain legal RAG pipeline.

Run with:
    uvicorn main:app --reload --port 8000

Requires:
    GROQ_API_KEY set as an environment variable.
    The ChromaDB collection already populated by ingest.py
    (expects it at ../data/chroma_db, collection name "legal_chunks_multidomain").
"""

import os
import re
import io
import csv
import time
import json
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
import fitz  # PyMuPDF

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "interaction_log.csv"
COLLECTION_NAME = "legal_chunks_multidomain"
VALID_DOMAINS = {"consumer", "bail", "matrimonial"}

embedder = SentenceTransformer("law-ai/InLegalBERT")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

llm_client = Groq()  # reads GROQ_API_KEY from environment

# --- Legal query rewriter (LoRA fine-tuned) ---
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

REWRITER_DIR = Path(__file__).resolve().parent.parent / "models" / "legal_query_rewriter_lora"
REWRITE_PROMPT_PREFIX = "Rewrite this legal question using precise legal terminology: "
_rewriter_available = False
try:
    _rewriter_tokenizer = T5Tokenizer.from_pretrained(str(REWRITER_DIR))
    _rewriter_base = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    _rewriter_base.config.tie_word_embeddings = False
    _rewriter_model = PeftModel.from_pretrained(_rewriter_base, str(REWRITER_DIR))
    _rewriter_model.eval()
    _rewriter_available = True
    print("[startup] Legal query rewriter loaded successfully")
except Exception as e:
    print(f"[startup] Query rewriter unavailable, falling back to raw queries: {e}")

def rewrite_query(query: str) -> str:
    if not _rewriter_available:
        return query
    inputs = _rewriter_tokenizer(REWRITE_PROMPT_PREFIX + query, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _rewriter_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return _rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)

_bm25_by_domain = {}
_chunks_by_domain = {}

def _load_bm25_indexes():
    for domain in VALID_DOMAINS:
        results = collection.get(where={"domain": domain})
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        ids = results.get("ids", [])
        chunks = [
            {"id": cid, "text": doc, "citation": meta["citation"]}
            for cid, doc, meta in zip(ids, docs, metas)
        ]
        _chunks_by_domain[domain] = chunks
        if chunks:
            tokenized = [c["text"].lower().split() for c in chunks]
            _bm25_by_domain[domain] = BM25Okapi(tokenized)
        print(f"[startup] domain='{domain}': {len(chunks)} chunks indexed for BM25")

_load_bm25_indexes()

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def bm25_search(query: str, domain: str, top_k: int = 20):
    bm25 = _bm25_by_domain.get(domain)
    chunks = _chunks_by_domain.get(domain, [])
    if not bm25 or not chunks:
        return []
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in ranked_idx]

def semantic_search(query: str, domain: str, top_k: int = 20):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k, where={"domain": domain})
    hits = []
    for doc, meta, dist, cid in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
    ):
        hits.append({"id": cid, "text": doc, "citation": meta["citation"]})
    return hits

def hybrid_retrieve(query: str, domain: str, top_k: int = 5, rrf_k: int = 60):
    bm25_hits = bm25_search(query, domain, top_k=20)
    semantic_hits = semantic_search(query, domain, top_k=20)
    rrf_scores, lookup = {}, {}
    for rank, chunk in enumerate(bm25_hits):
        rrf_scores[chunk["id"]] = rrf_scores.get(chunk["id"], 0) + 1 / (rrf_k + rank + 1)
        lookup[chunk["id"]] = chunk
    for rank, chunk in enumerate(semantic_hits):
        rrf_scores[chunk["id"]] = rrf_scores.get(chunk["id"], 0) + 1 / (rrf_k + rank + 1)
        lookup[chunk["id"]] = chunk
    if not rrf_scores:
        return []
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: max(top_k * 2, 10)]
    candidates = [lookup[cid] for cid, _ in fused]
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in reranked[:top_k]]

# ---------------------------------------------------------------------------
# Legal-aware chunking (used for live PDF uploads)
# ---------------------------------------------------------------------------

def chunk_legal_text(text: str, max_chars: int = 600) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, buf = [], ""
    for sentence in sentences:
        if len(buf) + len(sentence) <= max_chars:
            buf += (" " if buf else "") + sentence
        else:
            if buf:
                chunks.append(buf.strip())
            buf = sentence
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# ---------------------------------------------------------------------------
# Generation + citation verification
# ---------------------------------------------------------------------------

DOMAIN_LABELS = {
    "consumer": "Indian consumer protection law",
    "bail": "Indian bail adjudication and criminal procedure",
    "matrimonial": "Indian matrimonial and family law",
}

CITED_PATTERN = re.compile(r"[\[【]([^\]】]+)[\]】]")

def build_prompt(query: str, domain: str, retrieved_chunks: list) -> str:
    context_block = "\n\n".join(f"[{c['citation']}]\n{c['text']}" for c in retrieved_chunks)
    domain_label = DOMAIN_LABELS.get(domain, "Indian law")
    return f"""You are a legal information assistant specializing in {domain_label}. Answer using ONLY the context below.
Every factual claim MUST end with its exact citation in square brackets, e.g. [Case Name / Section].
If the context does not contain the answer, say so explicitly. Do not use outside knowledge.
This is general legal information, not legal advice — do not tell the user what they personally should do.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER (with inline citations):"""

def generate_answer(query: str, domain: str, retrieved_chunks: list) -> str:
    prompt = build_prompt(query, domain, retrieved_chunks)
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

def verify_citations(answer: str, retrieved_chunks: list) -> dict:
    retrieved_citations = {c["citation"] for c in retrieved_chunks}
    cited_in_answer = set(CITED_PATTERN.findall(answer))
    verified = cited_in_answer & retrieved_citations
    unverified = cited_in_answer - retrieved_citations
    return {
        "total_citations_in_answer": len(cited_in_answer),
        "verified": list(verified),
        "unverified_hallucinated": list(unverified),
        "is_fully_grounded": len(unverified) == 0,
    }

# ---------------------------------------------------------------------------
# Observability logging
# ---------------------------------------------------------------------------

def log_interaction(domain: str, query: str, latency_seconds: float, num_sources: int, grounded: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "domain", "query", "latency_seconds", "num_sources", "grounded"])
        writer.writerow([datetime.now().isoformat(), domain, query, round(latency_seconds, 3), num_sources, grounded])

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="TrustRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str
    domain: str  # "consumer" | "bail" | "matrimonial"
    use_rewriter: bool = False  # optional: rewrite query into legal terminology before retrieval

class AskResponse(BaseModel):
    query: str
    domain: str
    answer: str
    sources: list[str]
    verification: dict
    rewritten_query: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "domains_indexed": {d: len(_chunks_by_domain.get(d, [])) for d in VALID_DOMAINS}}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"domain must be one of {sorted(VALID_DOMAINS)}")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    start_time = time.time()

    search_query = req.query
    rewritten = None
    if req.use_rewriter:
        rewritten = rewrite_query(req.query)
        search_query = rewritten

    retrieved = hybrid_retrieve(search_query, req.domain, top_k=5)

    if not retrieved:
        log_interaction(req.domain, req.query, time.time() - start_time, 0, True)
        return AskResponse(
            query=req.query,
            domain=req.domain,
            answer="No indexed documents found for this domain yet. Run the ingestion script first.",
            sources=[],
            verification={"total_citations_in_answer": 0, "verified": [], "unverified_hallucinated": [], "is_fully_grounded": True},
            rewritten_query=rewritten,
        )

    answer = generate_answer(req.query, req.domain, retrieved)
    verification = verify_citations(answer, retrieved)
    latency = time.time() - start_time
    log_interaction(req.domain, req.query, latency, len(retrieved), verification["is_fully_grounded"])

    return AskResponse(
        query=req.query,
        domain=req.domain,
        answer=answer,
        sources=[c["citation"] for c in retrieved],
        verification=verification,
        rewritten_query=rewritten,
    )

@app.post("/upload")
async def upload_document(domain: str = Form(...), file: UploadFile = File(...)):
    if domain not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"domain must be one of {sorted(VALID_DOMAINS)}")

    contents = await file.read()
    try:
        doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
        full_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    chunks = chunk_legal_text(full_text, max_chars=600)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"upload-{file.filename}-{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"domain": domain, "citation": f"Uploaded: {file.filename}"} for _ in chunks],
    )
    _load_bm25_indexes()  # refresh in-memory BM25 so the new doc is searchable immediately

    return {"status": "ok", "filename": file.filename, "chunks_indexed": len(chunks), "domain": domain}


import numpy as np

@app.post("/ask-pdf")
async def ask_pdf(query: str = Form(...), file: UploadFile = File(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    contents = await file.read()
    try:
        doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
        full_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    chunks_text = chunk_legal_text(full_text, max_chars=600)
    if not chunks_text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    chunk_embeddings = embedder.encode(chunks_text)
    query_embedding = embedder.encode([query])[0]

    sims = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    top_idx = np.argsort(sims)[::-1][:5]
    retrieved = [
        {"text": chunks_text[i], "citation": f"{file.filename} (excerpt {i+1})"}
        for i in top_idx
    ]

    answer = generate_answer(query, "uploaded_document", retrieved)
    verification = verify_citations(answer, retrieved)

    return {
        "query": query,
        "filename": file.filename,
        "answer": answer,
        "sources": [c["citation"] for c in retrieved],
        "verification": verification,
    }

import uuid

# ---------------------------------------------------------------------------
# Attached-PDF Q&A (page-aware, per-document session)
# ---------------------------------------------------------------------------

document_sessions = {}  # doc_id -> {"filename", "num_pages", "chunks": [{"text","page","embedding"}]}

def extract_pages_and_chunk(contents: bytes, max_chars: int = 600):
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
    page_chunks = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if not page_text.strip():
            continue
        for chunk in chunk_legal_text(page_text, max_chars=max_chars):
            if chunk.strip():
                page_chunks.append({"text": chunk.strip(), "page": page_num})
    num_pages = doc.page_count
    doc.close()
    return page_chunks, num_pages

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        page_chunks, num_pages = extract_pages_and_chunk(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    if not page_chunks:
        raise HTTPException(
            status_code=400,
            detail="This PDF appears to be scanned or image-based with no extractable text layer. "
                   "Please try a text-based PDF instead (e.g. one exported from Word, or a digitally "
                   "published document rather than a photocopy/scan)."
        )

    texts = [c["text"] for c in page_chunks]
    embeddings = embedder.encode(texts)

    doc_id = str(uuid.uuid4())
    document_sessions[doc_id] = {
        "filename": file.filename,
        "num_pages": num_pages,
        "chunks": [
            {"text": c["text"], "page": c["page"], "embedding": emb}
            for c, emb in zip(page_chunks, embeddings)
        ],
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_pages": num_pages,
        "num_chunks": len(page_chunks),
    }

PAGE_QUERY_PATTERN = re.compile(r"page\s+(\d+)", re.IGNORECASE)
SUMMARY_KEYWORDS = ["summarize", "summary", "overview", "all the laws", "entire document",
                    "whole document", "explain everything", "give me a summary"]

def _cosine_sims(chunks, query_embedding):
    embs = np.array([c["embedding"] for c in chunks])
    sims = np.dot(embs, query_embedding) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    return sims

def build_doc_prompt(query: str, retrieved: list, mode: str) -> str:
    context_block = "\n\n".join(f"[{c['citation']}]\n{c['text']}" for c in retrieved)
    if mode == "summarize":
        instruction = (
            "Synthesize a clear summary across the ENTIRE document based on the excerpts below, "
            "which are sampled from across all pages. Organize the summary logically and cite the "
            "page number for each point you make, in square brackets, e.g. [document.pdf, page 4]."
        )
    else:
        instruction = (
            "Answer using ONLY the excerpts below. Every factual claim MUST end with its exact "
            "citation in square brackets, e.g. [document.pdf, page 4]. If the excerpts do not "
            "contain the answer, say so explicitly. Do not use outside knowledge."
        )
    return f"""You are a legal document assistant helping a user understand an attached PDF.
{instruction}

EXCERPTS:
{context_block}

QUESTION: {query}

ANSWER (with inline page citations):"""

CITATION_PAGE_PATTERN = re.compile(r"page[s]?\s*(\d+)(?:\s*[-\u2010-\u2015]\s*(\d+))?", re.IGNORECASE)

def verify_doc_citations(answer: str, retrieved_chunks: list) -> dict:
    retrieved_pages = {c["page"] for c in retrieved_chunks}
    cited_spans = CITED_PATTERN.findall(answer)
    verified, unverified = [], []
    for span in cited_spans:
        pages_in_span = set()
        for m in CITATION_PAGE_PATTERN.finditer(span):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            pages_in_span.update(range(start, end + 1))
        if pages_in_span and (pages_in_span & retrieved_pages):
            verified.append(span)
        else:
            unverified.append(span)
    return {
        "total_citations_in_answer": len(cited_spans),
        "verified": verified,
        "unverified_hallucinated": unverified,
        "is_fully_grounded": len(unverified) == 0,
    }

class AskDocRequest(BaseModel):
    doc_id: str
    query: str

@app.post("/ask-doc")
def ask_doc(req: AskDocRequest):
    doc_id = req.doc_id
    query = req.query.strip()

    if doc_id not in document_sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired doc_id. Please re-attach the PDF.")
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    session = document_sessions[doc_id]
    all_chunks = session["chunks"]
    filename = session["filename"]

    page_match = PAGE_QUERY_PATTERN.search(query)
    is_summary = any(kw in query.lower() for kw in SUMMARY_KEYWORDS)
    query_embedding = embedder.encode([query])[0]

    if page_match:
        mode = "page"
        target_page = int(page_match.group(1))
        candidates = [c for c in all_chunks if c["page"] == target_page]
        if not candidates:
            candidates = [c for c in all_chunks if abs(c["page"] - target_page) <= 1]
        if not candidates:
            candidates = all_chunks
        sims = _cosine_sims(candidates, query_embedding)
        top_idx = np.argsort(sims)[::-1][:5]
        retrieved_chunks = [candidates[i] for i in top_idx]

    elif is_summary:
        mode = "summarize"
        by_page = {}
        for c in all_chunks:
            by_page.setdefault(c["page"], []).append(c)
        retrieved_chunks = [by_page[p][0] for p in sorted(by_page.keys())]
        if len(retrieved_chunks) > 25:
            step = len(retrieved_chunks) / 25
            retrieved_chunks = [retrieved_chunks[int(i * step)] for i in range(25)]

    else:
        mode = "general"
        sims = _cosine_sims(all_chunks, query_embedding)

        # Definitional-query boost: for "what is X" / "what is meant by X" style
        # questions, also surface chunks containing '"X" means' verbatim (the
        # standard drafting pattern in Indian statutes), since these can be
        # semantically under-ranked in dense, consolidated definition sections.
        boosted_idx = []
        q_lower = query.strip().lower()
        is_definitional = bool(re.match(r"^(what is|what are|what does|define|meaning of)\b", q_lower)) or "meant by" in q_lower

        if is_definitional:
            strip_phrases = [
                "what is meant by", "what does", "what is", "what are",
                "define", "meaning of", "mean under the act", "under the act",
                "according to the act", "mean", "a ", "an ", "the ", "?",
            ]
            term = q_lower
            for phrase in strip_phrases:
                term = term.replace(phrase, " ")
            term = re.sub(r"\s+", " ", term).strip()

            if len(term) >= 3:
                for i, c in enumerate(all_chunks):
                    text_lower = c["text"].lower()
                    if f'"{term}" means' in text_lower or f"'{term}' means" in text_lower or f"{term} means" in text_lower:
                        boosted_idx.append(i)

        top_idx = list(np.argsort(sims)[::-1][:8])
        for i in boosted_idx:
            if i not in top_idx:
                top_idx.insert(0, i)
        top_idx = top_idx[:8]

        retrieved_chunks = [all_chunks[i] for i in top_idx]

    retrieved = [
        {"text": c["text"], "citation": f"{filename}, page {c['page']}"}
        for c in retrieved_chunks
    ]

    prompt = build_doc_prompt(query, retrieved, mode)
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    print(f"[DEBUG ask-doc] finish_reason={response.choices[0].finish_reason} answer_len={len(answer) if answer else 0} raw_repr={answer!r}")
    verification = verify_doc_citations(answer, retrieved_chunks)
    log_interaction(f"pdf:{filename}", query, 0.0, len(retrieved_chunks), verification["is_fully_grounded"])

    return {
        "query": query,
        "filename": filename,
        "mode": mode,
        "answer": answer,
        "sources": [c["citation"] for c in retrieved],
        "verification": verification,
    }
