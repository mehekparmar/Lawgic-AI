# TrustRAG — Multi-Domain Ingestion Script (v5 — restored + legal-aware chunking for bail)

import os, re, json, uuid, subprocess
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "law-ai/InLegalBERT"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="legal_chunks_multidomain",
    metadata={"hnsw:space": "cosine"}
)

def index_records(records, domain: str, batch_size: int = 32):
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        embeddings = embedder.encode([r["text"] for r in batch]).tolist()
        collection.upsert(
            ids=[str(uuid.uuid4()) for _ in batch],
            documents=[r["text"] for r in batch],
            embeddings=embeddings,
            metadatas=[{"domain": domain, "citation": r["citation"]} for r in batch],
        )
    print(f"Indexed {len(records)} chunks for domain='{domain}'. Collection total: {collection.count()}")

def chunk_legal_text(text: str, max_chars: int = 600) -> list:
    """Splits on sentence boundaries, grouping sentences up to max_chars per chunk —
    keeps clauses/sentences intact rather than cutting mid-sentence."""
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

print("Consumer domain: already indexed (942 records), skipping.")

# --- Bail domain: real dataset, now with legal-aware chunking ---
from datasets import load_dataset

existing_bail = collection.get(where={"domain": "bail"}, limit=1)
if existing_bail["ids"]:
    print("Clearing old (unchunked) bail data to re-index with clause-aware chunking...")
    collection.delete(where={"domain": "bail"})

bail_records = []
try:
    bail_ds = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200")
    split_name = list(bail_ds.keys())[0]
    ds = bail_ds[split_name]
    print(f"Bail dataset columns: {ds.column_names}")
    for row in ds:
        facts = row.get("facts", "") or ""
        reason = row.get("judgment_reason", "") or ""
        full_text = f"{facts}\n\nJudgment reasoning: {reason}".strip()
        citation = row.get("case_title") or row.get("case_id") or "Bail case"
        for chunk in chunk_legal_text(full_text, max_chars=600):
            if chunk.strip():
                bail_records.append({"text": chunk.strip(), "citation": str(citation)})
except Exception as e:
    print(f"Could not load bail dataset ({e})")

print(f"Loaded {len(bail_records)} bail chunks (clause-aware)")
if bail_records:
    index_records(bail_records, domain="bail")

# --- Matrimonial domain: already indexed, skip ---
existing_matrimonial = collection.get(where={"domain": "matrimonial"}, limit=1)
if existing_matrimonial["ids"]:
    print("Matrimonial domain already indexed, skipping.")
else:
    from datasets import load_dataset
    matrimonial_records = []
    try:
        imljd_ds = load_dataset("joyboseroy/imljd", "combined")
        split_name = list(imljd_ds.keys())[0]
        ds = imljd_ds[split_name]
        for row in ds:
            parts = [
                f"Case: {row.get('title', 'Unknown')}",
                f"Court: {row.get('court', 'Unknown')}",
                f"Judge: {row.get('judge', 'Unknown')}",
                f"Decision date: {row.get('decision_date', 'Unknown')}",
                f"Outcome: {row.get('disposal_nature', 'Unknown')}",
            ]
            text = ". ".join(parts)
            citation = row.get("title") or row.get("cnr") or "Matrimonial case"
            if text.strip():
                matrimonial_records.append({"text": text.strip(), "citation": str(citation)})
    except Exception as e:
        print(f"Could not load IMLJD ({e})")
    print(f"Loaded {len(matrimonial_records)} matrimonial records")
    if matrimonial_records:
        index_records(matrimonial_records, domain="matrimonial")

# --- Sanity check ---
def semantic_search(query: str, domain: str = None, top_k: int = 5):
    q_emb = embedder.encode([query]).tolist()
    where = {"domain": domain} if domain else None
    results = collection.query(query_embeddings=q_emb, n_results=top_k, where=where)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "citation": meta["citation"], "domain": meta["domain"], "score": 1 - dist})
    return hits

for domain in ["consumer", "bail", "matrimonial"]:
    print(f"--- {domain} ---")
    for h in semantic_search("What are my rights in this case?", domain=domain, top_k=2):
        print(h["citation"], "|", h["text"][:120].replace("\n", " "))
    print()
