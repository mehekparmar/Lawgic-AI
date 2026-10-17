"""
TrustRAG — Evaluation benchmark.
Run with: python evaluate.py
Scores retrieval + citation-grounding across a small gold question set per domain.
"""

import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq

CHROMA_DIR = Path("/Users/mehekparmar/Desktop/files/trustrag_multi/data/chroma_db")
VALID_DOMAINS = {"consumer", "bail", "matrimonial"}

print("Loading models...")
embedder = SentenceTransformer("law-ai/InLegalBERT")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name="legal_chunks_multidomain")
llm_client = Groq()

chunks_by_domain = {}
bm25_by_domain = {}
for domain in VALID_DOMAINS:
    results = collection.get(where={"domain": domain})
    chunks = [{"id": cid, "text": doc, "citation": meta["citation"]}
               for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])]
    chunks_by_domain[domain] = chunks
    if chunks:
        bm25_by_domain[domain] = BM25Okapi([c["text"].lower().split() for c in chunks])

def bm25_search(query, domain, top_k=20):
    bm25 = bm25_by_domain.get(domain)
    if not bm25:
        return []
    scores = bm25.get_scores(query.lower().split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks_by_domain[domain][i] for i in idx]

def semantic_search(query, domain, top_k=20):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k, where={"domain": domain})
    return [{"id": cid, "text": doc, "citation": meta["citation"]}
            for doc, meta, cid in zip(results["documents"][0], results["metadatas"][0], results["ids"][0])]

def hybrid_retrieve(query, domain, top_k=5, rrf_k=60):
    bm25_hits = bm25_search(query, domain)
    sem_hits = semantic_search(query, domain)
    rrf, lookup = {}, {}
    for rank, c in enumerate(bm25_hits):
        rrf[c["id"]] = rrf.get(c["id"], 0) + 1 / (rrf_k + rank + 1)
        lookup[c["id"]] = c
    for rank, c in enumerate(sem_hits):
        rrf[c["id"]] = rrf.get(c["id"], 0) + 1 / (rrf_k + rank + 1)
        lookup[c["id"]] = c
    if not rrf:
        return []
    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:max(top_k*2, 10)]
    candidates = [lookup[cid] for cid, _ in fused]
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in reranked[:top_k]]

CITED_PATTERN = re.compile(r"\[([^\]]+)\]")

def generate_answer(query, domain, chunks):
    context = "\n\n".join(f"[{c['citation']}]\n{c['text']}" for c in chunks)
    prompt = f"""Answer using ONLY the context below. Every claim MUST end with its citation in square brackets.
If context lacks the answer, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    resp = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def verify_citations(answer, chunks):
    retrieved = {c["citation"] for c in chunks}
    cited = set(CITED_PATTERN.findall(answer))
    return {"grounded": len(cited - retrieved) == 0, "total_cited": len(cited)}

GOLD_SET = {
    "consumer": [
        {"query": "What happens if a product I bought online is defective?", "expect_keyword": "refund"},
        {"query": "Can I get a refund if the product was delivered late?", "expect_keyword": "delivery"},
        {"query": "What if my bank is not registered with UPI?", "expect_keyword": "UPI"},
    ],
    "bail": [
        {"query": "What factors does a court consider when deciding to grant bail?", "expect_keyword": "bail"},
        {"query": "What happens if bail is cancelled after being granted?", "expect_keyword": "cancel"},
        {"query": "How does prior criminal history affect a bail decision?", "expect_keyword": "prior"},
    ],
    "matrimonial": [
        {"query": "What is the disposal outcome in a matrimonial case?", "expect_keyword": "dismiss"},
        {"query": "Which court handles matrimonial disputes in Karnataka?", "expect_keyword": "karnataka"},
    ],
}

def run_eval():
    results = []
    for domain, questions in GOLD_SET.items():
        for item in questions:
            query = item["query"]
            chunks = hybrid_retrieve(query, domain)
            retrieval_hit = any(item["expect_keyword"].lower() in c["text"].lower() for c in chunks)
            if not chunks:
                results.append({"domain": domain, "query": query, "retrieval_hit": False, "grounded": None})
                continue
            answer = generate_answer(query, domain, chunks)
            verification = verify_citations(answer, chunks)
            results.append({
                "domain": domain, "query": query,
                "retrieval_hit": retrieval_hit,
                "grounded": verification["grounded"],
                "answer_preview": answer[:150],
            })
    return results

if __name__ == "__main__":
    results = run_eval()

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for r in results:
        print(f"\n[{r['domain'].upper()}] {r['query']}")
        print(f"  Retrieval hit: {'✅' if r['retrieval_hit'] else '❌'}")
        print(f"  Citations grounded: {'✅' if r['grounded'] else ('❌' if r['grounded'] is False else 'N/A')}")

    total = len(results)
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    grounded = sum(1 for r in results if r["grounded"] is True)
    scored = sum(1 for r in results if r["grounded"] is not None)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Retrieval hit rate:     {retrieval_hits}/{total} ({100*retrieval_hits/total:.0f}%)")
    print(f"Citation grounding rate: {grounded}/{scored} ({100*grounded/scored:.0f}%)" if scored else "No answers generated")
