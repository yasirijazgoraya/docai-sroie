"""
Layer 2 -- chunk the SROIE OCR text and embed it into pgvector.

Only full document text goes here. Structured fields live in the documents
table and are queried directly with SQL, never embedded.

    cd /mnt/yasir_drive/E_DATA/ResearchPlan2/docvlm-rq1
    python layer2/build_chunks.py                 # all three strategies
    python layer2/build_chunks.py --strategy lines

Chunking strategy is the RQ2 experimental variable (H3). All strategies are
stored side by side under the `strategy` column so retrieval configurations can
be compared with a WHERE clause instead of an index rebuild.

Note the known limitation: an SROIE receipt is short, so 'whole' produces one
chunk per document and chunking barely varies. Demonstrating that chunking
dominates answer accuracy would need a long-document corpus.

Requires: pip install "psycopg[binary]" sentence-transformers
"""
import argparse
import json
import os
from pathlib import Path

import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "ocr_text__sroie__test.jsonl"
DATASET = "sroie"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dims
CHUNK_SIZE = 400          # chars, for the window strategy
CHUNK_OVERLAP = 80


def chunk(text: str, strategy: str):
    text = (text or "").strip()
    if not text:
        return []
    if strategy == "whole":
        return [text]
    if strategy == "lines":
        return [ln.strip() for ln in text.split("\n") if ln.strip()]
    if strategy == "window":
        out, i, step = [], 0, max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        while i < len(text):
            out.append(text[i:i + CHUNK_SIZE])
            i += step
        return out
    raise ValueError(strategy)


def build(strategy, records, model, conn):
    rows = []
    for r in records:
        for i, c in enumerate(chunk(r.get("text"), strategy)):
            rows.append((r["doc_id"], i, c))
    if not rows:
        print(f"{strategy}: nothing to index")
        return

    print(f"{strategy}: {len(rows)} chunks, embedding...")
    vecs = model.encode([t for _, _, t in rows], batch_size=64,
                        show_progress_bar=True, normalize_embeddings=True)

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE dataset=%s AND strategy=%s",
                    (DATASET, strategy))
        for (doc_id, no, text), v in zip(rows, vecs):
            cur.execute(
                """INSERT INTO chunks
                     (dataset, doc_id, chunk_no, strategy, text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (DATASET, doc_id, no, strategy, text, str(list(map(float, v)))),
            )
    conn.commit()
    print(f"{strategy}: indexed {len(rows)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["whole", "lines", "window", "all"],
                    default="all")
    a = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"missing: {SRC}")

    records = [json.loads(l) for l in SRC.open() if l.strip()]
    print(f"{len(records)} documents from {SRC.name}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    strategies = ["whole", "lines", "window"] if a.strategy == "all" else [a.strategy]

    with psycopg.connect(DSN) as conn:
        for s in strategies:
            build(s, records, model, conn)

        print("building HNSW index (once, after load)...")
        with conn.cursor() as cur:
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_vec
                           ON chunks USING hnsw (embedding vector_cosine_ops)""")
        conn.commit()

    print("done")


if __name__ == "__main__":
    main()
