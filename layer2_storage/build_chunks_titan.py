"""
Layer 2 (cloud arm) -- embed the SROIE OCR text with Bedrock Titan v2.

Parallel to the local MiniLM index. Titan is 1024-dim against MiniLM's 384, so
it needs its own table rather than another column; keys match exactly, so the
two are directly comparable at query time.

This is the CLOUD side of the H3 embedding comparison. The local run found the
gold receipt in the top 5 only 34% of the time, which is the ceiling on arm-A
accuracy. If a stronger embedding model lifts that substantially, the weakness
was MiniLM; if it does not, the weakness is dense retrieval itself on receipt
OCR. Either answer is a result.

Region is eu-west-1 (Ireland) throughout: the account is restricted to that
region, and Titan is callable there directly with no cross-region inference
profile.

    python layer2/build_chunks_titan.py --init
    python layer2/build_chunks_titan.py --strategy whole

Cost: 347 receipts is roughly 100k tokens total, a fraction of a cent.
"""
import argparse
import json
import os
import time
from pathlib import Path

import boto3
import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "ocr_text__sroie__test.jsonl"

REGION = "eu-west-1"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
DATASET = "sroie"

CHUNK_SIZE, CHUNK_OVERLAP = 400, 80

DDL = f"""
CREATE TABLE IF NOT EXISTS chunks_titan (
    id          BIGSERIAL PRIMARY KEY,
    dataset     TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    chunk_no    INT  NOT NULL,
    strategy    TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({EMBED_DIM}),
    UNIQUE (dataset, doc_id, chunk_no, strategy)
);
CREATE INDEX IF NOT EXISTS idx_chunks_titan_doc ON chunks_titan (dataset, doc_id);
"""


def chunk(text, strategy):
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


def embed(client, text, retries=4):
    """Titan embeds one text per call. Retries on throttling."""
    for attempt in range(retries):
        try:
            r = client.invoke_model(
                modelId=EMBED_MODEL,
                body=json.dumps({"inputText": text[:8000],
                                 "dimensions": EMBED_DIM,
                                 "normalize": True}))
            return json.loads(r["body"].read())["embedding"]
        except Exception as exc:                            # noqa: BLE001
            if "Throttl" in str(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="whole",
                    choices=["whole", "lines", "window", "all"])
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"missing: {SRC}")

    records = [json.loads(l) for l in SRC.open() if l.strip()]
    print(f"{len(records)} documents, model={EMBED_MODEL} ({EMBED_DIM}d), "
          f"region={REGION}")

    client = boto3.client("bedrock-runtime", region_name=REGION)

    # 'lines' would be 17,266 separate API calls for a strategy already shown to
    # be the worst performer locally (accuracy 0.010). Not worth the calls.
    strategies = (["whole", "window"] if a.strategy == "all" else [a.strategy])
    if "lines" in strategies:
        print("WARNING: 'lines' is 17k API calls; it scored 0.010 locally.")

    with psycopg.connect(DSN) as conn:
        if a.init:
            conn.execute(DDL)
            conn.commit()

        for strategy in strategies:
            rows = []
            for r in records:
                for i, c in enumerate(chunk(r.get("text"), strategy)):
                    rows.append((r["doc_id"], i, c))
            print(f"\n{strategy}: {len(rows)} chunks")

            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks_titan WHERE dataset=%s "
                            "AND strategy=%s", (DATASET, strategy))
                for n, (doc_id, no, text) in enumerate(rows, 1):
                    v = embed(client, text)
                    cur.execute(
                        """INSERT INTO chunks_titan
                             (dataset, doc_id, chunk_no, strategy, text, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s)""",
                        (DATASET, doc_id, no, strategy, text, str(v)))
                    if n % 50 == 0 or n == len(rows):
                        print(f"  {n}/{len(rows)}")
                        conn.commit()
            conn.commit()

            with conn.cursor() as cur:
                cur.execute(f"CREATE INDEX IF NOT EXISTS "
                            f"idx_chunks_titan_vec_{strategy} "
                            f"ON chunks_titan USING hnsw "
                            f"(embedding vector_cosine_ops) "
                            f"WHERE strategy='{strategy}'")
            conn.commit()
            print(f"{strategy}: indexed")

    print("\ndone")


if __name__ == "__main__":
    main()
