"""
Layer 3, arm A (cloud embeddings) -- fixed RAG over the Titan index.

Identical to run_rag_local.py in every respect except the retrieval index:
same questions, same k, same fixed retrieval, same local Qwen2.5-VL answering
model, same prompt. Only the embedding model changes -- MiniLM (384d, local)
becomes Titan v2 (1024d, Bedrock).

That isolation is the point. The local run found the gold receipt in the top 5
only 34% of the time, and answer accuracy tracked that ceiling almost exactly.
This run separates two explanations:

  * if retrieval hit rate rises substantially, the weakness was MiniLM, and
    dense retrieval on receipt OCR is viable with a stronger encoder;
  * if it does not, the weakness is dense retrieval itself on this kind of
    text, and the structured-first result stands as the general finding.

Region is eu-west-1 throughout (account restriction); Titan is callable there
directly with no cross-region inference profile.

    python layer3/run_rag_titan.py --limit 10
    python layer3/run_rag_titan.py

Writes layer3/runs/rag_titan__whole__k5.jsonl -- same record shape as the other
arms, so score_rag.py reads it unchanged.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import boto3
import psycopg
import torch

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"
RUNS = ROOT / "layer3" / "runs"

REGION = "eu-west-1"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET = "sroie"

# Titan v2 published rate, USD per 1M input tokens.
PRICE_EMBED_IN = 0.02

SYSTEM = (
    "You answer questions about receipts using ONLY the receipt text provided. "
    "Give the shortest possible answer: a number, a date, or an address. "
    "Do not explain, do not show working, do not add currency symbols. "
    "If the answer is not present in the provided receipts, reply exactly: NOT_FOUND"
)


def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VLModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = VLModel.from_pretrained(MODEL_ID, quantization_config=bnb,
                                    device_map="auto")
    model.eval()
    return model, proc


@torch.inference_mode()
def ask(model, proc, question, context):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text",
         "text": f"Receipts:\n\n{context}\n\nQuestion: {question}"}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    ms = (time.time() - t0) * 1000
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(gen, skip_special_tokens=True).strip(), ms


def embed_query(client, text, retries=4):
    for attempt in range(retries):
        try:
            r = client.invoke_model(
                modelId=EMBED_MODEL,
                body=json.dumps({"inputText": text[:8000],
                                 "dimensions": EMBED_DIM, "normalize": True}))
            body = json.loads(r["body"].read())
            return body["embedding"], body.get("inputTextTokenCount", 0)
        except Exception as exc:                            # noqa: BLE001
            if "Throttl" in str(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def retrieve(cur, qvec, strategy, k):
    cur.execute(
        """SELECT doc_id, text, 1 - (embedding <=> %s::vector) AS sim
           FROM chunks_titan
           WHERE dataset=%s AND strategy=%s
           ORDER BY embedding <=> %s::vector LIMIT %s""",
        (qvec, DATASET, strategy, qvec, k))
    return cur.fetchall()


def numbers_in(s):
    return set(re.findall(r"\d+(?:\.\d+)?", (s or "").replace(",", "")))


def grounded(answer, context):
    if not answer or answer.strip() == "NOT_FOUND":
        return None
    nums = numbers_in(answer)
    if nums:
        return nums.issubset(numbers_in(context))
    a = re.sub(r"[^a-z0-9]", "", answer.lower())
    c = re.sub(r"[^a-z0-9]", "", context.lower())
    return bool(a) and a in c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="whole", choices=["whole", "window"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    questions = [json.loads(l) for l in QA.open() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]
    print(f"{len(questions)} questions, Titan embeddings, "
          f"strategy={a.strategy}, k={a.k}")

    client = boto3.client("bedrock-runtime", region_name=REGION)
    print(f"loading {MODEL_ID} in 4-bit...")
    model, proc = load_model()

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"rag_titan__{a.strategy}__k{a.k}.jsonl"

    n_err = 0
    with psycopg.connect(DSN) as conn, conn.cursor() as cur, out_path.open("w") as fh:
        for i, q in enumerate(questions, 1):
            try:
                vec, ntok = embed_query(client, q["question"])
            except Exception as exc:                        # noqa: BLE001
                print(f"embed failed on {q['qid']}: {str(exc)[:120]}")
                n_err += 1
                continue

            hits = retrieve(cur, str(vec), a.strategy, a.k)
            context = "\n\n---\n\n".join(f"[receipt {d}]\n{t}" for d, t, _ in hits)
            docs = [h[0] for h in hits]
            sims = [round(float(h[2]), 4) for h in hits]

            try:
                ans, ms = ask(model, proc, q["question"], context)
                err = None
            except Exception as exc:                        # noqa: BLE001
                ans, ms, err = None, 0, str(exc)[:300]
                n_err += 1

            fh.write(json.dumps({
                "qid": q["qid"], "kind": q["kind"], "field": q["field"],
                "question": q["question"], "gold_answer": q["gold_answer"],
                "gold_docs": q["gold_docs"],
                "model": MODEL_ID, "embed_model": EMBED_MODEL,
                "strategy": a.strategy, "k": a.k,
                "retrieved_docs": docs, "similarities": sims,
                "top_similarity": sims[0] if sims else None,
                "retrieval_hit": bool(set(docs) & set(q["gold_docs"])),
                "n_gold_docs_retrieved": len(set(docs) & set(q["gold_docs"])),
                "model_answer": ans,
                "answer_grounded": grounded(ans, context),
                "context_chars": len(context),
                "latency_ms": ms,
                "input_tokens": 0, "output_tokens": 0,
                "embed_tokens": ntok,
                "cost_usd": round(ntok / 1e6 * PRICE_EMBED_IN, 8),
                "error": err,
            }) + "\n")

            if i % 25 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}")

    print(f"\nwrote {out_path}")
    if n_err:
        print(f"WARNING: {n_err} errors")


if __name__ == "__main__":
    main()
