"""
Layer 3, arm A -- fixed RAG with a LOCAL answering model.

Same design as the Bedrock version: retrieval is FIXED (same k, same strategy,
no query rewriting, no re-retrieval), because H3 and H4 are statements about a
fixed retrieval path. Arms B (router) and C (agent) are compared against this.

No abstention threshold is applied. Similarity is recorded for every question
so the cutoff can be DERIVED from where accuracy collapses, not guessed.

Answering model: Qwen2.5-VL-7B-Instruct, already cached, loaded in 4-bit and
used text-only. Same family as the Layer-1 extractor, so the whole pipeline is
one model family and cost stays at zero per question -- which is the RQ3 story.
A second model from a DIFFERENT family is needed later to test H3 properly;
a size change within one family is not a meaningful test of "LLM choice".

    cd /mnt/yasir_drive/E_DATA/ResearchPlan2/docvlm-rq1
    python layer3/run_rag_local.py --strategy whole --limit 10   # smoke test
    python layer3/run_rag_local.py --strategy whole
    python layer3/run_rag_local.py --strategy window
    python layer3/run_rag_local.py --strategy lines

Writes layer3/runs/rag_local__<strategy>__k<k>.jsonl -- one record per question.
Scoring is a separate pass, so metrics can change without re-running generation.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import psycopg
import torch

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"
RUNS = ROOT / "layer3" / "runs"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = "sroie"

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
    except ImportError:                                    # older transformers
        from transformers import Qwen2VLForConditionalGeneration as VLModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = VLModel.from_pretrained(MODEL_ID, quantization_config=bnb,
                                    device_map="auto")
    model.eval()
    return model, proc


def embed_fn():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cuda")
    return lambda q: str(list(map(float, m.encode(q, normalize_embeddings=True))))


def retrieve(cur, qvec, strategy, k):
    cur.execute(
        """SELECT doc_id, chunk_no, text,
                  1 - (embedding <=> %s::vector) AS sim
           FROM chunks
           WHERE dataset = %s AND strategy = %s
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
        (qvec, DATASET, strategy, qvec, k),
    )
    return cur.fetchall()


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
    latency_ms = (time.time() - t0) * 1000

    gen = out[0][inputs["input_ids"].shape[1]:]
    answer = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer, latency_ms, int(inputs["input_ids"].shape[1]), int(gen.shape[0])


def numbers_in(s):
    return set(re.findall(r"\d+(?:\.\d+)?", (s or "").replace(",", "")))


def grounded(answer, context):
    """Is the answer actually present in what was retrieved?

    Separates an honest wrong answer (the model reported what it saw) from a
    hallucination (the value appears in no retrieved chunk). Only the latter is
    a hallucination, and that separation is what H4 requires.
    """
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
    ap.add_argument("--strategy", default="whole",
                    choices=["whole", "lines", "window"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    a = ap.parse_args()

    questions = [json.loads(l) for l in QA.open() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]
    print(f"{len(questions)} questions, strategy={a.strategy}, k={a.k}")

    embed = embed_fn()
    print(f"loading {MODEL_ID} in 4-bit...")
    model, proc = load_model()

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"rag_local__{a.strategy}__k{a.k}.jsonl"

    n_err = 0
    with psycopg.connect(DSN) as conn, conn.cursor() as cur, out_path.open("w") as fh:
        for i, q in enumerate(questions, 1):
            qvec = embed(q["question"])
            hits = retrieve(cur, qvec, a.strategy, a.k)

            context = "\n\n---\n\n".join(
                f"[receipt {doc_id}]\n{text}" for doc_id, _, text, _ in hits
            )
            retrieved_docs = [h[0] for h in hits]
            sims = [round(float(h[3]), 4) for h in hits]

            try:
                answer, latency_ms, tin, tout = ask(model, proc,
                                                    q["question"], context)
                err = None
            except Exception as exc:                       # noqa: BLE001
                answer, latency_ms, tin, tout = None, None, 0, 0
                err = str(exc)[:500]
                n_err += 1

            fh.write(json.dumps({
                "qid": q["qid"],
                "kind": q["kind"],
                "field": q["field"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "gold_docs": q["gold_docs"],
                "model": MODEL_ID,
                "strategy": a.strategy,
                "k": a.k,
                "retrieved_docs": retrieved_docs,
                "similarities": sims,
                "top_similarity": sims[0] if sims else None,
                "retrieval_hit": bool(set(retrieved_docs) & set(q["gold_docs"])),
                "n_gold_docs_retrieved": len(set(retrieved_docs) & set(q["gold_docs"])),
                "model_answer": answer,
                "answer_grounded": grounded(answer, context),
                "context_chars": len(context),
                "latency_ms": latency_ms,
                "input_tokens": tin,
                "output_tokens": tout,
                "cost_usd": 0.0,
                "error": err,
            }) + "\n")

            if i % 25 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}")

    print(f"\nwrote {out_path}")
    if n_err:
        print(f"WARNING: {n_err} generation errors -- check the error field")


if __name__ == "__main__":
    main()
