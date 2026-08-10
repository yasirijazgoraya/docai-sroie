"""
Layer 3, arm B -- ROUTER (SQL path + vector path).

Arm A showed that aggregate questions are structurally unanswerable by
retrieval: with k=5 against vendors holding 7-30 receipts, the model never sees
the data it would need to sum. Accuracy was 0.011. This arm routes those
questions to SQL over the structured store instead, and leaves lookup questions
on the vector path.

Design choices, both deliberate:

  * INTENT EXTRACTION, not free-form text-to-SQL. The LLM emits a small JSON
    object (action, vendor, year); this code builds a parameterised query. A 7B
    model writing arbitrary SQL is unreliable, and an SME product should never
    execute model-authored SQL against a live database.

  * The SQL runs against EXTRACTED data, not gold. A vendor whose total was
    misread in Layer 1 produces a slightly wrong sum. That is correct: it
    measures the pipeline end to end rather than the database in isolation, and
    the gap between this arm and a gold-oracle upper bound is itself a result.

Routing is done by the model, not by the dataset's `kind` label. Using the
label would be oracle routing and would overstate what a deployed system can
do; the router's own misclassifications are recorded and counted.

    python layer3/run_router.py --strategy whole --limit 10
    python layer3/run_router.py --strategy whole

Writes layer3/runs/router__<strategy>__k<k>.jsonl
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
DATASET, ARM = "sroie", "zeroshot"

ROUTE_SYSTEM = """You convert a question about receipts into a JSON intent.

Reply with ONLY a JSON object, no other text. Fields:
  "action": one of
      "sum_vendor"    total spend at one vendor
      "count_vendor"  how many receipts from one vendor
      "max_vendor"    largest single purchase at one vendor
      "sum_year"      total spend in a year
      "count_year"    how many receipts in a year
      "lookup"        anything about ONE specific receipt
  "vendor": the company name if the question names one, else null
  "year": the four-digit year if the question names one, else null

Examples:
Q: How much did I spend at ABC TRADING in total?
{"action":"sum_vendor","vendor":"ABC TRADING","year":null}
Q: How many receipts do I have from 2018?
{"action":"count_year","vendor":null,"year":2018}
Q: What was the total on the XYZ receipt from 3 May 2018?
{"action":"lookup","vendor":"XYZ","year":null}"""

ANSWER_SYSTEM = (
    "You answer questions about receipts using ONLY the receipt text provided. "
    "Give the shortest possible answer: a number, a date, or an address. "
    "Do not explain, do not show working, do not add currency symbols. "
    "If the answer is not present in the provided receipts, reply exactly: NOT_FOUND"
)

SQL_ACTIONS = {"sum_vendor", "count_vendor", "max_vendor", "sum_year", "count_year"}


# ----------------------------------------------------------------- model io

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
def generate(model, proc, system, user, max_new_tokens=96):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    ms = (time.time() - t0) * 1000
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(gen, skip_special_tokens=True).strip(), ms


def parse_intent(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "action" not in d:
        return None
    return d


# ------------------------------------------------------------ vendor lookup

def norm_vendor(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


STOP = {"sdn", "bhd", "co", "ltd", "the", "and", "m", "s", "b", "enterprise"}


def tokens(s):
    return {t for t in norm_vendor(s).split() if t and t not in STOP}


def resolve_vendor(name, known):
    """Match a question's vendor string to a vendor string in the store.

    Extracted vendor strings differ from the question's wording (OCR noise,
    punctuation, suffixes), so exact equality would fail. Best token overlap
    with a minimum threshold; returns None rather than guessing when nothing
    matches well, so a bad match is not silently scored as a retrieval result.
    """
    want = tokens(name)
    if not want:
        return None
    # IDF-weighted symmetric overlap. Generic tokens (hardware, cash, carry,
    # restoran) appear across many vendors and must not drive a match; rare
    # tokens (unihakka, kuchai) must. A flat containment rule merged distinct
    # companies that share only generic words, and merged branches that gold
    # treats as separate.
    import math
    df = {}
    for v in known:
        for t in tokens(v):
            df[t] = df.get(t, 0) + 1
    n = max(len(known), 1)
    idf = lambda t: math.log(n / (1 + df.get(t, 0))) + 1.0

    w_want = sum(idf(t) for t in want)
    if w_want <= 0:
        return None
    matches = []
    for v in known:
        have = tokens(v)
        if not have:
            continue
        inter = sum(idf(t) for t in want & have)
        # must cover most of the question's information AND most of the
        # candidate's, so neither side carries unmatched distinctive tokens
        if inter / w_want >= 0.75 and inter / sum(idf(t) for t in have) >= 0.75:
            matches.append(v)
    return matches or None


# ----------------------------------------------------------------- sql path

def run_sql(cur, intent, known_vendors):
    action = intent.get("action")
    vendor_q = intent.get("vendor")
    year = intent.get("year")

    if action in ("sum_vendor", "count_vendor", "max_vendor"):
        vendors = resolve_vendor(vendor_q, known_vendors)
        if not vendors:
            return None, None, f"vendor not resolved: {vendor_q!r}"
        agg = {"sum_vendor": "sum(total)", "count_vendor": "count(*)",
               "max_vendor": "max(total)"}[action]
        sql = (f"SELECT {agg} FROM documents "
               f"WHERE dataset=%s AND arm=%s AND vendor = ANY(%s)")
        cur.execute(sql, (DATASET, ARM, vendors))
        val = cur.fetchone()[0]
        return val, f"{sql} -- matched {len(vendors)} vendor strings: {vendors}", None

    if action in ("sum_year", "count_year"):
        if not year:
            return None, None, "no year in intent"
        agg = "sum(total)" if action == "sum_year" else "count(*)"
        sql = (f"SELECT {agg} FROM documents "
               f"WHERE dataset=%s AND arm=%s "
               f"AND extract(year from doc_date)=%s")
        cur.execute(sql, (DATASET, ARM, int(year)))
        val = cur.fetchone()[0]
        return val, f"{sql} -- ({DATASET}, {ARM}, {int(year)})", None

    return None, None, f"unroutable action: {action}"


def fmt(val, action):
    if val is None:
        return "NOT_FOUND"
    if action in ("count_vendor", "count_year"):
        return str(int(val))
    return f"{float(val):.2f}"


# -------------------------------------------------------------- vector path

def retrieve(cur, qvec, strategy, k):
    cur.execute(
        """SELECT doc_id, text, 1 - (embedding <=> %s::vector) AS sim
           FROM chunks WHERE dataset=%s AND strategy=%s
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


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="whole",
                    choices=["whole", "lines", "window"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    questions = [json.loads(l) for l in QA.open() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]
    print(f"{len(questions)} questions, strategy={a.strategy}, k={a.k}")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMBED_MODEL, device="cuda")
    embed = lambda q: str(list(map(float, emb.encode(q, normalize_embeddings=True))))

    print(f"loading {MODEL_ID} in 4-bit...")
    model, proc = load_model()

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"router__{a.strategy}__k{a.k}.jsonl"

    with psycopg.connect(DSN) as conn, conn.cursor() as cur, out_path.open("w") as fh:
        cur.execute("SELECT DISTINCT vendor FROM documents "
                    "WHERE dataset=%s AND arm=%s AND vendor IS NOT NULL",
                    (DATASET, ARM))
        known_vendors = [r[0] for r in cur.fetchall()]
        print(f"{len(known_vendors)} distinct vendors in store")

        for i, q in enumerate(questions, 1):
            raw, ms_route = generate(model, proc, ROUTE_SYSTEM,
                                     q["question"], max_new_tokens=64)
            intent = parse_intent(raw)
            action = (intent or {}).get("action", "lookup")
            route = "sql" if action in SQL_ACTIONS else "vector"

            rec = {
                "qid": q["qid"], "kind": q["kind"], "field": q["field"],
                "question": q["question"], "gold_answer": q["gold_answer"],
                "gold_docs": q["gold_docs"], "model": MODEL_ID,
                "strategy": a.strategy, "k": a.k,
                "route": route, "intent": intent, "intent_raw": raw[:200],
                # was the route the right one for this question type?
                "route_correct": (route == "sql") == (q["kind"] == "aggregate"),
                "latency_route_ms": ms_route,
            }

            if route == "sql":
                try:
                    val, sql, err = run_sql(cur, intent, known_vendors)
                except Exception as exc:                    # noqa: BLE001
                    conn.rollback()
                    val, sql, err = None, None, str(exc)[:200]
                rec.update({
                    "sql": sql, "sql_error": err,
                    "model_answer": fmt(val, action),
                    "retrieved_docs": [], "similarities": [],
                    "top_similarity": None, "retrieval_hit": None,
                    "answer_grounded": None, "latency_ms": ms_route,
                    "cost_usd": 0.0, "error": None,
                })
            else:
                qvec = embed(q["question"])
                hits = retrieve(cur, qvec, a.strategy, a.k)
                context = "\n\n---\n\n".join(
                    f"[receipt {d}]\n{t}" for d, t, _ in hits)
                docs = [h[0] for h in hits]
                sims = [round(float(h[2]), 4) for h in hits]
                try:
                    ans, ms_ans = generate(model, proc, ANSWER_SYSTEM,
                                           f"Receipts:\n\n{context}\n\n"
                                           f"Question: {q['question']}",
                                           max_new_tokens=64)
                    err = None
                except Exception as exc:                    # noqa: BLE001
                    ans, ms_ans, err = None, 0, str(exc)[:200]
                rec.update({
                    "sql": None, "sql_error": None, "model_answer": ans,
                    "retrieved_docs": docs, "similarities": sims,
                    "top_similarity": sims[0] if sims else None,
                    "retrieval_hit": bool(set(docs) & set(q["gold_docs"])),
                    "answer_grounded": grounded(ans, context),
                    "latency_ms": ms_route + ms_ans,
                    "cost_usd": 0.0, "error": err,
                })

            fh.write(json.dumps(rec) + "\n")
            if i % 25 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}")

    print(f"\nwrote {out_path}")
    print("next: python layer3/score_rag.py " + str(out_path))


if __name__ == "__main__":
    main()
