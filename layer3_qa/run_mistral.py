"""
Layer 3 -- Mistral Large as the answering model.

This is the LLM axis of H3, and the only axis not yet varied. Retrieval design
has been changed three ways with the answering model held constant:

    baseline (MiniLM, fixed RAG)            overall 0.223
    + stronger embeddings (Titan v2)        overall 0.488   (+0.265)
    + structured-first resolution           overall 0.808   (+0.585)

H3 claims retrieval design dominates LLM choice. Without changing the LLM, that
claim is untested: a reviewer can reasonably ask whether a stronger answering
model would have closed the same gap. This run supplies the comparison span.

Mistral Large 2402 is chosen over another Anthropic model deliberately -- a
different vendor and training lineage is a stronger test of "LLM choice" than a
different size within one family.

Everything else is held fixed: same questions, same k, same retrieval indexes,
same prompt, same answer parsing. Only the generating model changes.

    python layer3/run_mistral.py --arm a --embed minilm      # vs rag_local
    python layer3/run_mistral.py --arm a --embed titan       # vs rag_titan
    python layer3/run_mistral.py --arm c                     # vs hybrid

Region eu-west-1; Mistral is callable there directly with no cross-region
inference profile. Bedrock's Mistral API uses prompt/[INST] rather than the
messages format, so the call wrapper differs from the Claude scripts.

Cost: roughly 287 calls per run at a few thousand input tokens each.
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import boto3
import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"
RUNS = ROOT / "layer3" / "runs"

REGION = "eu-west-1"
LLM_ID = "mistral.mistral-large-2402-v1:0"
TITAN_ID = "amazon.titan-embed-text-v2:0"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DATASET, ARM_DB = "sroie", "zeroshot"

# Mistral Large 2402 published rates, USD per 1M tokens.
PRICE_IN, PRICE_OUT = 4.00, 12.00

ANSWER_SYSTEM = (
    "You answer questions about receipts using ONLY the receipt text provided. "
    "Give the shortest possible answer: a number, a date, or an address. "
    "Do not explain, do not show working, do not add currency symbols. "
    "If the answer is not present in the provided receipts, reply exactly: NOT_FOUND"
)

INTENT_SYSTEM = """You convert a question about receipts into a JSON intent.

Reply with ONLY a JSON object, no other text. Fields:
  "action": one of
      "sum_vendor"    total spend at one vendor
      "count_vendor"  how many receipts from one vendor
      "max_vendor"    largest single purchase at one vendor
      "sum_year"      total spend in a year
      "count_year"    how many receipts in a year
      "lookup"        anything about ONE specific receipt
  "vendor": company name if named, else null
  "year": four-digit year if named, else null
  "date": full date as YYYY-MM-DD if the question gives one, else null
  "amount": money amount if the question gives one, else null
  "want": for lookup only: "total" | "date" | "address" | "vendor" | null

Examples:
Q: How much did I spend at ABC TRADING in total?
{"action":"sum_vendor","vendor":"ABC TRADING","year":null,"date":null,"amount":null,"want":null}
Q: How much was the receipt from XYZ SDN BHD on 24 June 2018?
{"action":"lookup","vendor":"XYZ SDN BHD","year":2018,"date":"2018-06-24","amount":null,"want":"total"}
Q: When was the ACME receipt for 6.20 issued?
{"action":"lookup","vendor":"ACME","year":null,"date":null,"amount":6.20,"want":"date"}"""

SQL_ACTIONS = {"sum_vendor", "count_vendor", "max_vendor", "sum_year", "count_year"}
STOP = {"sdn", "bhd", "co", "ltd", "the", "and", "m", "s", "b", "enterprise"}


# ----------------------------------------------------------------- bedrock

def call_mistral(client, system, user, max_tokens=96, retries=4):
    """Bedrock Mistral uses prompt/[INST], not the messages format."""
    prompt = f"<s>[INST] {system}\n\n{user} [/INST]"
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.invoke_model(
                modelId=LLM_ID,
                body=json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                                 "temperature": 0.0}))
            ms = (time.time() - t0) * 1000
            body = json.loads(r["body"].read())
            text = body["outputs"][0]["text"].strip()
            # Bedrock's Mistral response carries no usage block; approximate
            # from characters so the cost column is populated but clearly
            # marked as an estimate in the run record.
            tin = len(prompt) // 4
            tout = len(text) // 4
            return text, ms, tin, tout
        except Exception as exc:                            # noqa: BLE001
            if "Throttl" in str(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def titan_embed(client, text, retries=4):
    for attempt in range(retries):
        try:
            r = client.invoke_model(
                modelId=TITAN_ID,
                body=json.dumps({"inputText": text[:8000], "dimensions": 1024,
                                 "normalize": True}))
            return json.loads(r["body"].read())["embedding"]
        except Exception as exc:                            # noqa: BLE001
            if "Throttl" in str(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# ------------------------------------------------------------------ shared

def norm_vendor(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


def tokens(s):
    return {t for t in norm_vendor(s).split() if t and t not in STOP}


class VendorIndex:
    """IDF-weighted vendor matching -- generic tokens (hardware, cash, carry)
    must not drive a match; rare tokens (unihakka, kuchai) must."""

    def __init__(self, known, vid=None):
        self.known = known
        self.vid = vid or {}
        self.df = {}
        for v in known:
            for t in tokens(v):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = max(len(known), 1)

    def idf(self, t):
        return math.log(self.n / (1 + self.df.get(t, 0))) + 1.0

    def match(self, name):
        want = tokens(name)
        if not want:
            return []
        w = sum(self.idf(t) for t in want)
        if w <= 0:
            return []
        out = []
        for v in self.known:
            have = tokens(v)
            if not have:
                continue
            inter = sum(self.idf(t) for t in want & have)
            if inter / w >= 0.75 and inter / sum(self.idf(t) for t in have) >= 0.75:
                out.append(v)
        if self.vid:
            return sorted({self.vid[v] for v in out if v in self.vid})
        return out


def parse_intent(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and "action" in d else None


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


def fmt(val, action):
    if val is None:
        return "NOT_FOUND"
    if action in ("count_vendor", "count_year"):
        return str(int(val))
    return f"{float(val):.2f}"


def sql_aggregate(cur, intent, vidx):
    action, vq, year = intent.get("action"), intent.get("vendor"), intent.get("year")
    if action in ("sum_vendor", "count_vendor", "max_vendor"):
        vendors = vidx.match(vq)
        if not vendors:
            return None, f"vendor not resolved: {vq!r}"
        agg = {"sum_vendor": "sum(total)", "count_vendor": "count(*)",
               "max_vendor": "max(total)"}[action]
        cur.execute(f"SELECT {agg} FROM documents WHERE dataset=%s AND arm=%s "
                    f"AND vendor_id = ANY(%s)", (DATASET, ARM_DB, vendors))
        return cur.fetchone()[0], None
    if action in ("sum_year", "count_year"):
        if not year:
            return None, "no year"
        agg = "sum(total)" if action == "sum_year" else "count(*)"
        cur.execute(f"SELECT {agg} FROM documents WHERE dataset=%s AND arm=%s "
                    f"AND extract(year from doc_date)=%s",
                    (DATASET, ARM_DB, int(year)))
        return cur.fetchone()[0], None
    return None, f"unroutable: {action}"


def resolve_docs(cur, intent, vidx):
    vendors = vidx.match(intent.get("vendor")) if intent.get("vendor") else []
    date, amount = intent.get("date"), intent.get("amount")
    if vendors and date:
        cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                    "AND vendor_id = ANY(%s) AND doc_date=%s",
                    (DATASET, ARM_DB, vendors, date))
        rows = [r[0] for r in cur.fetchall()]
        if rows:
            return rows, "vendor+date"
    if vendors and amount is not None:
        try:
            cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                        "AND vendor_id = ANY(%s) AND abs(total-%s)<=0.01",
                        (DATASET, ARM_DB, vendors, float(amount)))
            rows = [r[0] for r in cur.fetchall()]
            if rows:
                return rows, "vendor+amount"
        except (TypeError, ValueError):
            pass
    if vendors:
        cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                    "AND vendor_id = ANY(%s) LIMIT 5", (DATASET, ARM_DB, vendors))
        rows = [r[0] for r in cur.fetchall()]
        if rows:
            return rows, "vendor only"
    return [], "unresolved"


def fetch_text(cur, doc_ids):
    cur.execute("SELECT doc_id, text FROM chunks WHERE dataset=%s "
                "AND strategy='whole' AND doc_id = ANY(%s)", (DATASET, doc_ids))
    return cur.fetchall()


def vector_search(cur, qvec, k, table):
    cur.execute(f"SELECT doc_id, text, 1-(embedding <=> %s::vector) FROM {table} "
                f"WHERE dataset=%s AND strategy='whole' "
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (qvec, DATASET, qvec, k))
    return cur.fetchall()


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "c"], default="a",
                    help="a = fixed RAG, c = structured-first hybrid")
    ap.add_argument("--embed", choices=["minilm", "titan"], default="minilm")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    questions = [json.loads(l) for l in QA.open() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]

    table = "chunks_titan" if a.embed == "titan" else "chunks"
    print(f"{len(questions)} questions | arm={a.arm} | embed={a.embed} "
          f"| llm={LLM_ID}")

    client = boto3.client("bedrock-runtime", region_name=REGION)

    local_emb = None
    if a.embed == "minilm":
        from sentence_transformers import SentenceTransformer
        local_emb = SentenceTransformer(MINILM, device="cuda")

    def embed(text):
        if a.embed == "titan":
            return str(titan_embed(client, text))
        return str(list(map(float, local_emb.encode(text,
                                                    normalize_embeddings=True))))

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"mistral_{a.arm}_{a.embed}__whole__k{a.k}.jsonl"

    total_cost, n_err = 0.0, 0
    with psycopg.connect(DSN) as conn, conn.cursor() as cur, out_path.open("w") as fh:
        cur.execute("SELECT DISTINCT vendor, vendor_id FROM documents "
                    "WHERE dataset=%s AND arm=%s AND vendor IS NOT NULL",
                    (DATASET, ARM_DB))
        _rows = cur.fetchall()
        vidx = VendorIndex([r[0] for r in _rows],
                           {r[0]: r[1] for r in _rows})

        for i, q in enumerate(questions, 1):
            rec = {"qid": q["qid"], "kind": q["kind"], "field": q["field"],
                   "question": q["question"], "gold_answer": q["gold_answer"],
                   "gold_docs": q["gold_docs"], "model": LLM_ID,
                   "embed_model": a.embed, "strategy": "whole", "k": a.k}
            cost = 0.0

            try:
                if a.arm == "a":
                    qv = embed(q["question"])
                    hits = vector_search(cur, qv, a.k, table)
                    docs = [h[0] for h in hits]
                    sims = [round(float(h[2]), 4) for h in hits]
                    context = "\n\n---\n\n".join(
                        f"[receipt {d}]\n{t}" for d, t, _ in hits)
                    ans, ms, tin, tout = call_mistral(
                        client, ANSWER_SYSTEM,
                        f"Receipts:\n\n{context}\n\nQuestion: {q['question']}",
                        max_tokens=64)
                    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                    rec.update({"route": "vector", "resolution": "dense",
                                "intent": None, "route_correct": None,
                                "model_answer": ans, "db_answer": None,
                                "retrieved_docs": docs, "similarities": sims,
                                "top_similarity": sims[0] if sims else None,
                                "retrieval_hit": bool(set(docs) & set(q["gold_docs"])),
                                "answer_grounded": grounded(ans, context),
                                "latency_ms": ms, "error": None})
                else:
                    raw, ms_i, ti, to = call_mistral(client, INTENT_SYSTEM,
                                                     q["question"], max_tokens=110)
                    cost += ti / 1e6 * PRICE_IN + to / 1e6 * PRICE_OUT
                    intent = parse_intent(raw) or {"action": "lookup"}
                    action = intent.get("action", "lookup")
                    rec.update({"intent": intent, "intent_raw": raw[:200],
                                "route_correct": (action in SQL_ACTIONS)
                                                 == (q["kind"] == "aggregate")})

                    if action in SQL_ACTIONS:
                        val, err = sql_aggregate(cur, intent, vidx)
                        rec.update({"route": "sql", "resolution": "aggregate",
                                    "model_answer": fmt(val, action),
                                    "db_answer": None, "retrieved_docs": [],
                                    "similarities": [], "top_similarity": None,
                                    "retrieval_hit": None,
                                    "answer_grounded": None,
                                    "latency_ms": ms_i, "error": err})
                    else:
                        doc_ids, how = resolve_docs(cur, intent, vidx)
                        sims = []
                        if doc_ids:
                            rows = fetch_text(cur, doc_ids)
                            docs = [r[0] for r in rows]
                            context = "\n\n---\n\n".join(
                                f"[receipt {d}]\n{t}" for d, t in rows)
                        else:
                            rows = vector_search(cur, embed(q["question"]),
                                                 a.k, table)
                            docs = [r[0] for r in rows]
                            sims = [round(float(r[2]), 4) for r in rows]
                            context = "\n\n---\n\n".join(
                                f"[receipt {d}]\n{t}" for d, t, _ in rows)
                            how = "vector fallback"
                        ans, ms_a, tin, tout = call_mistral(
                            client, ANSWER_SYSTEM,
                            f"Receipts:\n\n{context}\n\nQuestion: {q['question']}",
                            max_tokens=64)
                        cost += tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                        rec.update({"route": "structured" if doc_ids else "vector",
                                    "resolution": how, "model_answer": ans,
                                    "db_answer": None, "retrieved_docs": docs,
                                    "similarities": sims,
                                    "top_similarity": sims[0] if sims else None,
                                    "retrieval_hit": bool(set(docs) & set(q["gold_docs"])),
                                    "answer_grounded": grounded(ans, context),
                                    "latency_ms": ms_i + ms_a, "error": None})
            except Exception as exc:                        # noqa: BLE001
                conn.rollback()
                n_err += 1
                rec.setdefault("model_answer", None)
                rec.setdefault("retrieved_docs", [])
                rec.setdefault("similarities", [])
                rec.setdefault("top_similarity", None)
                rec.setdefault("retrieval_hit", None)
                rec.setdefault("answer_grounded", None)
                rec.setdefault("latency_ms", 0)
                rec["error"] = str(exc)[:300]

            total_cost += cost
            rec["cost_usd"] = round(cost, 6)
            fh.write(json.dumps(rec) + "\n")

            if i % 25 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}  cost so far USD {total_cost:.3f}")

    print(f"\nwrote {out_path}")
    print(f"estimated cost: USD {total_cost:.3f} (token counts approximated "
          f"from characters; Bedrock Mistral returns no usage block)")
    if n_err:
        print(f"WARNING: {n_err} errors")


if __name__ == "__main__":
    main()
