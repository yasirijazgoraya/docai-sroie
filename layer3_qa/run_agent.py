"""
Layer 3, arm D -- AGENT (multi-step tool use).

The router answers one query per question. Compositional questions -- "did I
spend more at X or Y?", "which vendor did I spend the most with?" -- need two or
more queries and a comparison between the results, which the single-intent
schema cannot express. This arm gives the model tools and lets it decide how
many steps to take.

Two tools, deliberately narrow:

  aggregate(metric, vendor|year)   sum/count/max/min/avg over the structured store
  find_receipts(vendor, date, amount)  resolve specific receipts and read them

Neither accepts free-form SQL. The model chooses an operation and supplies
parameters; this code builds the parameterised query. A 7B-to-frontier model
authoring SQL against a live database is a liability in a product, and the
research question is whether multi-step REASONING helps, not whether the model
can write SQL.

Expected result, stated in advance so it is a prediction rather than a
rationalisation: on the 287 single-intent questions the agent should roughly
match the router while costing more steps and more latency -- agency buys
nothing where one query suffices. On the 36 multi-step questions the router
scores near zero by construction, so anything the agent achieves is the value
of agency itself.

    python layer3/run_agent.py --qa multistep --limit 5
    python layer3/run_agent.py --qa multistep
    python layer3/run_agent.py --qa single

Writes layer3/runs/agent_<qa>__k5.jsonl
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
QA_SINGLE = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"
QA_MULTI = ROOT / "layer3" / "qa" / "sroie_qa_multistep.jsonl"
RUNS = ROOT / "layer3" / "runs"

REGION = "eu-west-1"
LLM_ID = "mistral.mistral-large-2402-v1:0"
DATASET, ARM_DB = "sroie", "zeroshot"
PRICE_IN, PRICE_OUT = 4.00, 12.00
MAX_STEPS = 6

SYSTEM = """You answer questions about a business's receipts by calling tools.
Reply with ONLY a JSON object, no other text, no explanation.

To call a tool:
  {"tool":"aggregate","metric":"sum|count|max|min|avg","vendor":"NAME or null","year":YYYY or null}
  {"tool":"find_receipts","vendor":"NAME or null","date":"YYYY-MM-DD or null","amount":NUMBER or null}
  {"tool":"list_vendors","metric":"sum|count","limit":N}

To give the final answer:
  {"answer":"..."}

Rules:
- One JSON object per reply.
- Call tools one at a time; you will see each result before deciding the next.
- To compare two vendors, call aggregate twice, then answer.
- To find a top or bottom vendor, call list_vendors.
- Answer as briefly as possible: a number, a date, a name, or an address.
- Numbers: two decimal places, no currency symbol.
- Dates in questions are DAY/MONTH/YEAR (12/03/2018 is 12 March 2018).
  Convert to YYYY-MM-DD before calling a tool.
- find_receipts accepts amount alone; vendor is optional.
- For "biggest purchase in YEAR" use aggregate with metric "max" and that
  year, then find_receipts with the returned amount to identify the vendor.
- For "biggest purchase in YEAR" use aggregate with metric "max" and that
  year, then find_receipts with the returned amount to identify the vendor.
- If a tool returns nothing useful, answer {"answer":"NOT_FOUND"}."""

STOP = {"sdn", "bhd", "co", "ltd", "the", "and", "m", "s", "b", "enterprise"}


def tokens(s):
    t = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()
    return {x for x in t if x and x not in STOP}


class VendorIndex:
    """IDF-weighted matching: generic tokens (hardware, cash, restoran) recur
    across vendors and must not drive a match; rare ones (unihakka) must."""

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
        """Resolve a question's vendor string to canonical vendor_ids.

        Matching is still fuzzy on the QUESTION side -- a user types "Gardenia",
        not the full registered name -- but the grouping is now fixed: each
        matched string maps to a vendor_id resolved once at load time, so OCR
        variants no longer split a group and generic tokens no longer merge
        distinct companies.
        """
        want = tokens(name)
        if not want:
            return []
        w = sum(self.idf(t) for t in want)
        if w <= 0:
            return []
        ids = set()
        for v in self.known:
            have = tokens(v)
            if not have:
                continue
            i = sum(self.idf(t) for t in want & have)
            if i / w >= 0.75 and i / sum(self.idf(t) for t in have) >= 0.75:
                vid = self.vid.get(v)
                if vid is not None:
                    ids.add(vid)
        return sorted(ids)


def call_llm(client, messages, retries=4):
    """Bedrock Mistral takes a single prompt string, so the running transcript
    is flattened into one [INST] block each turn."""
    convo = "\n\n".join(messages)
    prompt = f"<s>[INST] {SYSTEM}\n\n{convo} [/INST]"
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.invoke_model(modelId=LLM_ID, body=json.dumps(
                {"prompt": prompt, "max_tokens": 200, "temperature": 0.0}))
            ms = (time.time() - t0) * 1000
            text = json.loads(r["body"].read())["outputs"][0]["text"].strip()
            return text, ms, len(prompt) // 4, len(text) // 4
        except Exception as exc:                            # noqa: BLE001
            if "Throttl" in str(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def parse_json(raw):
    m = re.search(r"\{.*?\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------- tools

def tool_aggregate(cur, vidx, call):
    metric = (call.get("metric") or "sum").lower()
    agg = {"sum": "sum(total)", "count": "count(*)", "max": "max(total)",
           "min": "min(total)", "avg": "avg(total)"}.get(metric)
    if not agg:
        return {"error": f"unknown metric {metric!r}"}

    vendor, year = call.get("vendor"), call.get("year")
    where = ["dataset=%s", "arm=%s"]
    params = [DATASET, ARM_DB]
    matched = None

    if vendor:
        matched = vidx.match(vendor)
        if not matched:
            return {"error": f"no vendor matching {vendor!r}"}
        where.append("vendor_id = ANY(%s)")
        params.append(matched)
    if year:
        try:
            where.append("extract(year from doc_date)=%s")
            params.append(int(year))
        except (TypeError, ValueError):
            return {"error": f"bad year {year!r}"}

    cur.execute(f"SELECT {agg}, count(*) FROM documents "
                f"WHERE {' AND '.join(where)}", params)
    val, n = cur.fetchone()
    if val is None:
        return {"result": None, "n_receipts": n, "note": "no matching receipts"}
    out = int(val) if metric == "count" else round(float(val), 2)
    r = {"metric": metric, "result": out, "n_receipts": n}
    if matched:
        r["matched_vendors"] = matched
    return r


def tool_list_vendors(cur, call):
    metric = (call.get("metric") or "sum").lower()
    agg = "sum(total)" if metric == "sum" else "count(*)"
    try:
        limit = max(1, min(int(call.get("limit") or 5), 15))
    except (TypeError, ValueError):
        limit = 5
    # group by canonical vendor_id so OCR variants aggregate as one company;
    # display the canonical name
    cur.execute(f"SELECT max(c.canon_name), round({agg.replace('total','d.total')}::numeric,2), count(*) "
                f"FROM documents d JOIN vendor_canon c ON c.vendor_id = d.vendor_id "
                f"AND c.dataset = d.dataset AND c.arm = d.arm AND c.raw_vendor = d.vendor "
                f"WHERE d.dataset=%s AND d.arm=%s "
                f"GROUP BY d.vendor_id ORDER BY 2 DESC NULLS LAST LIMIT %s",
                (DATASET, ARM_DB, limit))
    return {"metric": metric,
            "vendors": [{"vendor": v, "value": float(x), "n_receipts": n}
                        for v, x, n in cur.fetchall()]}


def tool_find_receipts(cur, vidx, call):
    vendor, date, amount = call.get("vendor"), call.get("date"), call.get("amount")
    where = ["dataset=%s", "arm=%s"]
    params = [DATASET, ARM_DB]
    if vendor:
        matched = vidx.match(vendor)
        if not matched:
            return {"error": f"no vendor matching {vendor!r}"}
        where.append("vendor_id = ANY(%s)")
        params.append(matched)
    if date:
        # A bare year is not a date. The model sometimes passes "2017" here
        # after being told to normalise dates; route it to the year filter
        # rather than letting Postgres reject it.
        d = str(date).strip()
        if re.fullmatch(r"\d{4}", d):
            where.append("extract(year from doc_date)=%s")
            params.append(int(d))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            where.append("doc_date=%s")
            params.append(d)
        else:
            return {"error": f"date must be YYYY-MM-DD, got {date!r}"}
    if amount is not None:
        try:
            where.append("abs(total-%s)<=0.01")
            params.append(float(amount))
        except (TypeError, ValueError):
            pass
    if len(where) == 2:
        return {"error": "give at least one of vendor, date, amount"}

    cur.execute(f"SELECT doc_id, vendor, to_char(doc_date,'DD/MM/YYYY'), "
                f"total, address FROM documents WHERE {' AND '.join(where)} "
                f"LIMIT 5", params)
    rows = cur.fetchall()
    if not rows:
        return {"receipts": [], "note": "no matching receipts"}
    return {"receipts": [
        {"doc_id": d, "vendor": v, "date": dt,
         "total": None if t is None else round(float(t), 2), "address": ad}
        for d, v, dt, t, ad in rows]}


def run_tool(cur, vidx, call):
    name = call.get("tool")
    if name == "aggregate":
        return tool_aggregate(cur, vidx, call)
    if name == "list_vendors":
        return tool_list_vendors(cur, call)
    if name == "find_receipts":
        return tool_find_receipts(cur, vidx, call)
    return {"error": f"unknown tool {name!r}"}


# --------------------------------------------------------------------- loop

def answer_question(question, cur, vidx, client):
    messages = [f"Question: {question}"]
    trace, cost, total_ms = [], 0.0, 0.0

    for step in range(MAX_STEPS):
        raw, ms, tin, tout = call_llm(client, messages)
        total_ms += ms
        cost += tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
        obj = parse_json(raw)

        if obj is None:
            trace.append({"step": step, "unparseable": raw[:150]})
            messages.append("Your reply was not valid JSON. Reply with a single "
                            "JSON object only.")
            continue

        if "answer" in obj:
            return str(obj["answer"]).strip(), trace, step + 1, total_ms, cost

        result = run_tool(cur, vidx, obj)
        trace.append({"step": step, "call": obj, "result": result})
        messages.append(f"Tool call: {json.dumps(obj)}")
        messages.append(f"Tool result: {json.dumps(result, default=str)[:900]}")

    # ran out of steps: ask once for a final answer from what it has
    messages.append("Give your final answer now as {\"answer\":\"...\"}.")
    raw, ms, tin, tout = call_llm(client, messages)
    total_ms += ms
    cost += tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    obj = parse_json(raw) or {}
    return (str(obj.get("answer", "NOT_FOUND")).strip(), trace, MAX_STEPS,
            total_ms, cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", choices=["single", "multistep"], default="multistep")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    path = QA_MULTI if a.qa == "multistep" else QA_SINGLE
    questions = [json.loads(l) for l in path.open() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]
    print(f"{len(questions)} questions from {path.name} | agent | {LLM_ID}")

    client = boto3.client("bedrock-runtime", region_name=REGION)
    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"agent_{a.qa}__k5.jsonl"

    total_cost, steps_used = 0.0, []
    with psycopg.connect(DSN) as conn, conn.cursor() as cur, out_path.open("w") as fh:
        cur.execute("SELECT DISTINCT vendor, vendor_id FROM documents "
                    "WHERE dataset=%s AND arm=%s AND vendor IS NOT NULL",
                    (DATASET, ARM_DB))
        _rows = cur.fetchall()
        vidx = VendorIndex([r[0] for r in _rows],
                           {r[0]: r[1] for r in _rows})

        for i, q in enumerate(questions, 1):
            try:
                ans, trace, n_steps, ms, cost = answer_question(
                    q["question"], cur, vidx, client)
                err = None
            except Exception as exc:                        # noqa: BLE001
                conn.rollback()
                ans, trace, n_steps, ms, cost = None, [], 0, 0, 0.0
                err = str(exc)[:300]

            total_cost += cost
            steps_used.append(n_steps)
            fh.write(json.dumps({
                "qid": q["qid"], "kind": q["kind"],
                "field": q.get("field", q["kind"]),
                "question": q["question"], "gold_answer": q["gold_answer"],
                "gold_docs": q.get("gold_docs", []),
                "model": LLM_ID, "arm": "agent",
                "model_answer": ans, "n_steps": n_steps, "trace": trace,
                "retrieved_docs": [], "similarities": [], "top_similarity": None,
                "retrieval_hit": None, "answer_grounded": None,
                "latency_ms": ms, "cost_usd": round(cost, 6), "error": err,
                "note": q.get("note", ""),
            }) + "\n")

            if i % 10 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}  "
                      f"avg steps {sum(steps_used)/len(steps_used):.1f}  "
                      f"cost USD {total_cost:.3f}")

    print(f"\nwrote {out_path}")
    print(f"mean steps {sum(steps_used)/max(len(steps_used),1):.2f} | "
          f"estimated cost USD {total_cost:.3f}")


if __name__ == "__main__":
    main()
