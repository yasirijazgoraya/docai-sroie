"""
Interactive query tool -- ask anything, see the whole trace.

Complements the batch evaluation rather than replacing it. score_rag.py measures
accuracy over the 287 gold-derived questions; this shows what the pipeline
actually DOES for a single question, step by step, including the paths that
would otherwise be invisible in an aggregate score.

Its research use is finding failure modes the generated question set cannot
reach. The evaluation questions were built from gold annotations, so they always
name a vendor and a date -- real phrasing ("that hardware shop last month",
"the big Tesco one") is exactly what the benchmark does not test, and is the
main stated limitation of the structured-first result. Type those here and watch
where resolution breaks.

    python layer3/ask.py                          # interactive
    python layer3/ask.py "How much did I spend at AEON?"
    python layer3/ask.py --llm mistral            # cloud answering model

Commands inside the session:  \\stats   store summary
                              \\vendors <substring>
                              \\quit
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]

REGION = "eu-west-1"
MISTRAL_ID = "mistral.mistral-large-2402-v1:0"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DATASET, ARM_DB = "sroie", "zeroshot"

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

ANSWER_SYSTEM = (
    "You answer questions about receipts using ONLY the receipt text provided. "
    "Give the shortest possible answer: a number, a date, or an address. "
    "Do not explain, do not show working, do not add currency symbols. "
    "If the answer is not present in the provided receipts, reply exactly: NOT_FOUND"
)

SQL_ACTIONS = {"sum_vendor", "count_vendor", "max_vendor", "sum_year", "count_year"}
STOP = {"sdn", "bhd", "co", "ltd", "the", "and", "m", "s", "b", "enterprise"}

DIM, BOLD, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")


def step(label, value, colour=""):
    print(f"  {DIM}{label:<11}{RESET}{colour}{value}{RESET}")


# ---------------------------------------------------------------- vendor idx

def tokens(s):
    t = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()
    return {x for x in t if x and x not in STOP}


class VendorIndex:
    """IDF-weighted matching: generic tokens (hardware, cash, restoran) recur
    across vendors and must not drive a match; rare ones (unihakka) must."""

    def __init__(self, known):
        self.known = known
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
        return out


# --------------------------------------------------------------------- LLMs

class MistralLLM:
    name = MISTRAL_ID

    def __init__(self):
        import boto3
        self.c = boto3.client("bedrock-runtime", region_name=REGION)

    def __call__(self, system, user, max_tokens=110):
        prompt = f"<s>[INST] {system}\n\n{user} [/INST]"
        r = self.c.invoke_model(modelId=MISTRAL_ID, body=json.dumps(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}))
        return json.loads(r["body"].read())["outputs"][0]["text"].strip()


class QwenLLM:
    name = QWEN_ID

    def __init__(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as V
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as V
        self.torch = torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        self.proc = AutoProcessor.from_pretrained(QWEN_ID)
        self.model = V.from_pretrained(QWEN_ID, quantization_config=bnb,
                                       device_map="auto")
        self.model.eval()

    def __call__(self, system, user, max_tokens=110):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.proc.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ------------------------------------------------------------------ answering

def parse_intent(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and "action" in d else None


def answer(question, cur, vidx, llm, embed):
    t0 = time.time()

    raw = llm(INTENT_SYSTEM, question, max_tokens=110)
    intent = parse_intent(raw)
    if intent is None:
        step("intent", f"unparseable: {raw[:60]}", RED)
        intent = {"action": "lookup"}
    else:
        step("intent", json.dumps(intent))

    action = intent.get("action", "lookup")

    # ---- aggregate: SQL over the structured store ------------------------
    if action in SQL_ACTIONS:
        step("route", "SQL (aggregate)", GREEN)
        if action in ("sum_vendor", "count_vendor", "max_vendor"):
            vendors = vidx.match(intent.get("vendor"))
            if not vendors:
                step("vendor", f"not resolved: {intent.get('vendor')!r}", RED)
                print(f"  {BOLD}answer     I don't have receipts from that "
                      f"vendor.{RESET}")
                return
            step("vendor", f"{len(vendors)} string(s): "
                           f"{', '.join(v[:34] for v in vendors[:3])}"
                           f"{' …' if len(vendors) > 3 else ''}")
            agg = {"sum_vendor": "sum(total)", "count_vendor": "count(*)",
                   "max_vendor": "max(total)"}[action]
            cur.execute(f"SELECT {agg}, count(*) FROM documents "
                        f"WHERE dataset=%s AND arm=%s AND vendor = ANY(%s)",
                        (DATASET, ARM_DB, vendors))
            val, n = cur.fetchone()
            step("sql", f"SELECT {agg} … vendor = ANY(…)  → {n} receipts")
        else:
            year = intent.get("year")
            if not year:
                step("year", "no year in question", RED)
                print(f"  {BOLD}answer     Which year?{RESET}")
                return
            agg = "sum(total)" if action == "sum_year" else "count(*)"
            cur.execute(f"SELECT {agg}, count(*) FROM documents "
                        f"WHERE dataset=%s AND arm=%s "
                        f"AND extract(year from doc_date)=%s",
                        (DATASET, ARM_DB, int(year)))
            val, n = cur.fetchone()
            step("sql", f"SELECT {agg} … year={year}  → {n} receipts")

        if val is None:
            out = "No matching receipts."
        elif action in ("count_vendor", "count_year"):
            out = str(int(val))
        else:
            out = f"{float(val):.2f}"
        print(f"  {BOLD}answer     {GREEN}{out}{RESET}")
        step("elapsed", f"{(time.time()-t0)*1000:.0f} ms")
        return

    # ---- lookup: resolve the receipt, then read it -----------------------
    vendors = vidx.match(intent.get("vendor")) if intent.get("vendor") else []
    date, amount = intent.get("date"), intent.get("amount")
    docs, how = [], "unresolved"

    if vendors and date:
        cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                    "AND vendor = ANY(%s) AND doc_date=%s",
                    (DATASET, ARM_DB, vendors, date))
        docs = [r[0] for r in cur.fetchall()]
        how = "vendor+date" if docs else how
    if not docs and vendors and amount is not None:
        try:
            cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                        "AND vendor = ANY(%s) AND abs(total-%s)<=0.01",
                        (DATASET, ARM_DB, vendors, float(amount)))
            docs = [r[0] for r in cur.fetchall()]
            how = "vendor+amount" if docs else how
        except (TypeError, ValueError):
            pass
    if not docs and vendors:
        cur.execute("SELECT doc_id FROM documents WHERE dataset=%s AND arm=%s "
                    "AND vendor = ANY(%s) LIMIT 5", (DATASET, ARM_DB, vendors))
        docs = [r[0] for r in cur.fetchall()]
        how = "vendor only" if docs else how

    if docs:
        step("route", f"structured ({how})", GREEN)
        step("receipts", ", ".join(docs[:5]) + (" …" if len(docs) > 5 else ""))
        cur.execute("SELECT doc_id, text FROM chunks WHERE dataset=%s "
                    "AND strategy='whole' AND doc_id = ANY(%s)", (DATASET, docs))
        rows = cur.fetchall()
        want = intent.get("want")
        col = {"total": "total", "date": "doc_date", "address": "address",
               "vendor": "vendor"}.get(want)
        if col:
            cur.execute(f"SELECT {col} FROM documents WHERE dataset=%s AND arm=%s "
                        f"AND doc_id = ANY(%s) LIMIT 1", (DATASET, ARM_DB, docs))
            r = cur.fetchone()
            if r and r[0] is not None:
                v = r[0]
                shown = (f"{float(v):.2f}" if col == "total"
                         else v.strftime("%d/%m/%Y") if col == "doc_date"
                         else str(v))
                step("database", f"{col} = {shown}")
    else:
        step("route", "dense retrieval (fallback)", YELLOW)
        qv = str(embed(question))
        cur.execute("SELECT doc_id, text, 1-(embedding <=> %s::vector) FROM chunks "
                    "WHERE dataset=%s AND strategy='whole' "
                    "ORDER BY embedding <=> %s::vector LIMIT 5",
                    (qv, DATASET, qv))
        hits = cur.fetchall()
        rows = [(h[0], h[1]) for h in hits]
        step("retrieved", ", ".join(f"{h[0]} ({h[2]:.2f})" for h in hits[:3]))

    if not rows:
        print(f"  {BOLD}answer     {YELLOW}Nothing found.{RESET}")
        return

    context = "\n\n---\n\n".join(f"[receipt {d}]\n{t}" for d, t in rows)
    ans = llm(ANSWER_SYSTEM,
              f"Receipts:\n\n{context}\n\nQuestion: {question}", max_tokens=64)
    colour = YELLOW if ans.upper().startswith("NOT_FOUND") else GREEN
    print(f"  {BOLD}answer     {colour}{ans}{RESET}")
    step("elapsed", f"{(time.time()-t0)*1000:.0f} ms")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="ask once and exit")
    ap.add_argument("--llm", choices=["qwen", "mistral"], default="mistral")
    a = ap.parse_args()

    print(f"{DIM}loading {a.llm}…{RESET}")
    llm = MistralLLM() if a.llm == "mistral" else QwenLLM()

    _enc = None

    def embed(text):
        nonlocal _enc
        if _enc is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(MINILM)
            _enc = lambda t: list(map(float, m.encode(t, normalize_embeddings=True)))
        return _enc(text)

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT vendor FROM documents WHERE dataset=%s "
                    "AND arm=%s AND vendor IS NOT NULL", (DATASET, ARM_DB))
        vidx = VendorIndex([r[0] for r in cur.fetchall()])
        cur.execute("SELECT count(*), min(doc_date), max(doc_date), sum(total) "
                    "FROM documents WHERE dataset=%s AND arm=%s", (DATASET, ARM_DB))
        n, d0, d1, tot = cur.fetchone()

        print(f"{DIM}{n} receipts · {len(vidx.known)} vendors · "
              f"{d0} to {d1} · total {float(tot):.2f}{RESET}")

        if a.question:
            q = " ".join(a.question)
            print(f"\n{BOLD}> {q}{RESET}")
            answer(q, cur, vidx, llm, embed)
            return

        print(f"{DIM}\\stats  \\vendors <text>  \\quit{RESET}\n")
        while True:
            try:
                q = input(f"{BOLD}> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q in ("\\quit", "\\q", "quit", "exit"):
                break
            if q == "\\stats":
                cur.execute("SELECT vendor, count(*), round(sum(total),2) "
                            "FROM documents WHERE dataset=%s AND arm=%s "
                            "GROUP BY vendor ORDER BY 3 DESC NULLS LAST LIMIT 10",
                            (DATASET, ARM_DB))
                for v, c, s in cur.fetchall():
                    print(f"  {str(s):>10}  {c:>3}  {v}")
                continue
            if q.startswith("\\vendors"):
                term = q[len("\\vendors"):].strip()
                cur.execute("SELECT vendor, count(*) FROM documents "
                            "WHERE dataset=%s AND arm=%s AND vendor ILIKE %s "
                            "GROUP BY vendor ORDER BY 2 DESC LIMIT 15",
                            (DATASET, ARM_DB, f"%{term}%"))
                rows = cur.fetchall()
                for v, c in rows:
                    print(f"  {c:>3}  {v}")
                if not rows:
                    print(f"  {DIM}no match{RESET}")
                continue
            try:
                answer(q, cur, vidx, llm, embed)
            except Exception as exc:                        # noqa: BLE001
                conn.rollback()
                print(f"  {RED}error: {str(exc)[:200]}{RESET}")
            print()


if __name__ == "__main__":
    main()
