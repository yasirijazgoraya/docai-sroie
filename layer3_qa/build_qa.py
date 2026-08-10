"""
Layer 3 -- build the SROIE question set from GOLD labels.

SROIE is an extraction benchmark with no questions, so Layer 3 needs its own
evaluation set. Questions are generated from the gold annotations, never from
stored predictions: if the answer key came from our own extractor, a Layer-1
extraction error would be scored as a correct Layer-3 answer.

Two classes, matching the two answer paths:

  lookup     answerable from ONE receipt   -> vector retrieval path
  aggregate  spans MANY receipts           -> SQL path
             (RAG is expected to fail these; measuring that failure is the
              point, and is what justifies routing rather than asserting it)

Lookup questions identify a receipt by vendor + date rather than by doc_id,
because an SME asking a question does not know internal document IDs. Vendors
with several receipts on the same date are skipped -- the question would be
ambiguous and would score a correct answer as wrong.

    cd /mnt/yasir_drive/E_DATA/ResearchPlan2/docvlm-rq1
    python layer3/build_qa.py
    python layer3/build_qa.py --n_lookup 150 --seed 7

Output: layer3/qa/sroie_qa.jsonl
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = Path("/mnt/yasir_drive/E_DATA/data/SROIE2019/test/entities")
OUT = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"

_DATE_FORMATS = [
    "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y",
    "%d.%m.%Y", "%d.%m.%y", "%d %b %Y", "%d %b %y", "%d %B %Y", "%d %B %y",
    "%Y/%m/%d", "%b %d %Y", "%B %d %Y", "%d%m%Y", "%Y%m%d",
    "%d/%b/%Y", "%d/%b/%y",
]


def parse_date(s):
    if not s:
        return None
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*$", "", str(s).strip())
    for f in _DATE_FORMATS:
        try:
            d = datetime.strptime(s, f)
            return d.replace(year=d.year + 2000) if d.year < 100 else d
        except ValueError:
            continue
    return None


def parse_money(s):
    if s is None:
        return None
    t = re.sub(r"[^\d.,\-]", "", str(s))
    if not t:
        return None
    if "," in t and "." in t:
        t = (t.replace(".", "").replace(",", ".")
             if t.rindex(",") > t.rindex(".") else t.replace(",", ""))
    elif "," in t:
        t = t.replace(",", ".") if len(t.split(",")[-1]) == 2 else t.replace(",", "")
    try:
        return round(float(t), 2)
    except ValueError:
        return None


def load_gold():
    if not GOLD.exists():
        raise SystemExit(f"gold not found: {GOLD}")
    out = {}
    for f in sorted(GOLD.glob("*.txt")):
        try:
            d = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        out[f.stem] = {
            "vendor": (d.get("company") or "").strip() or None,
            "date": parse_date(d.get("date")),
            "date_raw": d.get("date"),
            "total": parse_money(d.get("total")),
            "address": (d.get("address") or "").strip() or None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lookup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    gold = load_gold()
    print(f"{len(gold)} gold receipts")

    by_vendor = defaultdict(list)
    for doc_id, g in gold.items():
        if g["vendor"]:
            by_vendor[g["vendor"].lower()].append(doc_id)

    # (vendor, date) pairs that identify exactly one receipt
    pair_count = Counter(
        (g["vendor"].lower(), g["date"].date())
        for g in gold.values() if g["vendor"] and g["date"]
    )

    qa = []

    def add(kind, question, answer, docs, field):
        qa.append({
            "qid": f"{kind[:2]}{len(qa):04d}",
            "kind": kind,
            "question": question,
            "gold_answer": str(answer),
            "gold_docs": docs if isinstance(docs, list) else [docs],
            "field": field,
        })

    # ---- lookup: one receipt, identified by vendor + date -----------------
    candidates = []
    for doc_id, g in gold.items():
        if not (g["vendor"] and g["date"]):
            continue
        if pair_count[(g["vendor"].lower(), g["date"].date())] != 1:
            continue          # ambiguous: same vendor, same day, several receipts
        candidates.append(doc_id)

    rng.shuffle(candidates)
    for doc_id in candidates[: a.n_lookup]:
        g = gold[doc_id]
        d = g["date"].strftime("%d %B %Y")
        field = rng.choice(
            ["total"] * 3 + ["address"] * 2 + ["date"]
            if g["address"] else ["total"] * 3 + ["date"]
        )
        if field == "total" and g["total"] is not None:
            add("lookup", f"How much was the receipt from {g['vendor']} on {d}?",
                f"{g['total']:.2f}", doc_id, "total")
        elif field == "address" and g["address"]:
            add("lookup", f"What is the address on the {g['vendor']} receipt from {d}?",
                g["address"], doc_id, "address")
        elif field == "date":
            add("lookup", f"When was the {g['vendor']} receipt for "
                          f"{g['total']:.2f} issued?" if g["total"] else
                          f"When was the {g['vendor']} receipt issued?",
                g["date"].strftime("%d/%m/%Y"), doc_id, "date")

    # ---- aggregate: spans several receipts --------------------------------
    for vendor_lc, docs in by_vendor.items():
        if len(docs) < 3:
            continue
        totals = [gold[d]["total"] for d in docs]
        if any(t is None for t in totals):
            continue
        name = gold[docs[0]]["vendor"]
        add("aggregate", f"How much did I spend at {name} in total?",
            f"{sum(totals):.2f}", docs, "sum_total")
        add("aggregate", f"How many receipts do I have from {name}?",
            len(docs), docs, "count")
        add("aggregate", f"What was my largest single purchase at {name}?",
            f"{max(totals):.2f}", docs, "max_total")

    # spend by year, across all vendors
    by_year = defaultdict(list)
    for doc_id, g in gold.items():
        if g["date"] and g["total"] is not None:
            by_year[g["date"].year].append((doc_id, g["total"]))
    for year, rows in sorted(by_year.items()):
        if len(rows) < 5:
            continue
        add("aggregate", f"How much did I spend in total during {year}?",
            f"{sum(t for _, t in rows):.2f}", [d for d, _ in rows], "sum_year")
        add("aggregate", f"How many receipts do I have from {year}?",
            len(rows), [d for d, _ in rows], "count_year")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for q in qa:
            fh.write(json.dumps(q) + "\n")

    n_lk = sum(1 for q in qa if q["kind"] == "lookup")
    print(f"wrote {OUT}")
    print(f"  {n_lk} lookup, {len(qa) - n_lk} aggregate, {len(qa)} total")
    print("\nsamples:")
    for q in qa[:3] + [q for q in qa if q["kind"] == "aggregate"][:3]:
        print(f"  [{q['kind']:9s}] {q['question']}")
        print(f"              -> {q['gold_answer']}")


if __name__ == "__main__":
    main()
