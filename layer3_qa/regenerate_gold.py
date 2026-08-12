"""
Regenerate aggregate gold over CANONICAL vendor identities.

Why this exists: SROIE's gold company annotations contain OCR-level noise of
their own. The same shop appears in gold as both

    KEDAI PAPAN YEW CHUAN     (7 receipts)
    KEDA PAPAN YEW CHJAN      (1 receipt)

No answer key built on raw string equality can be coherent here: the store
merges the two (correctly -- it is one shop), so "how much did I spend at
KEDAI PAPAN YEW CHUAN?" has one true answer over 8 receipts, while string-equal
gold says 7. Meanwhile GARDENIA BAKERIES appears correctly in gold on all 31
receipts, so there the OLD gold (built from our extractions, which split 30/1)
was the stale side. The only consistent definition is:

    a vendor IS its canonical cluster, and the same clustering is applied to
    gold entities as to stored predictions.

This script rebuilds the aggregate gold answers in sroie_qa.jsonl on that
definition, using the same Canonicaliser as layer2/canonicalise_vendors.py
(imported, not copied). Lookup questions are untouched -- they reference one
receipt and have no grouping problem. Year aggregates are untouched -- no vendor
involved.

The existing run files embed the old gold, so score_rag.py must re-join gold by
qid: pass the QA file via QA_GOLD (patch applied alongside this script).

    python layer3/regenerate_gold.py          # writes sroie_qa.jsonl (+ backup)

No model re-runs are needed for the 287-question set.
"""
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "layer3" / "qa" / "sroie_qa.jsonl"
BACKUP = ROOT / "layer3" / "qa" / "sroie_qa_v1_rawvendor.jsonl"
ENTITIES = Path("/mnt/yasir_drive/E_DATA/data/SROIE2019/test/entities")

# import the canonicaliser from layer2 without making it a package
spec = importlib.util.spec_from_file_location(
    "canon", ROOT / "layer2" / "canonicalise_vendors.py")
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)

TEMPLATES = {
    "sum_total": re.compile(r"^How much did I spend at (.+) in total\?$"),
    "count": re.compile(r"^How many receipts do I have from (.+)\?$"),
    "max_total": re.compile(r"^What was my largest single purchase at (.+)\?$"),
}


def parse_money(s):
    m = re.search(r"-?[\d,]*\.?\d+", str(s or "").replace(" ", ""))
    return round(float(m.group().replace(",", "")), 2) if m else None


def main():
    if not ENTITIES.exists():
        raise SystemExit(f"gold entities not found: {ENTITIES}")

    # ---- load gold entities ----------------------------------------------
    gold = {}
    for f in sorted(ENTITIES.glob("*.txt")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        gold[f.stem] = {"company": (d.get("company") or "").strip(),
                        "total": parse_money(d.get("total"))}
    print(f"{len(gold)} gold entity files")

    # ---- cluster gold company strings with the SAME rule as the store ----
    counts = defaultdict(int)
    for g in gold.values():
        if g["company"]:
            counts[g["company"]] += 1
    vendors = sorted(counts.items(), key=lambda x: -x[1])
    groups = canon.Canonicaliser(vendors).cluster()

    cluster_of = {}
    for g in groups:
        members = [raw for raw, _ in g]
        for raw in members:
            cluster_of[raw] = tuple(sorted(members))

    merged = [g for g in groups if len(g) > 1]
    print(f"{len(vendors)} raw gold vendors -> {len(groups)} canonical "
          f"({len(merged)} merges)")
    for g in merged:
        names = sorted(g, key=lambda x: -x[1])
        print("  " + " | ".join(f"{r} ({n})" for r, n in names))

    # doc ids per canonical cluster
    docs_of = defaultdict(list)
    for doc_id, g in gold.items():
        if g["company"] in cluster_of:
            docs_of[cluster_of[g["company"]]].append(doc_id)

    # ---- rewrite aggregate gold -------------------------------------------
    rows = [json.loads(l) for l in QA.open() if l.strip()]
    changed = 0
    for r in rows:
        if r["kind"] != "aggregate" or r["field"] not in TEMPLATES:
            continue
        m = TEMPLATES[r["field"]].match(r["question"])
        if not m:
            continue
        vendor = m.group(1).strip()
        key = cluster_of.get(vendor)
        if key is None:
            # question vendor came verbatim from gold, so this should not
            # happen; leave the record alone and say so rather than guess.
            print(f"  WARNING unmatched vendor in question: {vendor!r}")
            continue
        docs = docs_of[key]
        totals = [gold[d]["total"] for d in docs if gold[d]["total"] is not None]
        if r["field"] == "sum_total":
            new = f"{sum(totals):.2f}"
        elif r["field"] == "count":
            new = str(len(docs))
        else:
            new = f"{max(totals):.2f}"
        if new != r["gold_answer"] or set(docs) != set(r.get("gold_docs", [])):
            changed += 1
            print(f"  {r['qid']} {r['field']:<9} {r['gold_answer']} -> {new}  "
                  f"({vendor[:40]})")
        r["gold_answer"] = new
        r["gold_docs"] = sorted(docs)

    if not BACKUP.exists():
        BACKUP.write_bytes(QA.read_bytes())
        print(f"\nbacked up old QA to {BACKUP.name}")
    with QA.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"rewrote {QA.name}: {changed} aggregate gold answers updated")


if __name__ == "__main__":
    main()
