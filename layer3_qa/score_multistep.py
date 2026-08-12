"""
Score the multi-step question set.

The main scorer (score_rag.py) assumes single-value answers matched by field
type. Multi-step answers are vendor names, rankings, or field pairs, so matching
differs: name answers are compared after stripping corporate suffixes and
punctuation, and list answers count as correct when every gold item appears.

Reports accuracy by question class, mean steps, latency, and cost -- the last
three matter because the agent's cost is the argument against using it where a
router suffices.

    python layer3/score_multistep.py layer3/runs/agent_multistep__k5.jsonl
    python layer3/score_multistep.py layer3/runs/*multistep*.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SUFFIXES = (" sdn. bhd.", " sdn bhd", " sdn.bhd", " s/b", " sdn. bhd",
            " (m) bhd", " bhd", " sdn", " co.", " ltd")


def norm_name(s):
    t = " ".join(str(s or "").lower().split())
    for suf in SUFFIXES:
        if t.endswith(suf):
            t = t[: -len(suf)]
            break
    return re.sub(r"[^a-z0-9]", "", t)


def norm_num(s):
    m = re.search(r"-?[\d,]*\.?\d+", str(s or "").replace(" ", ""))
    if not m:
        return None
    try:
        return round(float(m.group().replace(",", "")), 2)
    except ValueError:
        return None


def is_correct(kind, pred, gold):
    if pred is None:
        return False
    p = str(pred).strip()
    if not p or p.upper().startswith("NOT_FOUND"):
        return False

    # numeric gold: compare numerically with 1% tolerance
    gnum = norm_num(gold)
    if gnum is not None and re.fullmatch(r"[\d.,\s]+", str(gold).strip()):
        pnum = norm_num(p)
        return pnum is not None and abs(pnum - gnum) <= max(0.01, abs(gnum) * 0.01)

    # list gold (top-three): every item must appear
    if ";" in str(gold) and kind == "rank":
        items = [norm_name(x) for x in str(gold).split(";")]
        pn = norm_name(p)
        return all(it in pn for it in items if it)

    # chained two-field answer: both parts must appear
    if ";" in str(gold):
        parts = [x.strip() for x in str(gold).split(";")]
        pn = norm_name(p)
        return all(norm_name(x) in pn or norm_name(pn) in norm_name(x)
                   for x in parts if x)

    # name or address: containment either way, suffix-insensitive
    g, pn = norm_name(gold), norm_name(p)
    if not g or not pn:
        return False
    return g == pn or g in pn or pn in g


def score(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"{path}: empty")
        return

    for r in rows:
        r["_ok"] = is_correct(r["kind"], r.get("model_answer"), r["gold_answer"])

    name = Path(path).name
    print(f"\n{'='*66}\n{name}   ({len(rows)} questions)\n{'='*66}")

    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)

    print(f"\n{'class':<12}{'n':>4}{'accuracy':>11}{'steps':>8}{'ms':>9}{'cost':>10}")
    print("-" * 54)
    for k in sorted(by):
        sub = by[k]
        acc = sum(x["_ok"] for x in sub) / len(sub)
        st = sum(x.get("n_steps") or 0 for x in sub) / len(sub)
        ms = sum(x.get("latency_ms") or 0 for x in sub) / len(sub)
        cost = sum(x.get("cost_usd") or 0 for x in sub)
        print(f"{k:<12}{len(sub):>4}{acc:>11.3f}{st:>8.1f}{ms:>9.0f}{cost:>10.4f}")

    acc = sum(r["_ok"] for r in rows) / len(rows)
    st = sum(r.get("n_steps") or 0 for r in rows) / len(rows)
    ms = sum(r.get("latency_ms") or 0 for r in rows) / len(rows)
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    print("-" * 54)
    print(f"{'ALL':<12}{len(rows):>4}{acc:>11.3f}{st:>8.1f}{ms:>9.0f}{cost:>10.4f}")

    wrong = [r for r in rows if not r["_ok"]]
    if wrong:
        print(f"\nfailures ({len(wrong)}):")
        for r in wrong[:12]:
            print(f"  [{r['kind']:<8}] {r['question'][:58]}")
            print(f"     pred: {str(r.get('model_answer'))[:52]}")
            print(f"     gold: {str(r['gold_answer'])[:52]}")
            if r.get("error"):
                print(f"     err : {r['error'][:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for p in a.runs:
        score(p)
