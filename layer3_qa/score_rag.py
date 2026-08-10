"""
Layer 3 -- score a RAG run.

Separate pass over layer3/runs/*.jsonl, so metrics can change without
re-running generation (same convention as Layer 1's 05_score.py).

Reports:
  * answer accuracy, split by question kind (lookup vs aggregate)
  * retrieval hit rate -- was a gold document in the top k
  * outcome breakdown, which is what makes H4 testable:
        correct                 answered and right
        llm_error               right receipt retrieved, wrong answer
        retrieval_error         wrong receipt retrieved, answer honest to it
        hallucination           answer appears in NO retrieved chunk
        abstained               said NOT_FOUND
  * accuracy by similarity bin -- DERIVES the abstention threshold instead of
    guessing it
  * coverage / selective-accuracy curve at each candidate threshold

    python layer3/score_rag.py layer3/runs/rag_local__whole__k5.jsonl
    python layer3/score_rag.py layer3/runs/*.jsonl        # compare strategies
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

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
            return (d.year + 2000 if d.year < 100 else d.year, d.month, d.day)
        except ValueError:
            continue
    m = re.search(r"(\d{1,2})[/\-. ](\d{1,2})[/\-. ](\d{2,4})", str(s))
    if m:
        dd, mm, yy = (int(x) for x in m.groups())
        return (yy + 2000 if yy < 100 else yy, mm, dd)
    return None


def parse_money(s):
    if s is None:
        return None
    m = re.findall(r"\d[\d,]*\.?\d*", str(s).replace(" ", ""))
    if not m:
        return None
    try:
        return round(float(m[0].replace(",", "")), 2)
    except ValueError:
        return None


def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def is_correct(field, pred, gold):
    """Field-appropriate matching, mirroring the Layer-1 conventions."""
    if pred is None:
        return False
    p = str(pred).strip()
    if p == "" or p.upper().startswith("NOT_FOUND"):
        return False

    if field in ("total", "sum_total", "max_total", "sum_year"):
        pv, gv = parse_money(p), parse_money(gold)
        if pv is None or gv is None:
            return False
        return abs(pv - gv) <= max(0.01, abs(gv) * 0.01)      # 1% tolerance

    if field in ("count", "count_year"):
        pv = re.search(r"\d+", p)
        return bool(pv) and int(pv.group()) == int(float(gold))

    if field == "date":
        return parse_date(p) is not None and parse_date(p) == parse_date(gold)

    pn, gn = norm_text(p), norm_text(gold)          # address, vendor
    if not pn or not gn:
        return False
    return pn == gn or gn in pn or pn in gn


def classify(r, correct):
    ans = (r.get("model_answer") or "").strip()
    # Mistral appends an explanation after the abstention token;
    # treat any answer starting with NOT_FOUND as an abstention so
    # the outcome split is comparable across models.
    if ans.upper().startswith("NOT_FOUND"):
        ans = "NOT_FOUND"
    if r.get("error"):
        return "error"
    if ans == "" or ans == "NOT_FOUND":
        return "abstained"
    if correct:
        return "correct"
    if r.get("answer_grounded") is False:
        return "hallucination"
    if r.get("retrieval_hit"):
        return "llm_error"
    return "retrieval_error"


def score(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"{path}: empty")
        return

    for r in rows:
        r["_correct"] = is_correct(r["field"], r.get("model_answer"),
                                   r["gold_answer"])
        r["_outcome"] = classify(r, r["_correct"])

    name = Path(path).name
    print(f"\n{'=' * 68}\n{name}   ({len(rows)} questions)\n{'=' * 68}")

    # ---- by question kind -------------------------------------------------
    print(f"\n{'kind':<12}{'n':>5}{'accuracy':>11}{'retr.hit':>11}{'abstain':>10}")
    print("-" * 49)
    for kind in ("lookup", "aggregate"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        acc = sum(r["_correct"] for r in sub) / len(sub)
        vec = [r for r in sub if r["retrieval_hit"] is not None]
        hit = (sum(r["retrieval_hit"] for r in vec) / len(vec)) if vec else float("nan")
        abst = sum(r["_outcome"] == "abstained" for r in sub) / len(sub)
        print(f"{kind:<12}{len(sub):>5}{acc:>11.3f}{hit:>11.3f}{abst:>10.3f}")
    acc = sum(r["_correct"] for r in rows) / len(rows)
    vecall = [r for r in rows if r["retrieval_hit"] is not None]
    hit = (sum(r["retrieval_hit"] for r in vecall) / len(vecall)) if vecall else float("nan")
    abst = sum(r["_outcome"] == "abstained" for r in rows) / len(rows)
    print(f"{'ALL':<12}{len(rows):>5}{acc:>11.3f}{hit:>11.3f}{abst:>10.3f}")

    # ---- outcome breakdown ------------------------------------------------
    print("\noutcomes")
    print("-" * 49)
    c = Counter(r["_outcome"] for r in rows)
    for k in ("correct", "abstained", "retrieval_error", "llm_error",
              "hallucination", "error"):
        if c[k]:
            print(f"  {k:<18}{c[k]:>5}  {c[k]/len(rows):>7.1%}")

    answered = [r for r in rows if r["_outcome"] != "abstained"
                and r["_outcome"] != "error"]
    if answered:
        hall = sum(r["_outcome"] == "hallucination" for r in answered)
        print(f"\n  hallucination rate among answered: {hall/len(answered):.1%}")

    # ---- accuracy by similarity bin (derives the threshold) ---------------
    print("\naccuracy by top-similarity bin")
    print("-" * 49)
    bins = [(0, .35), (.35, .45), (.45, .55), (.55, .65), (.65, 1.01)]
    for lo, hi in bins:
        sub = [r for r in rows
               if r["top_similarity"] is not None and lo <= r["top_similarity"] < hi]
        if not sub:
            continue
        a = sum(r["_correct"] for r in sub) / len(sub)
        vb = [r for r in sub if r["retrieval_hit"] is not None]
        h = (sum(r["retrieval_hit"] for r in vb) / len(vb)) if vb else float("nan")
        print(f"  {lo:.2f}-{hi:.2f}  n={len(sub):>4}  acc={a:.3f}  retr.hit={h:.3f}")

    # ---- coverage / selective accuracy ------------------------------------
    print("\nabstention threshold: coverage vs selective accuracy")
    print("-" * 49)
    print(f"  {'thresh':>7}{'coverage':>11}{'sel.acc':>10}{'halluc':>9}")
    for t in (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        kept = [r for r in rows if (r["top_similarity"] or 0) >= t]
        if not kept:
            continue
        cov = len(kept) / len(rows)
        sel = sum(r["_correct"] for r in kept) / len(kept)
        hl = sum(r["_outcome"] == "hallucination" for r in kept) / len(kept)
        print(f"  {t:>7.2f}{cov:>11.1%}{sel:>10.3f}{hl:>9.1%}")

    # ---- per-field --------------------------------------------------------
    print("\nby field")
    print("-" * 49)
    byf = defaultdict(list)
    for r in rows:
        byf[r["field"]].append(r["_correct"])
    for f, vals in sorted(byf.items(), key=lambda x: -len(x[1])):
        print(f"  {f:<14}n={len(vals):>4}  acc={sum(vals)/len(vals):.3f}")

    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    if lat:
        lat.sort()
        print(f"\nlatency: median {lat[len(lat)//2]:.0f} ms, "
              f"p95 {lat[int(len(lat)*0.95)]:.0f} ms")
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"cost: USD {cost:.4f} for {len(rows)} questions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for p in a.runs:
        score(p)
