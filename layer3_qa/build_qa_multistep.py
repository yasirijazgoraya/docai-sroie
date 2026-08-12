"""
Build the MULTI-STEP question set.

The existing 287 questions are single-intent: one lookup or one aggregate. The
router answers them at 0.82 and an agent would only reproduce that more slowly.
The questions worth adding are the ones the router structurally cannot answer --
those needing two or more queries and a comparison or ranking between results.

Four classes, all with gold answers computed by SQL over the stored records:

  compare    two vendors, which is larger        -> 2 queries + comparison
  rank       highest/lowest across all vendors   -> group + order
  extreme    biggest purchase in a year          -> max over a filtered set
  chain      find a receipt, then use a field    -> lookup, then second lookup

Gold comes from SQL over the SAME store the system queries, so a wrong answer
means the reasoning failed, not that the extraction was wrong. That isolates the
capability being tested. (The 287-question benchmark scores against dataset gold
and remains the accuracy claim; these measure compositional reasoning.)

    python layer3/build_qa_multistep.py
    python layer3/build_qa_multistep.py --n 40

Output: layer3/qa/sroie_qa_multistep.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "layer3" / "qa" / "sroie_qa_multistep.jsonl"
DATASET, ARM = "sroie", "zeroshot"


def short(v):
    """A vendor name as a person would say it: drop the corporate suffix."""
    t = v
    for suf in (" SDN. BHD.", " SDN BHD", " SDN.BHD", " S/B", " SDN. BHD",
                " (M) BHD", " BHD"):
        if t.upper().endswith(suf.upper()):
            t = t[: -len(suf)]
            break
    return t.strip().rstrip(",.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    qa = []

    def add(kind, question, answer, steps, note=""):
        qa.append({"qid": f"{kind[:2]}{len(qa):03d}", "kind": kind,
                   "question": question, "gold_answer": str(answer),
                   "n_steps": steps, "note": note})

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        # vendors with enough receipts to be worth asking about
        cur.execute("""SELECT c.canon_name, count(*), round(sum(d.total),2),
                              round(max(d.total),2)
                       FROM documents d
                       JOIN (SELECT DISTINCT vendor_id, canon_name
                             FROM vendor_canon WHERE dataset=%s AND arm=%s) c
                         ON c.vendor_id = d.vendor_id
                       WHERE d.dataset=%s AND d.arm=%s AND d.total IS NOT NULL
                       GROUP BY c.vendor_id, c.canon_name HAVING count(*) >= 3
                       ORDER BY sum(d.total) DESC""",
                    (DATASET, ARM, DATASET, ARM))
        vendors = cur.fetchall()
        print(f"{len(vendors)} vendors with 3+ receipts")

        # ---- compare: two vendors, which is larger -----------------------
        pairs = []
        for i in range(len(vendors)):
            for j in range(i + 1, len(vendors)):
                a_, b_ = vendors[i], vendors[j]
                # skip near-ties: the question would be unfair, not hard
                if abs(float(a_[2]) - float(b_[2])) / max(float(a_[2]), 1) > 0.15:
                    pairs.append((a_, b_))
        rng.shuffle(pairs)

        for a_, b_ in pairs[: a.n // 3]:
            hi = a_ if float(a_[2]) > float(b_[2]) else b_
            add("compare",
                f"Did I spend more at {short(a_[0])} or {short(b_[0])}?",
                short(hi[0]), 2,
                f"{short(a_[0])} {a_[2]} vs {short(b_[0])} {b_[2]}")

        for a_, b_ in pairs[a.n // 3: a.n // 3 + a.n // 6]:
            hi = a_ if a_[1] > b_[1] else b_
            if a_[1] == b_[1]:
                continue
            add("compare",
                f"Do I have more receipts from {short(a_[0])} or "
                f"{short(b_[0])}?",
                short(hi[0]), 2, f"{a_[1]} vs {b_[1]} receipts")

        # ---- rank: across all vendors ------------------------------------
        top = vendors[0]
        add("rank", "Which vendor did I spend the most with?", short(top[0]), 2,
            f"{top[2]} across {top[1]} receipts")
        add("rank", "Which vendor do I have the most receipts from?",
            short(max(vendors, key=lambda r: r[1])[0]), 2, "")
        cur.execute("""SELECT c.canon_name, round(sum(d.total),2)
                       FROM documents d
                       JOIN (SELECT DISTINCT vendor_id, canon_name
                             FROM vendor_canon WHERE dataset=%s AND arm=%s) c
                         ON c.vendor_id = d.vendor_id
                       WHERE d.dataset=%s AND d.arm=%s AND d.total IS NOT NULL
                       GROUP BY c.vendor_id, c.canon_name
                       ORDER BY sum(d.total) DESC LIMIT 3""",
                    (DATASET, ARM, DATASET, ARM))
        top3 = cur.fetchall()
        add("rank", "What are my top three vendors by total spend?",
            "; ".join(short(v) for v, _ in top3), 2, "")

        # ---- extreme: max/min over a filtered set ------------------------
        for year in (2017, 2018):
            cur.execute("""SELECT vendor, round(total,2), to_char(doc_date,'DD/MM/YYYY')
                           FROM documents WHERE dataset=%s AND arm=%s
                             AND extract(year from doc_date)=%s
                             AND total IS NOT NULL
                           ORDER BY total DESC LIMIT 1""", (DATASET, ARM, year))
            r = cur.fetchone()
            if r:
                add("extreme", f"What was my biggest single purchase in {year}?",
                    f"{r[1]:.2f}", 2, f"{short(r[0])} on {r[2]}")
                add("extreme",
                    f"Which vendor did I make my biggest {year} purchase from?",
                    short(r[0]), 2, f"{r[1]:.2f} on {r[2]}")

        cur.execute("""SELECT round(avg(total),2) FROM documents
                       WHERE dataset=%s AND arm=%s AND total IS NOT NULL""",
                    (DATASET, ARM))
        add("extreme", "What is my average receipt value?",
            f"{float(cur.fetchone()[0]):.2f}", 2, "")

        # ---- chain: resolve a receipt, then read a second field ----------
        cur.execute("""SELECT vendor, to_char(doc_date,'DD/MM/YYYY'),
                              round(total,2), address
                       FROM documents WHERE dataset=%s AND arm=%s
                         AND address IS NOT NULL AND total IS NOT NULL
                         AND vendor IS NOT NULL
                       ORDER BY total DESC LIMIT 40""", (DATASET, ARM))
        rows = cur.fetchall()
        rng.shuffle(rows)
        for v, d, t, addr in rows[: a.n // 6]:
            add("chain",
                f"What is the address of the vendor I spent {t:.2f} with "
                f"on {d}?", addr, 2, f"{short(v)}")

        for v, d, t, addr in rows[a.n // 6: a.n // 6 + 4]:
            add("chain",
                f"On what date did I make my purchase of {t:.2f} from "
                f"{short(v)}, and what was that shop's address?",
                f"{d}; {addr}", 2, "two fields, one receipt")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for q in qa:
            fh.write(json.dumps(q) + "\n")

    from collections import Counter
    c = Counter(q["kind"] for q in qa)
    print(f"\nwrote {OUT}: {len(qa)} questions")
    for k, n in c.most_common():
        print(f"  {k:<10} {n}")
    print("\nsamples:")
    for k in c:
        s = next(q for q in qa if q["kind"] == k)
        print(f"  [{k}] {s['question']}")
        print(f"       -> {s['gold_answer']}")


if __name__ == "__main__":
    main()
