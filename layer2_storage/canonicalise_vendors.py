"""
Layer 2 -- canonical vendor resolution.

The aggregate arm sits at 0.667 and every one of its failures is vendor-name
resolution, not arithmetic. Two symmetric errors:

  under-count   OCR variants of one vendor group separately
                GARDENIA BAKERIES (KL) vs GARDENIA BAKERIES (KI)
                MR. D.I.Y. (M) vs MR. D.I.Y.: (M)
  over-count    query-time IDF matching merges distinct companies that share
                generic tokens
                SUPER SEVEN CASH & CARRY + SEGI CASH & CARRY + FIVE STAR CASH & CARRY
                LIM SENG THO HARDWARE + KOH SENG HARDWARE

Query-time fuzzy matching cannot fix both at once: loosening it merges more
distinct vendors, tightening it splits more variants. The fix is to resolve
vendors ONCE at load time into a canonical table, then group by a stable id.

Clustering rule -- two vendor strings are the same company when:
  * their rare tokens agree (IDF-weighted overlap above threshold), AND
  * neither carries a distinctive token the other lacks

The second condition is what separates the two cases. GARDENIA (KL)/(KI) differ
only by a character-level OCR error inside a shared token set, so they merge.
MR. D.I.Y. (M)/(KUCHAI) each carry a distinctive branch token, so they stay
apart -- which matches the gold annotations, where those are counted separately.

    python layer2/canonicalise_vendors.py --dry-run    # inspect clusters first
    python layer2/canonicalise_vendors.py

Adds documents.vendor_id and a vendor_canon table. Nothing is overwritten:
documents.vendor keeps the raw extracted string for auditing.
"""
import argparse
import math
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
DATASET, ARM = "sroie", "zeroshot"

# Corporate suffixes and generic trade words. These carry no identity: every
# second Malaysian receipt says SDN BHD, and "cash & carry" is a shop type, not
# a company.
SUFFIX = {"sdn", "bhd", "s", "b", "co", "ltd", "berhad", "enterprise",
          "enterprises", "trading", "sdnbhd"}
GENERIC = {"cash", "carry", "hardware", "restaurant", "restoran", "stationery",
           "books", "book", "shop", "store", "mart", "supermarket", "trading",
           "services", "service", "holdings", "group", "the", "and", "of"}

DDL = """
CREATE TABLE IF NOT EXISTS vendor_canon (
    vendor_id    INT  NOT NULL,
    dataset      TEXT NOT NULL,
    arm          TEXT NOT NULL,
    raw_vendor   TEXT NOT NULL,
    canon_name   TEXT NOT NULL,
    n_receipts   INT,
    PRIMARY KEY (dataset, arm, raw_vendor)
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vendor_id INT;
CREATE INDEX IF NOT EXISTS idx_documents_vendor_id ON documents (vendor_id);
"""


def tokens(s):
    t = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()
    return [x for x in t if x and x not in SUFFIX]


def id_tokens(s):
    """Identity-bearing tokens: drop generic trade words as well as suffixes."""
    return {t for t in tokens(s) if t not in GENERIC}


def char_sim(a, b):
    return SequenceMatcher(None, re.sub(r"[^a-z0-9]", "", a.lower()),
                           re.sub(r"[^a-z0-9]", "", b.lower())).ratio()


class Canonicaliser:
    def __init__(self, vendors):
        self.vendors = vendors                 # list of (raw, n_receipts)
        self.df = defaultdict(int)
        for v, _ in vendors:
            for t in set(tokens(v)):
                self.df[t] += 1
        self.n = max(len(vendors), 1)

    def idf(self, t):
        return math.log(self.n / (1 + self.df[t])) + 1.0

    def same_company(self, a, b):
        ta, tb = id_tokens(a), id_tokens(b)
        if not ta or not tb:
            return False

        # A distinctive token present in one and absent from the other means a
        # different branch or a different company -- MR. D.I.Y. (KUCHAI) is not
        # MR. D.I.Y. (M). Tokens are compared fuzzily so a single OCR character
        # error does not count as "distinctive".
        def covered(x, other):
            # Short tokens cannot survive an 0.8 ratio after one OCR character
            # error: "kl" vs "ki" scores 0.5. Allow a single substitution on
            # tokens of 2-3 characters, where that is the whole difference.
            for y in other:
                if char_sim(x, y) >= 0.8:
                    return True
                if len(x) <= 3 and len(x) == len(y) and \
                        sum(1 for c, d in zip(x, y) if c != d) <= 1:
                    return True
            return False

        only_a = [t for t in ta if not covered(t, tb)]
        only_b = [t for t in tb if not covered(t, ta)]
        if only_a or only_b:
            return False

        shared = sum(self.idf(t) for t in ta if covered(t, tb))
        total = max(sum(self.idf(t) for t in ta), sum(self.idf(t) for t in tb))
        if total <= 0 or shared / total < 0.75:
            return False

        # Final guard: the full strings must still look alike, which catches
        # pathological token-set coincidences.
        return char_sim(a, b) >= 0.7

    def cluster(self):
        """Union-find over pairwise same_company."""
        parent = list(range(len(self.vendors)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        for i in range(len(self.vendors)):
            for j in range(i + 1, len(self.vendors)):
                if self.same_company(self.vendors[i][0], self.vendors[j][0]):
                    union(i, j)

        groups = defaultdict(list)
        for i, (raw, n) in enumerate(self.vendors):
            groups[find(i)].append((raw, n))
        return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print clusters, write nothing")
    a = ap.parse_args()

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT vendor, count(*) FROM documents "
                    "WHERE dataset=%s AND arm=%s AND vendor IS NOT NULL "
                    "GROUP BY vendor ORDER BY count(*) DESC",
                    (DATASET, ARM))
        vendors = [(r[0], r[1]) for r in cur.fetchall()]
        print(f"{len(vendors)} distinct vendor strings")

        groups = Canonicaliser(vendors).cluster()
        merged = [g for g in groups if len(g) > 1]
        print(f"{len(groups)} canonical vendors "
              f"({len(merged)} clusters merge 2+ strings)\n")

        for g in sorted(merged, key=lambda g: -sum(n for _, n in g)):
            canon = max(g, key=lambda x: x[1])[0]
            print(f"  {sum(n for _, n in g):>3} receipts  →  {canon}")
            for raw, n in sorted(g, key=lambda x: -x[1]):
                mark = " " if raw == canon else "·"
                print(f"      {mark} {n:>3}  {raw}")

        if a.dry_run:
            print("\ndry run: nothing written")
            return

        cur.execute(DDL)
        cur.execute("DELETE FROM vendor_canon WHERE dataset=%s AND arm=%s",
                    (DATASET, ARM))

        for vid, g in enumerate(sorted(groups,
                                       key=lambda g: -sum(n for _, n in g)), 1):
            canon = max(g, key=lambda x: x[1])[0]
            total = sum(n for _, n in g)
            for raw, n in g:
                cur.execute(
                    """INSERT INTO vendor_canon
                         (vendor_id, dataset, arm, raw_vendor, canon_name, n_receipts)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (vid, DATASET, ARM, raw, canon, total))
                cur.execute("UPDATE documents SET vendor_id=%s WHERE dataset=%s "
                            "AND arm=%s AND vendor=%s",
                            (vid, DATASET, ARM, raw))
        conn.commit()

        cur.execute("SELECT count(*) FROM documents WHERE dataset=%s AND arm=%s "
                    "AND vendor_id IS NULL", (DATASET, ARM))
        print(f"\nwrote vendor_canon; {cur.fetchone()[0]} documents unassigned")

        print("\ntop canonical vendors by spend:")
        cur.execute("""SELECT c.canon_name, count(*), round(sum(d.total),2)
                       FROM documents d JOIN vendor_canon c
                         ON c.vendor_id = d.vendor_id AND c.dataset = d.dataset
                            AND c.arm = d.arm AND c.raw_vendor = d.vendor
                       WHERE d.dataset=%s AND d.arm=%s
                       GROUP BY c.canon_name ORDER BY 3 DESC LIMIT 8""",
                    (DATASET, ARM))
        for name, n, spend in cur.fetchall():
            print(f"  {str(spend):>9}  {n:>3}  {name}")


if __name__ == "__main__":
    main()
