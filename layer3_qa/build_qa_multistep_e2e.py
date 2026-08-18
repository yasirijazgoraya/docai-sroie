"""
Regenerate multi-step gold from the SROIE ANNOTATIONS instead of the store.

The 36 multi-step questions currently take their gold from SQL over `documents`
-- our own extractions. The agent answers by querying the same table, so both
sides share a source: if Layer 1 misread a total, gold and answer are wrong
together and the agent scores correct. That measures compositional REASONING,
which is a legitimate ablation, but it is not end-to-end correctness.

This script recomputes the same questions' answers from the human-labelled
SROIE entity files. Nothing about the system changes -- the agent still answers
from the store. Only the answer key moves.

    reasoning accuracy   gold from store        (existing, sroie_qa_multistep.jsonl)
    end-to-end accuracy  gold from annotations  (this file, ..._e2e.jsonl)

The difference between the two scores IS the extraction penalty on
compositional queries -- a quantity not otherwise measured.

Worked example, real numbers:

    AEON, store       723.10 over  8 receipts
    AEON, annotations 870.55 over  9 receipts   <- one receipt lost to OCR error

For "did I spend more at AEON or Gardenia (1104.55)?" both keys give the same
answer, because the error does not flip the ordering. For "how much did I spend
at AEON?" they disagree outright.

Vendor grouping uses the SAME canonicaliser as the store, applied to the gold
company strings -- SROIE's own annotations contain OCR-level vendor noise
(KEDA PAPAN YEW CHJAN), so raw string equality is not coherent on either side.

    python layer3/build_qa_multistep_e2e.py
    python layer3/score_multistep.py layer3/runs/agent_multistep__k5.jsonl \
        --qa layer3/qa/sroie_qa_multistep_e2e.jsonl

Writes layer3/qa/sroie_qa_multistep_e2e.jsonl -- same qids, same questions,
recomputed gold_answer.
"""
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "layer3" / "qa" / "sroie_qa_multistep.jsonl"
OUT = ROOT / "layer3" / "qa" / "sroie_qa_multistep_e2e.jsonl"
ENTITIES = Path("/mnt/yasir_drive/E_DATA/data/SROIE2019/test/entities")

spec = importlib.util.spec_from_file_location(
    "canon", ROOT / "layer2" / "canonicalise_vendors.py")
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)

SUFFIXES = (" sdn. bhd.", " sdn bhd", " sdn.bhd", " s/b", " sdn. bhd",
            " (m) bhd", " bhd", " sdn", " co.", " ltd")


def parse_money(s):
    m = re.search(r"-?[\d,]*\.?\d+", str(s or "").replace(" ", ""))
    if not m:
        return None
    try:
        return round(float(m.group().replace(",", "")), 2)
    except ValueError:
        return None


def parse_date(s):
    from datetime import datetime
    if not s:
        return None
    t = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*$", "", str(s).strip())
    for f in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y",
              "%d %b %Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(t, f)
            return d.replace(year=d.year + 2000) if d.year < 100 else d
        except ValueError:
            continue
    return None


def short(v):
    t = v
    for suf in (" SDN. BHD.", " SDN BHD", " SDN.BHD", " S/B", " SDN. BHD",
                " (M) BHD", " BHD"):
        if t.upper().endswith(suf.upper()):
            t = t[: -len(suf)]
            break
    return t.strip().rstrip(",.")


def norm_name(s):
    t = " ".join(str(s or "").lower().split())
    for suf in SUFFIXES:
        if t.endswith(suf):
            t = t[: -len(suf)]
            break
    return re.sub(r"[^a-z0-9]", "", t)


# --------------------------------------------------------------- gold load

def load_gold():
    recs = []
    for f in sorted(ENTITIES.glob("*.txt")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        recs.append({
            "doc_id": f.stem,
            "company": (d.get("company") or "").strip(),
            "total": parse_money(d.get("total")),
            "date": parse_date(d.get("date")),
            "address": (d.get("address") or "").strip(),
        })
    return recs


def cluster_vendors(recs):
    counts = defaultdict(int)
    for r in recs:
        if r["company"]:
            counts[r["company"]] += 1
    vendors = sorted(counts.items(), key=lambda x: -x[1])
    groups = canon.Canonicaliser(vendors).cluster()

    cluster_id, canon_name = {}, {}
    for i, g in enumerate(groups):
        name = max(g, key=lambda x: x[1])[0]
        for raw, _ in g:
            cluster_id[raw] = i
            canon_name[i] = name
    return cluster_id, canon_name


# ------------------------------------------------------------- resolution

def resolve(name, cluster_id, canon_name):
    """Map a vendor name as it appears in a question to a gold cluster id."""
    want = norm_name(name)
    if not want:
        return None
    best, best_len = None, 0
    for raw, cid in cluster_id.items():
        have = norm_name(raw)
        if not have:
            continue
        if want == have or want in have or have in want:
            if len(have) > best_len:
                best, best_len = cid, len(have)
    return best


def main():
    if not ENTITIES.exists():
        raise SystemExit(f"gold entities not found: {ENTITIES}")

    recs = load_gold()
    cluster_id, canon_name = cluster_vendors(recs)
    print(f"{len(recs)} gold receipts, {len(canon_name)} canonical vendors")

    by_cluster = defaultdict(list)
    for r in recs:
        cid = cluster_id.get(r["company"])
        if cid is not None and r["total"] is not None:
            by_cluster[cid].append(r)

    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    out, changed, unresolved = [], 0, 0

    for q in rows:
        text, kind = q["question"], q["kind"]
        old = q["gold_answer"]
        new = None

        # ---- compare: "Did I spend more at X or Y?" / "more receipts from" --
        m = re.match(r"^Did I spend more at (.+?) or (.+?)\?$", text)
        if m:
            a, b = (resolve(m.group(1), cluster_id, canon_name),
                    resolve(m.group(2), cluster_id, canon_name))
            if a is not None and b is not None:
                sa = sum(r["total"] for r in by_cluster[a])
                sb = sum(r["total"] for r in by_cluster[b])
                new = short(canon_name[a] if sa > sb else canon_name[b])
                q["note"] = f"gold: {short(canon_name[a])} {sa:.2f} vs " \
                            f"{short(canon_name[b])} {sb:.2f}"

        m = re.match(r"^Do I have more receipts from (.+?) or (.+?)\?$", text)
        if m and new is None:
            a, b = (resolve(m.group(1), cluster_id, canon_name),
                    resolve(m.group(2), cluster_id, canon_name))
            if a is not None and b is not None:
                na, nb = len(by_cluster[a]), len(by_cluster[b])
                new = short(canon_name[a] if na > nb else canon_name[b])
                q["note"] = f"gold: {na} vs {nb} receipts"

        # ---- rank ----------------------------------------------------------
        if text.startswith("Which vendor did I spend the most with"):
            cid = max(by_cluster, key=lambda c: sum(r["total"]
                                                    for r in by_cluster[c]))
            new = short(canon_name[cid])
        elif text.startswith("Which vendor do I have the most receipts from"):
            cid = max(by_cluster, key=lambda c: len(by_cluster[c]))
            new = short(canon_name[cid])
        elif text.startswith("What are my top three vendors"):
            top = sorted(by_cluster,
                         key=lambda c: -sum(r["total"] for r in by_cluster[c]))[:3]
            new = "; ".join(short(canon_name[c]) for c in top)

        # ---- extreme -------------------------------------------------------
        m = re.match(r"^What was my biggest single purchase in (\d{4})\?$", text)
        if m:
            yr = int(m.group(1))
            cand = [r for r in recs
                    if r["date"] and r["date"].year == yr and r["total"]]
            if cand:
                new = f"{max(r['total'] for r in cand):.2f}"

        m = re.match(r"^Which vendor did I make my biggest (\d{4}) purchase from\?$",
                     text)
        if m:
            yr = int(m.group(1))
            cand = [r for r in recs
                    if r["date"] and r["date"].year == yr and r["total"]]
            if cand:
                new = short(max(cand, key=lambda r: r["total"])["company"])

        if text.startswith("What is my average receipt value"):
            tot = [r["total"] for r in recs if r["total"] is not None]
            new = f"{sum(tot)/len(tot):.2f}"

        # ---- chain: keyed on an amount, so the amount itself may differ -----
        # These reference a receipt BY its extracted total ("the vendor I spent
        # 283.55 with"). If extraction was wrong, no gold receipt carries that
        # amount and the question is unanswerable against annotations. Left
        # unchanged and flagged rather than silently dropped.
        if kind == "chain" and new is None:
            m = re.search(r"([\d]+\.[\d]{2})", text)
            amt = parse_money(m.group(1)) if m else None
            hit = [r for r in recs
                   if amt is not None and r["total"] is not None
                   and abs(r["total"] - amt) <= 0.01]
            if len(hit) == 1:
                r = hit[0]
                if "address of the vendor" in text:
                    new = r["address"]
                elif "On what date" in text:
                    d = r["date"].strftime("%d/%m/%Y") if r["date"] else "?"
                    new = f"{d}; {r['address']}"
            else:
                q["note"] = (f"amount {amt} matches {len(hit)} gold receipts "
                             f"-- not resolvable against annotations")

        if new is None:
            unresolved += 1
            q["gold_source"] = "store (unchanged)"
        else:
            if str(new) != str(old):
                changed += 1
                print(f"  {q['qid']:<8} {kind:<8} {str(old)[:34]:<34} -> "
                      f"{str(new)[:34]}")
            q["gold_answer"] = str(new)
            q["gold_source"] = "annotations"
        out.append(q)

    with OUT.open("w") as fh:
        for q in out:
            fh.write(json.dumps(q) + "\n")

    print(f"\nwrote {OUT.name}")
    print(f"{len(out)} questions | {changed} gold answers changed | "
          f"{unresolved} could not be recomputed from annotations")
    if unresolved:
        print("Unresolved questions keep store-derived gold; they are marked "
              "gold_source='store (unchanged)' and should be excluded from the "
              "end-to-end figure or reported separately.")


if __name__ == "__main__":
    main()
