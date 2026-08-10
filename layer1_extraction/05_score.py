"""Scoring pass: zero-shot vs fine-tuned, per dataset and field."""
import argparse, json, re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from src.config import OUTPUTS
from src import datasets as ds

CORD_TOTAL_TOL = 0.01
ENGINES = {"qwen_zeroshot": "zeroshot", "qwen_ft": "qwen_ft", "qwen_ft_sep": "qwen_ft_sep", "textract": "textract", "claude": "claude"}
FIELDS = {"sroie": ["vendor", "date", "total", "address"], "cord": ["total"], "funsd": []}

def norm_str(x):
    if x is None: return None
    t = str(x).lower()
    t = re.sub(r"[^a-z0-9]", "", t)       # drop punctuation AND whitespace
    return t or None
def norm_date(x):
    return None if x is None else re.sub(r"[^0-9]", "", str(x))

def _parse_date(s):
    if s is None: return None
    s = str(s).strip()
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*$", "", s).strip()  # drop trailing time
    for f in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d","%d-%m-%Y","%d-%m-%y",
              "%d.%m.%Y","%d.%m.%y","%d %b %Y","%d %b %y","%d %B %Y","%d %B %y",
              "%Y/%m/%d","%b %d %Y","%B %d %Y","%d%m%Y","%Y%m%d"):
        try:
            dt = datetime.strptime(s, f)
            yr = dt.year + 2000 if dt.year < 100 else dt.year
            return (yr, dt.month, dt.day)
        except ValueError:
            continue
    return None
def str_match(pred, gold):
    p, g = norm_str(pred), norm_str(gold)
    if g is None: return None
    if p is None: return False
    return p == g or g in p or p in g
def date_match(pred, gold):
    if gold is None: return None
    if pred is None: return False
    pg, gg = _parse_date(pred), _parse_date(gold)
    if pg and gg:
        return pg == gg
    p, g = norm_date(pred), norm_date(gold)
    return p == g if p else False
def total_match(pred, gold, tol=0.0):
    if gold is None: return None
    if pred is None: return False
    return abs(pred-gold) < 0.005 if tol == 0 else abs(pred-gold) <= tol*max(abs(gold),1.0)
def field_match(field, pred, gold, dataset):
    if field == "date": return date_match(pred, gold)
    if field == "total": return total_match(pred, gold, CORD_TOTAL_TOL if dataset in ("cord", "sroie") else 0.0)
    return str_match(pred, gold)

def line_item_prf(pred_items, gold_items):
    def key(it):
        p = it.get("price")
        return (norm_str(it.get("description")), round(p,2) if isinstance(p,(int,float)) else None)
    gk, pk = [key(g) for g in gold_items], [key(p) for p in pred_items]
    pool, matched = list(gk), 0
    for k in pk:
        if k in pool: matched += 1; pool.remove(k)
    prec = matched/len(pk) if pk else (1.0 if not gk else 0.0)
    rec = matched/len(gk) if gk else 1.0
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def load_jsonl(path):
    out = {}
    if not path.exists(): return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line); out[r["doc_id"]] = r
    return out

def _norm_kv(x):
    if x is None:
        return None
    t = re.sub(r"[^a-z0-9]", "", str(x).lower())
    return t or None


def funsd_kv_prf(pred_kv, gold_kv):
    gold_pairs = [(_norm_kv(k), _norm_kv(v)) for k, v in (gold_kv or {}).items()]
    pred_pairs = [(_norm_kv(k), _norm_kv(v)) for k, v in (pred_kv or {}).items()]
    gold_pairs = [pp for pp in gold_pairs if pp[0] and pp[1]]
    pred_pairs = [pp for pp in pred_pairs if pp[0] and pp[1]]
    pool, tp = list(gold_pairs), 0
    for pp in pred_pairs:
        if pp in pool:
            tp += 1; pool.remove(pp)
    n_pred, n_gold = len(pred_pairs), len(gold_pairs)
    prec = tp / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    rec = tp / n_gold if n_gold else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def score_engine(dataset, tag, split, gold_by_id):
    preds = load_jsonl(OUTPUTS / f"{tag}__{dataset}__{split}.jsonl")
    if not preds: return None
    hits = defaultdict(lambda: [0, 0]); li = []
    for doc_id, gold in gold_by_id.items():
        p = preds.get(doc_id)
        if p is None: continue
        for f in FIELDS[dataset]:
            res = field_match(f, p.get(f), getattr(gold, f), dataset)
            if res is None: continue
            hits[f][1] += 1; hits[f][0] += int(res)
        if dataset == "cord":
            li.append(line_item_prf(p.get("line_items", []),
                                    [x.model_dump() for x in gold.line_items]))
        if dataset == "funsd":
            li.append(funsd_kv_prf(p.get("kv_pairs", {}) or {},
                                   getattr(gold, "kv_pairs", {}) or {}))
    acc = {f: (h[0]/h[1] if h[1] else None) for f, h in hits.items()}
    if dataset == "cord":
        acc["line_items_f1"] = sum(li)/len(li) if li else None
    if dataset == "funsd" and li:
        acc["kv_precision"] = sum(x[0] for x in li)/len(li)
        acc["kv_recall"]    = sum(x[1] for x in li)/len(li)
        acc["kv_f1"]        = sum(x[2] for x in li)/len(li)
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="+", default=["sroie", "cord"])
    args = ap.parse_args()
    for dataset in args.datasets:
        gold = {s.doc_id: s.gold for s in ds.load(dataset, args.split)}
        print("\n" + "="*64)
        print(f"{dataset.upper()}  ({len(gold)} gold docs)")
        print("="*64)
        results = {t: r for t in ENGINES.values()
                   if (r := score_engine(dataset, t, args.split, gold))}
        metrics = list(FIELDS[dataset]) + (["line_items_f1"] if dataset=="cord" else [])
        if dataset == "funsd":
            metrics = ["kv_precision", "kv_recall", "kv_f1"]
        header = f"{'metric':<16}" + "".join(f"{e:>16}" for e in results)
        print(header); print("-"*len(header))
        for m in metrics:
            row = f"{m:<16}"
            for e in results:
                v = results[e].get(m)
                row += f"{(f'{v:.3f}' if v is not None else '-'):>16}"
            print(row)

if __name__ == "__main__":
    main()
