"""
Dataset loaders. Each returns Sample(doc_id, image_path, doc_type, dataset, gold).
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PATHS
from .schema import GoldRecord, LineItem


@dataclass
class Sample:
    doc_id: str
    image_path: Path
    doc_type: str
    dataset: str
    gold: GoldRecord


def _to_float(x) -> Optional[float]:
    """Simple parser for clean numeric strings (SROIE totals)."""
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").replace("RM", "").strip())
    except ValueError:
        return None


def _amount(text: str) -> Optional[float]:
    """Locale-aware amount parser for CORD (Indonesian: '.'/',' = thousands sep).
    Strips text prefixes like 'TOTAL', grabs the trailing number, removes
    thousands separators. 'TOTAL 60.000' -> 60000, 'TOTAL 28,000' -> 28000."""
    if not text:
        return None
    nums = re.findall(r"\d[\d.,]*\d|\d", text)
    if not nums:
        return None
    digits = re.sub(r"[.,]", "", nums[-1])
    try:
        return float(digits)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# SROIE 2019 : SROIE2019/<split>/{img,box,entities}                             #
# --------------------------------------------------------------------------- #
def load_sroie(split: str = "test") -> list[Sample]:
    root = PATHS["sroie"]
    split_dir = root / split
    img_dir = split_dir / "img"
    ent_dir = split_dir / "entities"
    if not img_dir.exists():
        img_dir = split_dir / "images"
    samples: list[Sample] = []
    for ent_file in sorted(ent_dir.glob("*.txt")):
        doc_id = ent_file.stem
        img = next((img_dir / f"{doc_id}{ext}" for ext in (".jpg", ".png", ".jpeg")
                    if (img_dir / f"{doc_id}{ext}").exists()), None)
        if img is None:
            continue
        try:
            g = json.loads(ent_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        gold = GoldRecord(
            doc_id=doc_id, dataset="sroie", doc_type="receipt",
            vendor=g.get("company"), date=g.get("date"),
            address=g.get("address"), total=_to_float(g.get("total")),
        )
        samples.append(Sample(doc_id, img, "receipt", "sroie", gold))
    return samples


# --------------------------------------------------------------------------- #
# CORD : CORD/CORD/{train,dev,test}/{image,json}                                #
#   No vendor/date (correct -> None). Total + line items, locale-aware amounts. #
# --------------------------------------------------------------------------- #
TOTAL_PRIORITY = ["total.total_price", "total.cashprice",
                  "total.creditcardprice", "sub_total.subtotal_price"]


def _cord_split_dir(split: str) -> Path:
    root = PATHS["cord"]
    for c in (root / split, root / "CORD" / split):
        if (c / "json").exists() and (c / "image").exists():
            return c
    for c in root.rglob(f"{split}/json"):
        if ".fr-" not in str(c):
            return c.parent
    raise FileNotFoundError(f"Could not locate CORD '{split}' under {root}")


def _line_text(line: dict) -> str:
    return " ".join(w.get("text", "") for w in line.get("words", [])).strip()


def load_cord(split: str = "test") -> list[Sample]:
    split_dir = _cord_split_dir(split)
    img_dir = split_dir / "image"
    json_dir = split_dir / "json"
    samples: list[Sample] = []
    for jf in sorted(json_dir.glob("*.json")):
        doc_id = jf.stem
        img = next((img_dir / f"{doc_id}{ext}" for ext in (".png", ".jpg", ".jpeg")
                    if (img_dir / f"{doc_id}{ext}").exists()), None)
        if img is None:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))

        items, cur, totals = [], {}, {}
        for line in data.get("valid_line", []):
            cat = line.get("category", "")
            txt = _line_text(line)
            if cat == "menu.nm":
                if cur.get("description"):
                    items.append(LineItem(**cur))
                cur = {"description": txt}
            elif cat == "menu.cnt":
                cur["quantity"] = _amount(txt)
            elif cat in ("menu.price", "menu.unitprice"):
                cur["price"] = _amount(txt)
            elif cat in TOTAL_PRIORITY and cat not in totals:
                totals[cat] = _amount(txt)
        if cur.get("description"):
            items.append(LineItem(**cur))

        total = next((totals[c] for c in TOTAL_PRIORITY
                      if totals.get(c) is not None), None)
        gold = GoldRecord(
            doc_id=doc_id, dataset="cord", doc_type="receipt",
            total=total, line_items=items,
        )
        samples.append(Sample(doc_id, img, "receipt", "cord", gold))
    return samples


def load_funsd(split: str = "test") -> list[Sample]:
    """FUNSD forms. Gold key-value pairs are built from linked
    question->answer items and stored in GoldRecord.kv_pairs."""
    root = PATHS["funsd"]
    split_dir = root / ("testing_data" if split == "test" else "training_data")
    ann_dir = split_dir / "annotations"
    img_dir = split_dir / "images"
    samples: list[Sample] = []
    for ann_file in sorted(ann_dir.glob("*.json")):
        doc_id = ann_file.stem
        img = next((img_dir / f"{doc_id}{ext}" for ext in (".png", ".jpg", ".jpeg")
                    if (img_dir / f"{doc_id}{ext}").exists()), None)
        if img is None:
            continue
        try:
            data = json.loads(ann_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        items = data.get("form", [])
        by_id = {it["id"]: it for it in items if "id" in it}
        kv: dict[str, str] = {}
        for it in items:
            if it.get("label") == "question":
                q = (it.get("text") or "").strip()
                if not q:
                    continue
                for link in it.get("linking", []):
                    other = link[1] if link[0] == it.get("id") else link[0]
                    ans = by_id.get(other)
                    if ans and ans.get("label") == "answer":
                        a = (ans.get("text") or "").strip()
                        if a:
                            kv[q] = a
        gold = GoldRecord(doc_id=doc_id, dataset="funsd", doc_type="form", kv_pairs=kv)
        samples.append(Sample(doc_id, img, "form", "funsd", gold))
    return samples


LOADERS = {"sroie": load_sroie, "cord": load_cord, "funsd": load_funsd}


def load(dataset: str, split: str = "test") -> list[Sample]:
    return LOADERS[dataset](split)
