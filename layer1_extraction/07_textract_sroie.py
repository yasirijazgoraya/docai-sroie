"""
ARM (commercial OCR) -- AWS Textract AnalyzeExpense extraction on SROIE.

Runs Textract's AnalyzeExpense API over the SROIE test split, maps Textract's
native expense fields onto the SROIE schema (vendor, date, total, address),
captures latency, and writes one JSONL to outputs/ in the common
ExtractionRecord shape. No scoring here -- that is the separate 05_score pass.

    cd docvlm-rq1
    python -m scripts.07_textract_sroie --split test --limit 5    # smoke test
    python -m scripts.07_textract_sroie --split test              # full run

Cost note: AnalyzeExpense is a paid API (about USD 10 per 1,000 pages). The
SROIE test split is 347 receipts, so a full run costs roughly USD 3-4.
Credentials are read from ~/.aws/credentials (never committed).
"""
import argparse
import json
import time
import re

import boto3

from src.config import OUTPUTS
from src import datasets as ds
from src.schema import ExtractionRecord

REGION = "us-east-1"


def _num(x):
    if x is None:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(x)))
    except (TypeError, ValueError):
        return None


def map_textract_to_sroie(summary_fields) -> dict:
    """
    Map Textract AnalyzeExpense SummaryFields onto the SROIE schema.

    summary_fields: list of dicts from the Textract response. Each has
    Type.Text, optional LabelDetection.Text, and ValueDetection.Text.

    Returns vendor / date / total / address. Chooses the seller (VENDOR_NAME),
    not the bill-to party, and picks a sensible grand total.
    """
    by_type = {}
    for field in summary_fields:
        ftype = field.get("Type", {}).get("Text", "")
        label = field.get("LabelDetection", {}).get("Text", "") or ""
        value = field.get("ValueDetection", {}).get("Text", "") or ""
        by_type.setdefault(ftype, []).append((label, value))

    def first(ftype):
        for _label, value in by_type.get(ftype, []):
            if value and value.strip():
                return value.strip()
        return None

    # vendor: the seller. VENDOR_NAME is the issuer; NAME may be the bill-to party.
    vendor = first("VENDOR_NAME") or first("NAME")

    # date
    date = first("INVOICE_RECEIPT_DATE")

    # total: prefer a field whose label mentions "total", else AMOUNT_PAID,
    # else the largest numeric candidate among TOTAL/SUBTOTAL/AMOUNT_PAID.
    total_cands = []
    for ftype in ("TOTAL", "SUBTOTAL", "AMOUNT_PAID"):
        for label, value in by_type.get(ftype, []):
            n = _num(value)
            if n is not None:
                total_cands.append((("total" in (label or "").lower()), n))
    total = None
    if total_cands:
        total_cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        total = total_cands[0][1]

    # address: ADDRESS_BLOCK is cleanest (no vendor name); fall back as needed.
    address = first("ADDRESS_BLOCK") or first("VENDOR_ADDRESS") or first("ADDRESS")
    if address:
        address = address.replace("\n", " ").strip()

    return {"vendor": vendor, "date": date, "total": total, "address": address}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = boto3.client("textract", region_name=REGION)
    samples = ds.load("sroie", args.split)
    if args.limit:
        samples = samples[:args.limit]
    print(f"{len(samples)} SROIE samples ({args.split})")

    out_path = OUTPUTS / f"textract__sroie__{args.split}.jsonl"
    n_ok, n_err = 0, 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(samples, 1):
            try:
                with open(s.image_path, "rb") as imgf:
                    img_bytes = imgf.read()
                t0 = time.time()
                resp = client.analyze_expense(Document={"Bytes": img_bytes})
                latency = (time.time() - t0) * 1000

                summary = []
                for doc in resp.get("ExpenseDocuments", []):
                    summary.extend(doc.get("SummaryFields", []))
                mapped = map_textract_to_sroie(summary)

                rec = ExtractionRecord(
                    doc_id=s.doc_id, dataset="sroie", doc_type="receipt",
                    source_engine="textract",
                    vendor=mapped["vendor"], date=mapped["date"],
                    total=mapped["total"], address=mapped["address"],
                    raw_output=json.dumps(summary)[:5000],
                    latency_ms=latency, cost_usd=0.01,
                )
                n_ok += 1
            except Exception as e:
                rec = ExtractionRecord(
                    doc_id=s.doc_id, dataset="sroie", doc_type="receipt",
                    source_engine="textract", status="error",
                    raw_output=f"ERROR: {e}", cost_usd=0.0,
                )
                n_err += 1
            fh.write(rec.model_dump_json() + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(samples)}  (ok={n_ok} err={n_err})")

    print(f"\nWrote {out_path}  (ok={n_ok}, errors={n_err})")
    print(f"Approx cost: USD {n_ok * 0.01:.2f}")


if __name__ == "__main__":
    main()
