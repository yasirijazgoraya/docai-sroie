"""
ARM (commercial LLM) -- Claude via AWS Bedrock, all datasets.

Sends each test image to Claude (Bedrock Messages API) with the same prompt
used for the open Qwen model (receipt JSON for SROIE/CORD, key-value JSON for
FUNSD), parses into the common ExtractionRecord, captures latency, and writes
one JSONL to outputs/. Scoring is the separate 05_score pass.

    cd docvlm-rq1
    python -m scripts.09_claude_all --dataset cord  --split test --limit 5   # smoke
    python -m scripts.09_claude_all --dataset cord  --split test
    python -m scripts.09_claude_all --dataset funsd --split test

Model id must be the Bedrock inference profile, e.g.
    us.anthropic.claude-haiku-4-5-20251001-v1:0
Credentials come from ~/.aws/credentials (never committed).
"""
import argparse
import base64
import json
import time

import boto3

from src.config import OUTPUTS
from src import datasets as ds
from src.parsing import PROMPT, PROMPT_FUNSD, parse_json
from src.schema import ExtractionRecord, LineItem

REGION = "us-east-1"
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _as_str(x):
    if x is None:
        return None
    return str(x).strip() or None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _media_type(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".png"):
        return "image/png"
    return "image/jpeg"


def claude_call(client, model_id, img_bytes, media_type, prompt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": base64.b64encode(img_bytes).decode("utf-8")}},
                {"type": "text", "text": prompt},
            ]}
        ],
    }
    t0 = time.time()
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    latency = (time.time() - t0) * 1000
    payload = json.loads(resp["body"].read())
    text = "".join(b.get("text", "") for b in payload.get("content", [])
                   if b.get("type") == "text")
    return text, latency


def to_record_receipt(doc_id, dataset, raw_text, latency_ms, engine):
    j = parse_json(raw_text)
    items = [LineItem(description=_as_str(i.get("description")),
                      quantity=_num(i.get("quantity")), price=_num(i.get("price")))
             for i in (j.get("line_items") or []) if isinstance(i, dict)]
    return ExtractionRecord(
        doc_id=doc_id, dataset=dataset, doc_type="receipt", source_engine=engine,
        vendor=_as_str(j.get("vendor")), date=_as_str(j.get("date")),
        total=_num(j.get("total")), tax=_num(j.get("tax")),
        currency=_as_str(j.get("currency")), address=_as_str(j.get("address")),
        line_items=items, raw_output=raw_text, latency_ms=latency_ms, cost_usd=0.0,
    )


def to_record_form(doc_id, raw_text, latency_ms, engine):
    j = parse_json(raw_text)
    kv = {}
    if isinstance(j, dict):
        for k, v in j.items():
            if v is None:
                continue
            kv[str(k).strip()] = str(v).strip()
    return ExtractionRecord(
        doc_id=doc_id, dataset="funsd", doc_type="form", source_engine=engine,
        kv_pairs=kv, raw_output=raw_text, latency_ms=latency_ms, cost_usd=0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sroie", "cord", "funsd"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="claude")
    args = ap.parse_args()

    prompt = PROMPT_FUNSD if args.dataset == "funsd" else PROMPT
    client = boto3.client("bedrock-runtime", region_name=REGION)
    samples = ds.load(args.dataset, args.split)
    if args.limit:
        samples = samples[:args.limit]
    print(f"{len(samples)} {args.dataset} samples ({args.split})  model={args.model}")

    out_path = OUTPUTS / f"{args.tag}__{args.dataset}__{args.split}.jsonl"
    n_ok, n_err = 0, 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(samples, 1):
            try:
                with open(s.image_path, "rb") as imgf:
                    img_bytes = imgf.read()
                raw, latency = claude_call(client, args.model, img_bytes,
                                           _media_type(s.image_path), prompt)
                if args.dataset == "funsd":
                    rec = to_record_form(s.doc_id, raw, latency, args.tag)
                else:
                    rec = to_record_receipt(s.doc_id, args.dataset, raw, latency, args.tag)
                n_ok += 1
            except Exception as e:
                rec = ExtractionRecord(
                    doc_id=s.doc_id, dataset=args.dataset,
                    doc_type="form" if args.dataset == "funsd" else "receipt",
                    source_engine=args.tag, status="error",
                    raw_output=f"ERROR: {e}", cost_usd=0.0)
                n_err += 1
            fh.write(rec.model_dump_json() + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(samples)}  (ok={n_ok} err={n_err})")

    print(f"\nWrote {out_path}  (ok={n_ok}, errors={n_err})")


if __name__ == "__main__":
    main()
