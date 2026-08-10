"""Shared output-parsing helpers for Qwen inference (zero-shot and fine-tuned)."""
import json, re
from src.schema import ExtractionRecord, LineItem

PROMPT = (
    "You are extracting key information from a receipt image. "
    "Return ONLY a JSON object, no prose, with exactly these keys: "
    '"vendor", "date", "total" (number), "tax" (number or null), "currency", '
    '"address", and "line_items" (a list of {"description","quantity","price"}). '
    "Use null for anything not present."
)


def _as_str(x):
    if x is None:
        return None
    return str(x).strip() or None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def to_record(doc_id, dataset, raw_text, latency_ms, engine) -> ExtractionRecord:
    j = parse_json(raw_text)
    items = [LineItem(description=_as_str(i.get("description")),
                      quantity=_num(i.get("quantity")), price=_num(i.get("price")))
             for i in (j.get("line_items") or []) if isinstance(i, dict)]
    return ExtractionRecord(
        doc_id=doc_id, dataset=dataset, doc_type="receipt", source_engine=engine,
        vendor=_as_str(j.get("vendor")), date=_as_str(j.get("date")), total=_num(j.get("total")),
        tax=_num(j.get("tax")), currency=_as_str(j.get("currency")), address=_as_str(j.get("address")),
        line_items=items, raw_output=raw_text, latency_ms=latency_ms, cost_usd=0.0,
    )


PROMPT_FUNSD = (
    "You are extracting key-value pairs from a form image. "
    "A key is a field label and the value is the text filled in for it. "
    "Return ONLY a JSON object mapping each key string to its value string, "
    "no prose. Example: {\"TO:\": \"John Smith\", \"DATE:\": \"April 1 1990\"}. "
    "Include every label-value pair you can read. Use an empty object {} if none."
)


def to_record_funsd(doc_id, raw_text, latency_ms, engine) -> ExtractionRecord:
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
