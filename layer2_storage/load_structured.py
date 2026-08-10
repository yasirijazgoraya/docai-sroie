"""
Layer 2 -- load SROIE structured records into Postgres.

Reads the production extractor's predictions (fine-tuned Qwen2.5-VL, combined
adapter) and writes them to the documents table. Structured fields are queried
directly with SQL; they are never chunked or embedded. The OCR text feed is
handled separately by the chunking script.

    cd /mnt/yasir_drive/E_DATA/ResearchPlan2/docvlm-rq1
    python layer2/load_structured.py

Requires: pip install "psycopg[binary]"
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import psycopg

DSN = os.environ.get("DOCAI_DSN", "postgresql:///docai")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "zeroshot__sroie__test.jsonl"
DATASET, ARM = "sroie", "zeroshot"

_MONEY = re.compile(r"[^\d.,\-]")
_EPOCH = re.compile(r"^-?\d{9,13}(\.\d+)?$")
_DATE_FORMATS = [  # mirrors _parse_date in scripts/05_score.py
    "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y",
    "%d.%m.%Y", "%d.%m.%y", "%d %b %Y", "%d %b %y", "%d %B %Y", "%d %B %y",
    "%Y/%m/%d", "%b %d %Y", "%B %d %Y", "%d%m%Y", "%Y%m%d",
    "%d/%b/%Y", "%d/%b/%y",
]


def norm_money(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = _MONEY.sub("", str(v))
    if not s:
        return None
    if "," in s and "." in s:
        s = (s.replace(".", "").replace(",", ".")
             if s.rindex(",") > s.rindex(".") else s.replace(",", ""))
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) == 2 else s.replace(",", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def norm_date(v):
    """Parse to a date, or None. Handles the 3 epoch-millisecond records the
    fine-tuned arm emits, as well as normal receipt date strings."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if _EPOCH.match(s):
        n = float(s)
        if abs(n) > 1e11:
            n /= 1000.0
        try:
            return datetime.fromtimestamp(n, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*$", "", s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def norm_text(v):
    if v is None:
        return None
    return " ".join(str(v).split()) or None


def main():
    if not SRC.exists():
        raise SystemExit(f"missing: {SRC}")

    n = skipped = epoch = unparsed = 0

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE dataset=%s AND arm=%s",
                    (DATASET, ARM))

        for line in SRC.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            if r.get("status") != "extracted" or not r.get("doc_id"):
                skipped += 1
                continue

            raw_date = r.get("date")
            if raw_date is not None and _EPOCH.match(str(raw_date).strip()):
                epoch += 1
            d = norm_date(raw_date)
            if raw_date is not None and d is None:
                unparsed += 1

            cur.execute(
                """INSERT INTO documents
                     (dataset, arm, doc_id, vendor, doc_date, date_raw,
                      total, address, latency_ms, cost_usd, status, raw)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (DATASET, ARM, r["doc_id"],
                 norm_text(r.get("vendor")), d,
                 None if raw_date is None else str(raw_date),
                 norm_money(r.get("total")), norm_text(r.get("address")),
                 r.get("latency_ms"), r.get("cost_usd") or 0,
                 r.get("status"), json.dumps(r)),
            )
            n += 1

        conn.commit()

    print(f"loaded {n} documents, {skipped} skipped")
    print(f"dates: {epoch} epoch-format, {unparsed} unparseable")


if __name__ == "__main__":
    main()
