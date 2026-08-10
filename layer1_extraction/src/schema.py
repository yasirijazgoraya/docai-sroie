"""
The COMMON SCHEMA every extraction engine (Textract / MLLM / Qwen zero-shot /
Qwen fine-tuned) is normalized into. Downstream code only ever sees this shape.

Receipts (SROIE, CORD) map cleanly onto these fields. FUNSD forms do not have a
fixed field set, so for forms we will use the generic `kv_pairs` field instead
(added in the second pass). For now the receipt fields are primary.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None


class ExtractionRecord(BaseModel):
    # --- identity ---
    doc_id: str
    dataset: str                       # "sroie" | "cord" | "funsd"
    doc_type: str                      # "receipt" | "form"
    source_engine: str                 # "textract" | "mllm" | "qwen_zeroshot" | "qwen_ft"

    # --- receipt fields (primary for SROIE/CORD) ---
    vendor: Optional[str] = None
    date: Optional[str] = None
    total: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None
    address: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)

    # --- generic key-value (used for FUNSD forms in pass 2) ---
    kv_pairs: dict[str, str] = Field(default_factory=dict)

    # --- full text (this is what gets chunked + embedded for RAG, NOT the fields) ---
    full_text: Optional[str] = None

    # --- measured at extraction time (cannot be recovered later) ---
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    confidence: Optional[float] = None

    # --- bookkeeping ---
    status: str = "extracted"          # extracted -> validated -> scored
    raw_output: Optional[str] = None   # untouched engine output, for audit


class GoldRecord(BaseModel):
    """Ground-truth label loaded from the dataset annotations."""
    doc_id: str
    dataset: str
    doc_type: str
    vendor: Optional[str] = None
    date: Optional[str] = None
    total: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None
    address: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    kv_pairs: dict[str, str] = Field(default_factory=dict)
