# Cloud-Native Document AI for SMEs — SROIE Pipeline

Extraction → storage → question answering over business receipts, evaluated to
determine what is accurate, reliable, and economically viable for small and
medium enterprises.

**Headline result:** retrieval design determines answer accuracy; the choice of
answering LLM barely matters. Across four retrieval configurations, swapping a
local 7B model for a commercial frontier model changed lookup accuracy by at
most 0.015, while retrieval design changed it by 0.565.

Dataset: SROIE test split, 347 scanned English receipts.

---

## Results

### Layer 1 — extraction (per-field accuracy, 347 receipts)

| Field | Qwen2.5-VL zero-shot | Qwen2.5-VL LoRA | AWS Textract | Claude Haiku 4.5 |
|---|---|---|---|---|
| vendor | **0.951** | 0.948 | 0.856 | 0.853 |
| date | **0.968** | 0.899 | **0.968** | 0.888 |
| total | **0.977** | 0.963 | 0.911 | 0.928 |
| address | **0.908** | 0.816 | 0.729 | 0.749 |

No paid service wins any field. The open model runs 4-bit quantized on a single
consumer GPU at zero inference cost.

Fine-tuning is flat-to-lower here because the base model is already ≥0.90 on
every field. It pays where the base model is weak: on CORD line items it roughly
doubles field F1 (0.409 → 0.831).

### Layer 3 — question answering (287 questions: 200 lookup, 87 aggregate)

**Lookup accuracy** — answerable from one receipt:

| Retrieval | Embeddings | Qwen2.5-VL (local) | Mistral Large (cloud) | Δ LLM |
|---|---|---|---|---|
| fixed RAG | MiniLM | 0.315 | 0.320 | +0.005 |
| fixed RAG | Titan v2 | 0.670 | 0.670 | 0.000 |
| structured-first | MiniLM | 0.860 | 0.875 | +0.015 |
| structured-first | Titan v2 | **0.880** | **0.890** | +0.010 |

Effect sizes, same questions and prompt throughout:

| Intervention | Δ lookup accuracy |
|---|---|
| retrieval design (dense → structured-first) | **+0.565** |
| embedding model (MiniLM → Titan v2) | **+0.355** |
| answering LLM (local 7B → commercial) | **+0.010** |

**Aggregate accuracy** — spanning many receipts:

| Retrieval | Qwen2.5-VL | Mistral Large |
|---|---|---|
| fixed RAG + MiniLM | 0.011 | 0.103 |
| fixed RAG + Titan | 0.069 | 0.287 |
| routed to SQL | **0.690** | **0.690** |

Two things to note. Under fixed RAG the LLM *does* matter for aggregation —
retrieval hit rate is 0.989 in both Titan rows, so both models see the same
documents and Mistral simply aggregates better. And once routed to SQL both
models score identically to three decimals, because the answer comes from the
database rather than the model.

**Overall, and cost:**

| Configuration | Overall accuracy | Cost / 287 questions |
|---|---|---|
| local (Qwen + MiniLM + structured) | 0.822 | **USD 0.00** |
| cloud (Mistral + Titan + structured) | **0.829** | USD 0.74 |

The free configuration is within 0.7 points of the paid one.

### Reliability

Hallucination — an answer appearing in no retrieved text — stayed at or below
1% in every configuration. The prompt offers an explicit `NOT_FOUND` option, and
retrieval failure produces abstention rather than invention. Removing that
option would likely change this substantially; the low rate is a property of the
prompt design, not of the models.

---

## Architecture

```
receipt image
     │
     ├─ Layer 1  Qwen2.5-VL (4-bit, local GPU)
     │            ├─→ structured fields ──→ documents table    (SQL, never embedded)
     │            └─→ full OCR text ──────→ chunks table       (embedded, retrieval only)
     │
     ├─ Layer 2  PostgreSQL 16 + pgvector, single instance
     │
     └─ Layer 3  question → intent (JSON) → route
                    ├─ aggregate  → SQL over documents
                    └─ lookup     → resolve vendor+date in documents,
                                    fetch that receipt's text,
                                    answer from it
                                    (dense retrieval as fallback)
```

**Boundary rule:** structured fields are queried directly and are never chunked
or embedded. Only full document text is embedded. This separation is what makes
the aggregate result possible.

---

## Repository layout

```
layer1_extraction/     extraction arms and scoring
layer2_storage/        schema, structured loader, chunking + embedding
layer3_qa/             question generation, three answering arms, scoring
results/               scored run outputs (JSONL) and summary tables
docs/                  methodology, evaluation, architecture diagram
```

---

## Reproducing

```bash
# database
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
sudo -u postgres createuser --createdb $USER
sudo -u postgres createdb -O $USER docai
sudo -u postgres psql -d docai -c "CREATE EXTENSION vector;"

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# layer 2
python layer2_storage/load_structured.py
python layer2_storage/build_chunks.py                    # MiniLM, local
python layer2_storage/build_chunks_titan.py --init       # Titan, Bedrock

# layer 3
python layer3_qa/build_qa.py
python layer3_qa/run_rag_local.py --strategy whole       # arm A
python layer3_qa/run_router.py    --strategy whole       # arm B
python layer3_qa/run_hybrid.py    --embed titan          # arm C
python layer3_qa/run_mistral.py   --arm c --embed titan  # arm C, cloud LLM

python layer3_qa/score_rag.py results/runs/*.jsonl
```

Layer-1 extraction is not reproduced here — it requires the model weights and
several GPU-hours. Its outputs are committed under `data/extractions/` as the
frozen input to Layers 2 and 3.

AWS runs use `eu-west-1` with Bedrock; credentials come from `~/.aws/credentials`
and are never committed.

---

## Limitations

**Questions are generated from gold annotations**, so vendor + date is a
well-formed handle by construction. The structured-first arm is therefore an
upper bound for well-specified questions; the dense-retrieval arm is the lower
bound for vague phrasing. Real SME questions fall between the two.

**Aggregate accuracy is bounded by extraction accuracy.** Summing across many
receipts compounds per-document error: year totals were within 5% of correct but
scored zero under exact match. Relative error is the fairer measure there, and
is reported alongside.

**Vendor-name variants fragment SQL grouping.** OCR produces `MR. D.I.Y. (M)`
and `MR. D.I.Y.: (M)` as distinct strings. Query-time IDF matching mitigates
this; a canonical vendor table resolved at load time would be the proper fix.

**One dataset, one language.** SROIE is English Malaysian receipts. CORD
(Indonesian) and FUNSD (forms) were evaluated at Layer 1 only.

**Mistral costs are estimated** from character counts — Bedrock's Mistral
response carries no usage block. Local cost is zero at inference time; amortized
GPU cost is handled separately in the cost analysis.

**Single-intent questions only.** The router handles one intent per question.
Comparative questions ("did I spend more at X or Y?") and chained lookups are
not supported, and are the case for an agentic arm.
