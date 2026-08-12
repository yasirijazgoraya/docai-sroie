# Pipeline Layers — Methodology and Results

SROIE receipt processing: extraction → storage → question answering.
Each component is marked **LOCAL** (free, runs on one consumer GPU) or
**CLOUD** (paid AWS Bedrock service, `eu-west-1`).

Total study cost: under **$10**.

---

# Layer 1 — Vision / Key Information Extraction

## Dataset
- SROIE test split: **347 scanned receipts**
- Fields extracted: company/vendor, date, address, total

## Arms compared

| Type | Arm | Deployment | Cost |
|---|---|---|---|
| OCR baseline | Tesseract, PaddleOCR | LOCAL | $0 |
| Vision-language model | Qwen2.5-VL-7B zero-shot (4-bit) | LOCAL | $0 |
| Vision-language model | Qwen2.5-VL-7B LoRA fine-tuned | LOCAL | $0 |
| Commercial OCR/KIE | AWS Textract (AnalyzeExpense) | **CLOUD — paid** | ~$0.01/page |
| Commercial MLLM | Claude Haiku 4.5 (Bedrock) | **CLOUD — paid** | ~$0.003/page |

## Results — exact-match accuracy

| Arm | vendor | date | total | address |
|---|---|---|---|---|
| **Qwen2.5-VL zero-shot** (LOCAL) | **0.951** | **0.968** | **0.977** | **0.908** |
| Qwen2.5-VL LoRA (LOCAL) | 0.948 | 0.899 | 0.963 | 0.816 |
| AWS Textract (CLOUD) | 0.856 | 0.968 | 0.911 | 0.729 |
| Claude Haiku 4.5 (CLOUD) | 0.853 | 0.888 | 0.928 | 0.749 |

Single-value fields, so accuracy = precision = recall = F1. Token-level F1 was
deprecated: it understates structured-extraction quality.

## Outcome

**Selected: Qwen2.5-VL-7B zero-shot (LOCAL).** Highest on every field, $0 per
page, 4-bit quantised on a single consumer GPU. No paid service won any field.

Fine-tuning gave no gain because the base model was already ≥0.90 on all four
fields. It pays where the task is structurally harder: on CORD line items,
field F1 roughly doubles (0.409 → 0.831).

---

# Layer 2 — Storage and Indexing

## Infrastructure

**PostgreSQL 16 + pgvector, single local instance (LOCAL, $0).** No managed
cloud database. AWS Knowledge Bases with OpenSearch was costed and rejected —
roughly $700/month minimum, far beyond an SME budget and beyond this study's.

## Two stores, one extraction

| Store | Contents | Purpose | Queried by | Deployment |
|---|---|---|---|---|
| `documents` | 347 records: vendor, date, total, address, `vendor_id` | exact arithmetic | SQL | LOCAL |
| `chunks` | OCR text + 384-d MiniLM embeddings | semantic retrieval | vector similarity | LOCAL |
| `chunks_titan` | OCR text + 1024-d Titan v2 embeddings | semantic retrieval | vector similarity | **CLOUD — paid** (~$0.01 total) |

**Boundary rule:** structured fields are queried directly with SQL and are never
chunked or embedded; only full OCR text is embedded.

**Why two stores.** SQL needs exact values in typed columns — you cannot `SUM`
text. Retrieval needs text in vectors — you cannot search four fields for
content they do not contain. One store gives one capability; the measured cost
of using only vectors is aggregate accuracy of 0.011.

## Processing applied
- Money normalisation (locale-aware), date parsing across 17 formats
- Deduplication of repeated numeric strings (receipts repeat totals, so
  extractors over-emit)
- Vendor canonicalisation at load time: 150 raw strings → 143 canonical vendors
- Chunking strategies: whole (347), window (883), lines (17,266)

## Findings

**Index design.** One *partial* HNSW index per chunking strategy is required. A
single index over the table filters after the approximate scan, so the 50:1
lines-to-whole row ratio starved the other strategies of results.

**Vendor canonicalisation.** OCR splits one company across variants
(`GARDENIA BAKERIES (KL)` vs `(KI)`) while query-time fuzzy matching merges
distinct ones (`SUPER SEVEN CASH & CARRY` vs `SEGI CASH & CARRY`). Resolving
identity once at load time fixed both directions:
**aggregate accuracy 0.667 → 0.782.**

SROIE's own gold annotations contain the same class of noise
(`KEDA PAPAN YEW CHJAN` for `KEDAI PAPAN YEW CHUAN`). Aggregate gold is
therefore defined over canonical identities, with the same clustering applied
to gold and to predictions — no answer key built on raw string equality is
coherent here.

---

# Layer 3 — Question Answering

## Question sets

| Set | n | Composition | Gold source |
|---|---|---|---|
| Single-intent | 287 | 200 lookup, 87 aggregate | SROIE gold annotations |
| Multi-step | 36 | 18 compare, 10 chain, 5 extreme, 3 rank | SQL over canonical vendors |

## Arms compared

| Arm | Method | Embeddings | Answering LLM |
|---|---|---|---|
| A | Dense RAG — top-k retrieval for every question | MiniLM (LOCAL) / Titan (CLOUD) | Qwen (LOCAL) / Mistral (CLOUD) |
| B | Router — SQL for aggregates, dense retrieval for lookups | MiniLM (LOCAL) | Qwen (LOCAL) |
| C | Structured-first — resolve receipt by vendor+date, dense as fallback | MiniLM / Titan | Qwen / Mistral |
| D | Agent — bounded tool loop (`aggregate`, `find_receipts`, `list_vendors`) | — | Mistral Large (**CLOUD — paid**) |

Paid components: **Titan v2 embeddings** and **Mistral Large** answering, both
on AWS Bedrock in `eu-west-1`. Everything else runs locally at zero marginal
cost.

## Results

| Arm | Embed | LLM | lookup (200) | aggregate (87) | multi-step (36) | cost / 287 q |
|---|---|---|---|---|---|---|
| A | MiniLM (L) | Qwen (L) | 0.315 | 0.011 | — | $0.00 |
| A | Titan (C) | Qwen (L) | 0.670 | 0.069 | — | ~$0.01 |
| B | MiniLM (L) | Qwen (L) | 0.315 | 0.782 | 0.028 | $0.00 |
| C | MiniLM (L) | Qwen (L) | 0.860 | 0.782 | — | $0.00 |
| C | Titan (C) | Qwen (L) | 0.880 | 0.782 | — | ~$0.01 |
| C | Titan (C) | Mistral (C) | 0.890 | 0.782 | — | $0.74 |
| D | — | Mistral (C) | 0.890 | 0.713 | **0.972** | $1.12 |

(L) = local, free · (C) = cloud, paid

## Effect sizes — same questions, same prompt

| Intervention | Δ lookup accuracy |
|---|---|
| Retrieval design (dense → structured-first) | **+0.565** |
| Embedding model (MiniLM → Titan v2) | **+0.355** |
| Answering LLM (local 7B → commercial frontier) | **+0.015** |

## Findings

**H3 supported.** Retrieval design dominates LLM choice by roughly 40×, and the
ordering holds in every one of the four configurations tested.

**Aggregation must be arithmetic, not generation.** Retrieval-based aggregates
scored 0.011–0.069; SQL over the structured store scored 0.782 — identical to
three decimal places for both LLM families, because the answer comes from the
database rather than the model.

**H4 refined.** Hallucination stayed at or below 1% in every configuration.
Given an explicit `NOT_FOUND` option, retrieval failure produced *abstention*
rather than invention — a property of the prompt design, not of the models.

**Agency is not general capability.** On single-intent questions the agent
matched the router (0.836 vs 0.850) while costing 1.5× more and taking 2.6
steps instead of one. On compositional questions the router scored 0.028 — and
on 3 of 18 comparisons returned a bare single-vendor total, a silent wrong
answer with no signal to the user. **Design rule: route by question complexity;
reserve the agent for compositional queries.**

**H5 answered.** The fully local configuration reached 0.833 overall against
0.857 for the paid one — 2.4 points apart, at $0.00 versus ~$0.91 per 287
questions.

## Limitations

- Questions were generated from gold annotations, so vendor+date is always a
  well-formed handle. The structured arm is an **upper bound** for
  well-specified questions; the dense arm is the **lower bound** for vague
  phrasing. Real SME questions fall between.
- Year aggregates (n=6) score zero under exact match while within 5% relative
  error — extraction error compounds over 200+ receipts.
- One dataset, one language (English Malaysian receipts). CORD and FUNSD were
  evaluated at Layer 1 only.
- Mistral costs are estimated from character counts; Bedrock's Mistral response
  carries no usage block.
- The agent required three prompt corrections (date convention, bare-year
  guard, max-in-year hint) to reach 0.972. Brittleness is part of the cost of
  agency.
