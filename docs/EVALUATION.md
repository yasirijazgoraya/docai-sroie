# Evaluation Methodology

How every number in this study is produced: what is measured, how matching
works, why each metric was chosen, and which metrics were rejected. Read this
before reading any results table.

---

## 1. Principle

The pipeline is apparatus; the contribution is the empirical knowledge it
produces. Every arm is therefore held to the same measurement, on the same test
split, with the same normalization applied identically. Where a metric flatters
one arm by construction, that is stated in the results rather than hidden.

Three rules hold across all layers:

1. **Same test split for every arm.** No arm is scored on data another arm did
   not see.
2. **No contamination.** Fine-tuning trains on `train` only; the dev slice is
   carved out of `train`, never `test`. No paid model was ever used to label or
   clean open-model training data — doing so would make the open-vs-paid
   comparison meaningless.
3. **Extraction and scoring are separate passes.** Extraction writes
   `outputs/<arm>__<dataset>__test.jsonl`; scoring reads those files. Metrics
   can therefore change without re-running expensive extraction, and every
   reported number is recomputable from cached predictions.

---

## 2. Choosing the metric: it follows from the field, not from preference

The correct metric depends on whether a field holds **one value** or **many**.

### 2.1 Single-value fields → accuracy

SROIE `vendor`, `date`, `total`, `address`; CORD `total`.

Each holds exactly one value per document. The model either recovers it or does
not. There is no possibility of emitting two vendors and being partly right, so
**precision = recall = F1 = accuracy**. Reporting four identical columns would
be noise; one column is reported.

### 2.2 Multi-value fields → field-level F1 with separate precision and recall

CORD `line_items`.

A receipt has many line items, so an extractor can fail in two independent
directions: miss items (recall drops) or invent them (precision drops). A single
number hides which. Precision and recall are therefore reported separately, with
F1 as the headline.

This matters concretely. Receipts repeat numeric strings — a price appears in
the line, again in a subtotal, again in the total — so extractors over-emit and
**precision falls below recall**. This is a known, designed-around failure mode,
not a surprise: structured values are deduplicated before storage in Layer 2 for
exactly this reason.

### 2.3 Form key-value extraction → entity-level P/R/F1

FUNSD. Arbitrary keys, so the receipt schema does not apply. Scored as
key-value pairs with precision and recall reported separately.

### 2.4 Document QA → ANLS

DocVQA, if brought into scope. Answers are spans grounded in provided OCR;
ANLS is the benchmark's own metric and tolerates minor string variation.

---

## 3. Normalization

Applied identically to every arm. **Format is never allowed to count as an
extraction error.**

| Field | Rule |
|---|---|
| vendor, address, descriptions | lowercase; whitespace- and punctuation-insensitive |
| total, prices | numeric comparison with ±1% tolerance, absorbing sub-cent rounding |
| date | parsed to a canonical date before comparison |

The date rule is the one that most affects cross-arm fairness. `15/01/2019`,
`2019-01-15` and `15 Jan 2019` are the same date written three ways, and Claude
returns ISO while receipts print DMY. Canonicalising before comparison means
the reported figure reflects **content, not presentation**. The same logic
covers Textract splitting addresses into components: whitespace-insensitive
matching absorbs it.

A worked example of why this discipline matters: the fine-tuned arm emits 3
dates (of 346) as epoch-millisecond floats. That was initially suspected of
depressing its date score. Measurement rejected the hypothesis — 0.9% of
records cannot account for a 6.9-point gap, and the separate adapter has zero
epoch records while scoring similarly. **The suspected artifact was tested and
found not to be one**, and the result stands as real extraction behaviour.

---

## 4. Confidence intervals

Point estimates alone cannot distinguish a real difference from noise.

- **Proportions** (per-field accuracy, n = 347): **Wilson score interval**.
  Exact for a proportion; bootstrapping is unnecessary.
- **Composite metrics** (field-level F1, which is not a simple proportion):
  **95% bootstrap CI** over documents.

Reading rule, applied consistently: **overlapping intervals mean the difference
is reported as noise.** This is what licenses saying the Qwen2-VL lead over the
OCR baselines on SROIE is real (non-overlapping) while the PaddleOCR/Qwen2-VL
difference on CORD is not (overlapping) — and it is why a 0.016 F1 separation
on a 50-document FUNSD split is reported as a three-way tie rather than a win.

---

## 5. Rejected and deprecated metrics

**Token-level F1 — deprecated, not reported.** It compares bags of tokens
between prediction and gold, penalising every extra emitted word. An extractor
that correctly recovers all four SROIE fields but also emits surrounding receipt
text scores poorly despite perfect key-information extraction. It measures
verbosity, not extraction quality. Concretely, the arm that scores 0.817
field-level F1 on SROIE scores 0.349 on token F1 — the same extraction, a metric
that obscures it. Any surviving token-F1 column in older documents should be
deleted, not merely footnoted.

**Recall-only scoring — superseded.** An earlier scorer reported field recall
alone. A backend that dumps the whole page scores high on recall while emitting
large amounts of noise. Precision was added so over-emission is penalised.

---

## 6. Fairness caveats that must accompany results

State these wherever the numbers appear:

- **Cost figures are derived, not billed.** Cloud costs are computed from list
  price and per-document counts. Local model cost is 0 *at inference time only*;
  GPU running cost is amortized separately in the cost analysis.
- **Latency is hardware-specific, not a model property.** Open-arm timings are
  4-bit-quantized on an RTX 4080 16 GB. They are not comparable to cloud API
  latency as a claim about model speed.
- **Scoring generations are not comparable.** Where two scorers exist, their
  numbers must never share a table column — a change of model, scorer, and
  metric definition at once can masquerade as an improvement.
- **Leaderboard SOTA is cited, never tabulated.** SROIE Task 3 SOTA
  (StrucTexT, 98.7% F1) is a fine-tuned reference ceiling, cited for context and
  never placed alongside these arms.
- **Fine-tuned vs zero-shot must always be labelled.** Comparing a fine-tuned
  arm to a zero-shot one without labelling is not a fair comparison.

---

## 7. Layer 3 evaluation (planned)

SROIE is an extraction benchmark with no questions, so a question set is
generated from **gold** structured fields — never from predictions, so that
Layer-1 extraction error is not scored as Layer-3 retrieval failure.

Two question classes map to the two retrieval paths:

| Class | Example | Expected path |
|---|---|---|
| lookup | *What was the total on this receipt?* | vector retrieval |
| aggregate | *How much did I spend at this vendor in total?* | SQL over the structured store |

Aggregate questions are included specifically because **retrieval is expected to
fail them**. Top-k retrieval cannot sum across documents, and a model asked to
add up retrieved chunks will produce a plausible wrong number. Measuring that
failure is what justifies routing aggregation to SQL rather than asserting it.

Planned measures: answer accuracy per class; retrieval hit rate (is the gold
document in top-k); and hallucination rate, separated into cases where retrieval
succeeded and cases where it failed — the separation that H4 requires.

### Known limitation

H3 states that chunking strategy dominates answer accuracy. A receipt is short
enough to be a single chunk, so on SROIE chunking has almost nothing to vary.
Embedding choice and top-k remain testable; chunking does not. Either a
long-document corpus (DocVQA) is added, or H3 is narrowed to the variables the
receipt corpus can actually test. This is a scope decision, and it should be
recorded as one rather than left as an unexplained gap.
