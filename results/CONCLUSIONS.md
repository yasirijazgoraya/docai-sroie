# Conclusions — Best Configuration per Layer

Final summary of the SROIE pipeline study: which component won at each layer,
what it costs, and why it matters. Full per-question evidence in
[`ANALYSIS.md`](ANALYSIS.md); full grids in [`RESULTS_DETAILED.md`](RESULTS_DETAILED.md).

## Best component per layer

| Layer | Component | Selected | Performance | Cost | Significance |
|---|---|---|---|---|---|
| 1 — Extraction | Vision model | **Qwen2.5-VL-7B zero-shot** (4-bit, local GPU) | vendor 0.951 · date 0.968 · total 0.977 · address 0.908 | $0 / page | Beat AWS Textract and Claude Haiku on every field; fine-tuning added nothing on these fields |
| 2 — Storage | Store design | **PostgreSQL + pgvector, dual store** with canonical vendor IDs | aggregates 0.667 → 0.782 after canonicalisation | $0 (single instance) | Structured fields queried by SQL, never embedded; load-time vendor resolution fixed both under- and over-grouping |
| 3 — Retrieval | Document resolution | **Structured-first** (vendor+date → SQL; dense fallback) | retrieval hit 0.975 vs 0.340 dense | $0 | Largest single effect in the study: +0.565 lookup accuracy |
| 3 — Embeddings | Fallback encoder | **Titan v2** (Bedrock) | dense hit 0.730 vs MiniLM 0.340 | ~$0.01 / 287 q | Matters only when structured resolution fails (~5% of lookups) |
| 3 — Answering LLM | Generator | **Either** — Qwen (local) or Mistral Large | 0.880 vs 0.890 lookup | $0 vs $0.74 / 287 q | LLM choice moved accuracy ≤0.015 across all four retrieval configurations |
| 3 — Compositional | Multi-step | **Agent** (Mistral + 3 narrow tools) | 0.972 vs router 0.028 | $0.17 / 36 q | Necessary for comparisons and rankings; adds nothing on single-intent questions (0.836 vs 0.850) |

## The two headline configurations

| | Fully local | Best cloud-assisted |
|---|---|---|
| Configuration | Qwen2.5-VL + MiniLM + structured routing | Qwen extraction + Titan + Mistral + agent routing |
| Single-intent accuracy (287 q) | 0.833 | 0.857 |
| Multi-step accuracy (36 q) | — | 0.972 |
| Hallucination rate | <1% | <1% |
| Cost per 287 questions | **$0.00** | **$0.91** |
| Data leaves premises | No | Queries only, not documents |

## Findings

**Retrieval design dominates model choice.** Retrieval interventions moved
lookup accuracy by +0.565 (structured-first) and +0.355 (embedding model);
swapping the answering LLM between a local 7B and a commercial frontier model
moved it by at most +0.015. The ordering held in every one of the four
configurations tested (H3).

**The free configuration is competitive with the paid one.** 0.833 vs 0.857
overall at $0.00 vs ~$0.91 per 287 questions, running 4-bit on a single
consumer GPU (H5).

**Route by question complexity.** SQL for aggregates, structured lookup for
single receipts, an agent only for compositional questions. Each mechanism is
near its ceiling within its class and wasteful outside it: the agent matched
the router on single-intent questions while costing 1.5× more, and the router
scored 0.028 on compositional questions — including three cases where it
returned a bare single-vendor total in answer to a comparison, a silent wrong
answer with no signal to the user.

**The accuracy ceiling is extraction, not answering.** Remaining aggregate
errors trace to Layer-1 OCR noise in vendor names and totals. SROIE's own gold
annotations contain the same class of noise (e.g. `KEDA PAPAN YEW CHJAN`), so
aggregate gold is defined over canonical vendor identities, with the same
clustering applied to gold and predictions.

**Reliability came from the prompt, not the models.** With an explicit
`NOT_FOUND` option, hallucination stayed at or below 1% in every configuration;
retrieval failure produced abstention rather than invention.

Total study cost: under $10 against a $100/month budget.
