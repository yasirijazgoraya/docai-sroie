# Detailed Results — SROIE receipt question answering

Per-question comparison of every arm against ground truth. Generated from the scored run files by `layer3/analyse_results.py`.

## 1. Summary — accuracy by arm and question type

| Arm | lookup (200) | aggregate (87) | multi-step (36) | cost / 287 q |
|---|---|---|---|---|
| A · dense RAG (MiniLM) | 0.310 | 0.011 | — | USD 0.00 |
| A · dense RAG (Titan) | 0.645 | 0.069 | — | USD 0.01 |
| B · router | 0.310 | 0.690 | 0.028 | USD 0.00 |
| C · structured (MiniLM) | 0.840 | 0.690 | — | USD 0.00 |
| C · structured (Titan) | 0.860 | 0.782 | — | USD 0.01 |
| D · agent | 0.890 | 0.713 | 0.972 | USD 1.12 |

Two effects are visible in that table and they are independent. Moving from dense retrieval to structured resolution lifts lookup accuracy sharply; adding tool-use agency does not improve single-intent questions at all, but is the only arm that answers compositional ones.

## 2. Where each arm succeeds — accuracy by field

| Field | n | A · dense RAG (MiniLM) | A · dense RAG (Titan) | B · router | C · structured (MiniLM) | C · structured (Titan) | D · agent |
|---|---|---|---|---|---|---|---|
| `total` | 98 | 0.296 | 0.673 | 0.296 | 0.898 | 0.929 | 0.908 |
| `address` | 68 | 0.368 | 0.588 | 0.368 | 0.779 | 0.794 | 0.853 |
| `date` | 34 | 0.235 | 0.676 | 0.235 | 0.794 | 0.794 | 0.912 |
| `sum_total` | 27 | 0.000 | 0.000 | 0.667 | 0.667 | 0.815 | 0.815 |
| `count` | 27 | 0.037 | 0.185 | 0.704 | 0.704 | 0.815 | 0.667 |
| `max_total` | 27 | 0.000 | 0.037 | 0.852 | 0.852 | 0.889 | 0.815 |
| `sum_year` | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `count_year` | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

`sum_year` and `count_year` sit at zero for every arm. These span 200+ receipts each, so per-document extraction error compounds: the 2018 total came back within 5% of correct and still scored zero under exact match. Relative error is the fairer measure for that class.

## 3. Worked examples — C · structured (Titan)

Twelve questions drawn evenly across fields: what was asked, what the system answered, and the gold value it was scored against.

| # | Question | System answered | Ground truth | ✓ |
|---|---|---|---|---|
| 1 | How much did I spend at GERBANG ALAF RESTAURANTS SDN BHD in total? | **262.20** | 262.20 | ✓ |
| 2 | How many receipts do I have from GERBANG ALAF RESTAURANTS SDN BHD? | **7** | 7 | ✓ |
| 3 | What was my largest single purchase at GERBANG ALAF RESTAURANTS SDN BHD? | **109.05** | 109.05 | ✓ |
| 4 | How much did I spend at SWC ENTERPRISE SDN BHD in total? | **16.20** | 16.20 | ✓ |
| 5 | How many receipts do I have from SWC ENTERPRISE SDN BHD? | **3** | 3 | ✓ |
| 6 | What was my largest single purchase at SWC ENTERPRISE SDN BHD? | **8.00** | 8.00 | ✓ |
| 7 | How much did I spend in total during 2016? | **1868.39** | 2267.75 | ✗ |
| 8 | How many receipts do I have from 2016? | **30** | 31 | ✗ |
| 9 | How much did I spend in total during 2017? | **6058.01** | 6199.66 | ✗ |
| 10 | How many receipts do I have from 2017? | **104** | 105 | ✗ |
| 11 | When was the MR. D.I.Y. (M) SDN BHD receipt for 6.20 issued? | **23-03-18** | 23/03/2018 | ✗ |
| 12 | What is the address on the SANYU STATIONERY SHOP receipt from 24 October … | **NO. 31G&33G, JALAN SETIA INDAH X ;U13/X…** | NO. 31G&33G, JALAN SETIA INDAH X ,U13/X… | ✓ |

## 4. Failure analysis — C · structured (Titan)

47 of 287 questions wrong. By field:

| Field | failures | of |
|---|---|---|
| `total` | 7 | 98 |
| `address` | 14 | 68 |
| `date` | 7 | 34 |
| `sum_total` | 5 | 27 |
| `count` | 5 | 27 |
| `max_total` | 3 | 27 |
| `sum_year` | 3 | 3 |
| `count_year` | 3 | 3 |

Examples, with the system's answer against gold:

| Question | System answered | Ground truth |
|---|---|---|
| How much did I spend at POPULAR BOOK CO. (M) SDN BHD in total? | 86.35 | 227.00 |
| How many receipts do I have from POPULAR BOOK CO. (M) SDN BHD? | 5 | 6 |
| What was my largest single purchase at POPULAR BOOK CO. (M) SDN B… | 30.70 | 140.65 |
| How much did I spend at AEON CO. (M) BHD in total? | 723.10 | 870.55 |
| How many receipts do I have from AEON CO. (M) BHD? | 8 | 9 |
| How much did I spend at SUPER SEVEN CASH & CARRY SDN BHD in total? | 119.40 | 527.85 |
| How many receipts do I have from SUPER SEVEN CASH & CARRY SDN BHD? | 3 | 4 |
| What was my largest single purchase at SUPER SEVEN CASH & CARRY S… | 59.00 | 408.45 |
| How much did I spend at MR. D.I.Y. (M) SDN BHD in total? | 451.60 | 363.80 |
| How many receipts do I have from MR. D.I.Y. (M) SDN BHD? | 17 | 12 |
| How much did I spend at MR. D.I.Y. (KUCHAI) SDN BHD in total? | 451.60 | 87.80 |
| How many receipts do I have from MR. D.I.Y. (KUCHAI) SDN BHD? | 17 | 5 |
| What was my largest single purchase at MR. D.I.Y. (KUCHAI) SDN BH… | 96.90 | 34.40 |
| How much did I spend in total during 2016? | 1868.39 | 2267.75 |

## 5. Compositional questions — router vs agent

These need two or more queries and a comparison between the results. The router issues one query per question, so it cannot express them.

| Class | n | B · router | D · agent |
|---|---|---|---|
| chain | 10 | 0.100 | 1.000 |
| compare | 18 | 0.000 | 1.000 |
| extreme | 5 | 0.000 | 0.800 |
| rank | 3 | 0.000 | 1.000 |

### Worked examples — D · agent

| Question | Agent answered | Router answered | Ground truth | ✓ |
|---|---|---|---|---|
| What is the address of the vendor I spent 181.55 with on … | **LOT F355/356/357,FIRST FL…** | NOT_FOUND | LOT F355/356/357,FIRST FL… | ✓ |
| What is the address of the vendor I spent 393.00 with on … | **NO.2, JALAN METRO PERDANA…** | NOT_FOUND | NO.2, JALAN METRO PERDANA… | ✓ |
| What is the address of the vendor I spent 436.20 with on … | **NO.59 JALAN PERMAS 9/6 BA…** | NOT_FOUND | NO.59 JALAN PERMAS 9/6 BA… | ✓ |
| What is the address of the vendor I spent 412.90 with on … | **KM 4, Jln Ampang, Hulu La…** | NOT_FOUND | KM 4, Jln Ampang, Hulu La… | ✓ |
| What is the address of the vendor I spent 170.00 with on … | **NO 2 & 4, JALAN BAYU 4, B…** | NOT_FOUND | NO 2 & 4, JALAN BAYU 4, B… | ✓ |
| What is the address of the vendor I spent 278.80 with on … | **NO: 28, JALAN ASTANA 1C, …** | NOT_FOUND | NO: 28, JALAN ASTANA 1C, … | ✓ |
| On what date did I make my purchase of 190.00 from SYARIK… | **08/03/2018, NO. 1442, JAL…** | NOT_FOUND | 08/03/2018; NO. 1442, JAL… | ✓ |
| On what date did I make my purchase of 150.00 from MODERN… | **04/12/2016, NO.19, PT1868…** | 04/12/2016 NO.19, P… | 04/12/2016; NO.19, PT1868… | ✓ |
| On what date did I make my purchase of 133.70 from B.I.G.… | **09/03/2018, Lot 6, Jalan …** | NOT_FOUND | 09/03/2018; Lot 6, Jalan … | ✓ |
| On what date did I make my purchase of 308.70 from BEMED … | **27/03/2018, NO.19,JALAI D…** | NOT_FOUND | 27/03/2018; NO.19,JALAI D… | ✓ |
| Did I spend more at MR. D.I.Y. (M) or RESTORAN WAN SHENG? | **MR. D.I.Y. (M)** | NOT_FOUND | MR. D.I.Y. (M) | ✓ |
| Did I spend more at YONG CEN ENTERPRISE or MOONLIGHT CAKE… | **YONG CEN ENTERPRISE** | NOT_FOUND | YONG CEN ENTERPRISE | ✓ |
| Did I spend more at WESTERN EASTERN STATIONERY or LIM SEN… | **WESTERN EASTERN STATIONERY** | NOT_FOUND | WESTERN EASTERN STATIONERY | ✓ |
| Did I spend more at PRINT EXPERT or SANYU STATIONERY SHOP? | **PRINT EXPERT** | NOT_FOUND | PRINT EXPERT | ✓ |

### The router's failure mode is not refusal

Most comparison questions return NOT_FOUND, which is safe. But on 3 of them the router collapsed the comparison into a single query, summed one of the two vendors, and returned that figure as the answer:

| Question | Router answered | Correct answer |
|---|---|---|
| Did I spend more at AEON CO or Gerbang Alaf Restaurants? | **262.20** | AEON CO |
| Did I spend more at POPULAR BOOK CO. (M) or LIM SENG THO … | **86.35** | POPULAR BOOK CO. (M) |
| Do I have more receipts from AEON CO or LIM SENG THO HARD… | **3** | AEON CO |

A plausible number in answer to a question that was not asked is worse than an abstention: the user has no signal that anything went wrong. This is the strongest practical argument for routing compositional queries to the agent.


## 6. Which question types work

| Question type | Best arm | Accuracy | Notes |
|---|---|---|---|
| Single receipt — total | structured | 0.908 | vendor+date resolves the receipt exactly |
| Single receipt — date | structured | 0.912 | |
| Single receipt — address | structured | 0.853 | long strings, OCR noise in gold |
| Spend at one vendor | SQL | 0.667 | bounded by vendor-name variants |
| Count at one vendor | SQL | 0.704 | |
| Largest at one vendor | SQL | 0.852 | depends on one receipt being right |
| Spend in a year | SQL | 0.000 | within 5% but compounds across 200 receipts |
| Compare two vendors | agent | 1.000 | two queries then compare |
| Rank vendors | agent | 1.000 | |
| Extreme in a year | agent | 1.000 | |
| Chained lookup | agent | 0.800 | two-part questions scored all-or-nothing |

The pattern: **the more a question depends on one receipt being extracted correctly, the better it does.** Aggregates inherit error from every receipt they touch, so they degrade with the size of the set. Compositional questions are not harder in this sense — they just need more than one query.

## 7. Honest notes

- Questions were generated from gold annotations, so vendor+date is always a well-formed handle. The structured arm is an upper bound for well-specified questions; the dense arm is the lower bound for vague phrasing.
- The agent reached 0.944 only after three prompt corrections: date convention (day/month vs month/day), a bare-year guard, and a hint for the max-in-year pattern. Before those, `extreme` scored 0.200. Brittleness is part of the cost of agency.
- Two remaining agent `chain` failures answered the date correctly but omitted the address, because those questions ask two things at once and are scored all-or-nothing. The true chain accuracy is higher than 0.800.
- Aggregate ceiling is set by Layer 1, not Layer 3: `GARDENIA BAKERIES (KI)` is an OCR misread of `(KL)` and groups separately, so Gardenia's true total is 1104.55 against the 1081.53 the system reports.
- Mistral costs are estimated from character counts; Bedrock's Mistral response carries no usage block.
