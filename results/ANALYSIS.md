# Per-Question Analysis — best configurations

Single-intent arm: **C · structured-first, Titan embeddings, Mistral Large** (`mistral_c_titan__whole__k5.jsonl`)  
Multi-step arm: **D · agent (Mistral Large, tools: aggregate / find_receipts / list_vendors)** (`agent_multistep__k5.jsonl`)

## Outcome definitions

| Outcome | Meaning |
|---|---|
| **TP** | answered and correct |
| **FP** | answered but wrong — the harmful case: a confident wrong answer |
| **FN** | abstained (`NOT_FOUND`) although a correct answer existed |
| **TN** | correctly abstained on an unanswerable question |

Every question in both sets has a gold answer by construction, so **TN is structurally zero** here — reported as n/a rather than silently omitted. Precision = TP/(TP+FP): of the answers the system actually gave, how many were right. For an SME this is the number that matters, because an FP (a plausible wrong figure) is worse than an FN (an honest refusal the user can see).

---

# Part 1 — Single-intent questions (287)

## Lookup · receipt total  (`total`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | How much was the receipt from GERBANG ALAF RESTAURANTS SDN BHD on 24 … | 7.35 | 7.35 | **TP** |
| 2 | How much was the receipt from ALI BABA INTERNATIONAL SDN BHD on 19 Fe… | 23.80 | 23.80 | **TP** |
| 3 | How much was the receipt from MR. D.I.Y. (KUCHAI) SDN BHD on 10 Janua… | RM 8.10 | 8.10 | **TP** |
| 4 | How much was the receipt from MEI LET RESTAURANT on 12 January 2017? | 59.35 | 59.35 | **TP** |
| 5 | How much was the receipt from RESTORAN WAN SHENG on 05 May 2018? | 7.60 | 7.60 | **TP** |
| 6 | How much was the receipt from AEON CO. (M) BHD on 11 May 2018? | 458.55 | 458.55 | **TP** |
| 7 | How much was the receipt from YHM MID VALLEY on 15 December 2017? | 21.60 | 21.60 | **TP** |
| 8 | How much was the receipt from MR. D. I. Y. (KUCHAI) SDN BHD on 19 Mar… | 22.00 | 11.90 | **FP** |
| 9 | How much was the receipt from TED HENG STATIONERY & BOOKS on 27 Janua… | NOT_FOUND (The receipt provided f… | 27.35 | **FN** |
| 10 | How much was the receipt from YIN MA (M) SDN.BHD. on 19 July 2016? | 15.50 | 14.50 | **FP** |
| 11 | How much was the receipt from KEDA PAPAN YEW CHJAN on 08 March 2018? | NOT_FOUND | 312.70 | **FN** |
| 12 | How much was the receipt from MR. D.I.Y. (KUCHAI) SDN BHD on 11 Janua… | 7.00 | 7.00 | **TP** |
| 13 | How much was the receipt from SYARIKAT PERNIAGAAN GIN KEE on 30 Decem… | 89.04 | 89.04 | **TP** |
| 14 | How much was the receipt from B & BEST RESTAURANT on 19 April 2017? | 14.85 | 14.85 | **TP** |
| 15 | How much was the receipt from KEDAI BUKU NEW ACHEIVERS on 15 Septembe… | 48.00 | 48.00 | **TP** |
| 16 | How much was the receipt from HENG KEE DELIGHTS BAK KUT TEH. on 04 Ja… | 42.00 | 42.00 | **TP** |
| 17 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 06 Ap… | $7.60 | 7.60 | **TP** |
| 18 | How much was the receipt from AEON CO. (M) BHD on 21 March 2018? | 65.10 | 65.10 | **TP** |
| 19 | How much was the receipt from BOOK TALK (MUTIARA RINI) SDN BHD on 21 … | 9.30 | 9.30 | **TP** |
| 20 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 05 Ap… | $6.90 | 6.90 | **TP** |
| 21 | How much was the receipt from KEDAI UBAT & RUNCIT HONG NING SDN. BHD.… | 8.00 | 8.00 | **TP** |
| 22 | How much was the receipt from SANYU STATIONERY SHOP on 24 September 2… | 8.70 | 8.70 | **TP** |
| 23 | How much was the receipt from VIVOPAC MARKETING SDN BHD on 13 Septemb… | 52.45 | 52.45 | **TP** |
| 24 | How much was the receipt from RESTORAN WAN SHENG on 21 March 2018? | 4.80 | 4.80 | **TP** |
| 25 | How much was the receipt from 99 SPEED MART S/B on 31 March 2018? | 4.00 | 4.00 | **TP** |
| 26 | How much was the receipt from PETRON BKT LANJAN SB on 01 February 201… | 4.90 | 4.90 | **TP** |
| 27 | How much was the receipt from KEDAI PAPAN YEW CHUAN on 12 March 2018? | 283.55 | 283.55 | **TP** |
| 28 | How much was the receipt from BEMED (SP) SDN. BHD. on 01 April 2017? | 165.00 | 165.00 | **TP** |
| 29 | How much was the receipt from TRIPLE SIX POINT ENTERPRISE 666 on 22 M… | 22.60 | 22.60 | **TP** |
| 30 | How much was the receipt from PAPPARICH BMC on 29 June 2018? | 46.90 | 46.90 | **TP** |
| 31 | How much was the receipt from SUPER SEVEN CASH & CARRY SDN BHD on 01 … | 42.10 | 42.10 | **TP** |
| 32 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 23 Ju… | 26.58 | 26.58 | **TP** |
| 33 | How much was the receipt from 99 SPEED MART S/B on 07 February 2017? | 37.45 | 37.45 | **TP** |
| 34 | How much was the receipt from YONG TAT HARDWARE TRADING on 13 March 2… | 72.00 | 72.00 | **TP** |
| 35 | How much was the receipt from MUN HENG ELECTRICAL & HARDWARE SUPPLY o… | 7.00 | 7.00 | **TP** |
| 36 | How much was the receipt from BECON ENTERPRISE SDN BHD on 13 November… | 270.10 | 270.10 | **TP** |
| 37 | How much was the receipt from SEGI CASH & CARRY SDN.BHD on 06 Februar… | 27.00 | 27.00 | **TP** |
| 38 | How much was the receipt from PASARAYA BORONG PINTAR SDN BHD on 09 Ma… | 10.40 | 10.40 | **TP** |
| 39 | How much was the receipt from BEMED (SP) SDN. BHD. on 20 January 2017? | 203.00 | 203.00 | **TP** |
| 40 | How much was the receipt from ROYALTEA on 09 May 2018? | 10.90 | 10.90 | **TP** |
| 41 | How much was the receipt from PELITA SAMUDRA PERTAMA (M) SDN BHD. on … | 61.70 | 61.70 | **TP** |
| 42 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 31 Au… | 1.52 | 1.52 | **TP** |
| 43 | How much was the receipt from SANYU STATIONERY SHOP on 16 December 20… | 8.70 | 8.70 | **TP** |
| 44 | How much was the receipt from THONG RECIPE on 27 June 2018? | 58.70 | 58.70 | **TP** |
| 45 | How much was the receipt from YHM AEON TEBRAU CITY on 11 March 2018? | 49.70 | 49.70 | **TP** |
| 46 | How much was the receipt from BEMED (SP) SDN BHD. on 27 March 2018? | 308.70 | 308.70 | **TP** |
| 47 | How much was the receipt from SIN LIANHAP SDN BHD on 05 February 2018? | 7.30 | 7.30 | **TP** |
| 48 | How much was the receipt from MY MYDIN SDN BHD on 15 January 2018? | 5.50 | 5.50 | **TP** |
| 49 | How much was the receipt from GERBANG ALAF RESTAURANTS SDN BHD on 18 … | 38.90 | 38.90 | **TP** |
| 50 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 02 Se… | 48.57 | 48.57 | **TP** |
| 51 | How much was the receipt from POPULAR BOOK CO. (M) SDN BHD on 20 Augu… | 5.05 | 5.05 | **TP** |
| 52 | How much was the receipt from PASAR MINI JIN SENG on 03 March 2018? | 33.80 | 33.80 | **TP** |
| 53 | How much was the receipt from MR. D.I.Y. (M) SDN BHD on 22 April 2018? | 27.20 | 27.20 | **TP** |
| 54 | How much was the receipt from TEO HENG STATIONERY & BOOKS on 15 March… | 127.20 | 127.20 | **TP** |
| 55 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 07 Ma… | $8.20 | 8.20 | **TP** |
| 56 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 04 Ma… | $6.60 | 6.60 | **TP** |
| 57 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 29 Au… | 37.90 | 37.90 | **TP** |
| 58 | How much was the receipt from CHOP YEW LIAN on 23 March 2017? | 30.00 | 30.00 | **TP** |
| 59 | How much was the receipt from RESTORAN WAN SHENG on 07 March 2018? | 4.80 | 4.80 | **TP** |
| 60 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 14 Ju… | RM10.15 | 10.15 | **TP** |
| 61 | How much was the receipt from MR. D.I.Y. (M) SDN BHD on 14 April 2018? | 5.00 | 5.00 | **TP** |
| 62 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 02 Oc… | 35.58 | 35.58 | **TP** |
| 63 | How much was the receipt from GRAND COMPANIONS SDN BHD on 03 March 20… | NOT_FOUND | 269.40 | **FN** |
| 64 | How much was the receipt from SUPER SEVEN CASH & CARRY SDN BHD on 01 … | NOT_FOUND | 408.45 | **FN** |
| 65 | How much was the receipt from HIGH FIVE ENTERPRISE on 05 April 2018? | 95.48 | 95.48 | **TP** |
| 66 | How much was the receipt from GERBANG ALAF RESTAURANTS SDN BHD on 30 … | NOT_FOUND | 47.15 | **FN** |
| 67 | How much was the receipt from KECHARA VEGETARIAN RESTAURANT S/B on 15… | 64.50 | 64.50 | **TP** |
| 68 | How much was the receipt from TS TOOLS HARDWARE & MACHINERY SDN BHD o… | 7.40 | 7.40 | **TP** |
| 69 | How much was the receipt from BEYOND BROTHERS HARDWARE on 10 November… | 67.85 | 67.85 | **TP** |
| 70 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 05 Ju… | RM7.50 | 7.50 | **TP** |
| 71 | How much was the receipt from 99 SPEED MART S/B on 06 December 2016? | 85.45 | 85.45 | **TP** |
| 72 | How much was the receipt from MR. D.I.Y. (M) SDN BHD on 27 April 2018? | 1.90 | 1.90 | **TP** |
| 73 | How much was the receipt from SANYU STATIONERY SHOP on 28 October 201… | 21.90 | 21.90 | **TP** |
| 74 | How much was the receipt from KEDAI PAPAN YEW CHUAN on 10 March 2018? | 84.80 | 84.80 | **TP** |
| 75 | How much was the receipt from SWC ENTERPRISE SDN BHD on 08 January 20… | 8.00 | 8.00 | **TP** |
| 76 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 27 Oc… | 35.88 | 35.88 | **TP** |
| 77 | How much was the receipt from LIM SENG THO HARDWARE TRADING on 08 Feb… | 10.50 | 10.50 | **TP** |
| 78 | How much was the receipt from GARDENIA BAKERIES (KL) SDN BHD on 28 Oc… | 32.26 | 32.26 | **TP** |
| 79 | How much was the receipt from HON HWA HARDWARE TRADING on 21 Septembe… | 10.40 | 10.40 | **TP** |
| 80 | How much was the receipt from THE CUT STEAKHOUSE &BURGERS on 01 Janua… | 167.55 | 167.55 | **TP** |
| 81 | How much was the receipt from SANYU STATIONERY SHOP on 12 September 2… | 8.70 | 8.70 | **TP** |
| 82 | How much was the receipt from MR. D.I.Y. (M) SDN BHD on 11 January 20… | 32.80 | 32.80 | **TP** |
| 83 | How much was the receipt from AA PHARMACY on 29 January 2018? | 46.20 | 46.20 | **TP** |
| 84 | How much was the receipt from GLOBAL FOOD EMPIRE SDN BHD on 01 May 20… | 38.35 | 38.35 | **TP** |
| 85 | How much was the receipt from C W KHOO HARDWARE SDN BHD on 31 March 2… | 14.65 | 14.65 | **TP** |
| 86 | How much was the receipt from SYARIKAT PERNIAGAAN GIN KEE on 27 Janua… | 22.26 | 22.26 | **TP** |
| 87 | How much was the receipt from RESTORAN WAN SHENG on 10 March 2018? | 7.20 | 7.20 | **TP** |
| 88 | How much was the receipt from SUPER SEVEN CASH & CARRY SDN BHD on 21 … | 18.30 | 18.30 | **TP** |
| 89 | How much was the receipt from KAISON FURNISHING SDN BHD on 25 March 2… | 18.70 | 18.70 | **TP** |
| 90 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 26 Ma… | $8.20 | 8.20 | **TP** |
| 91 | How much was the receipt from RESTORAN HWA MUI SUTERA SDN BHD on 29 A… | 91.50 | 91.50 | **TP** |
| 92 | How much was the receipt from GUDANG HASIL RESTAURANT SDN BHD on 02 D… | RM20.00 | 15.80 | **FP** |
| 93 | How much was the receipt from BIRU PETROLEUM on 11 November 2016? | 6.40 | 6.40 | **TP** |
| 94 | How much was the receipt from UNIHAKKA INTERNATIONAL SDN BHD on 12 Ma… | $6.60 | 6.60 | **TP** |
| 95 | How much was the receipt from RESTORAN WAN SHE on 29 September 2017? | 164.40 | 164.40 | **TP** |
| 96 | How much was the receipt from TRIPLE SIX POINT ENTERPRISE 666 on 23 A… | 157.90 | 157.90 | **TP** |
| 97 | How much was the receipt from CONTENTO on 20 March 2018? | NOT_FOUND | 21.60 | **FN** |
| 98 | How much was the receipt from CHA FOR TEA on 19 April 2018? | 46.25 | 46.25 | **TP** |

**98 questions — TP 89 · FP 3 · FN 6 · TN n/a** &nbsp; accuracy 0.908 · precision of given answers 0.967 · answer rate 0.939

## Lookup · receipt date  (`date`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | When was the MR. D.I.Y. (M) SDN BHD receipt for 6.20 issued? | 23-03-18 | 23/03/2018 | **TP** |
| 2 | When was the BENS INDEPENDENT GROCER SDN. BHD receipt for 133.70 issu… | NOT_FOUND (The total amount on th… | 09/03/2018 | **FN** |
| 3 | When was the DREAM ARTS receipt for 15.00 issued? | 29/03/2018 07:35:38 PM | 29/03/2018 | **FP** |
| 4 | When was the SWC ENTERPRISE SDN BHD receipt for 4.00 issued? | 06/03/2018 | 06/03/2018 | **TP** |
| 5 | When was the SUPER TERMINAL SDN BHD receipt for 132.50 issued? | 20/01/2018 | 20/01/2018 | **TP** |
| 6 | When was the SYARIKAT PERNIAGAAN GIN KEE receipt for 11.13 issued? | 15/01/2018 | 15/01/2018 | **TP** |
| 7 | When was the UNIHAKKA INTERNATIONAL SDN BHD receipt for 8.90 issued? | 14 Apr 2018 | 14/04/2018 | **TP** |
| 8 | When was the GERBANG ALAF RESTAURANTS SDN BHD receipt for 19.00 issue… | 30/05/2018 | 30/05/2018 | **TP** |
| 9 | When was the PERNIAGAAN ZHENG HUI receipt for 436.20 issued? | 09/02/2018 | 09/02/2018 | **TP** |
| 10 | When was the UNIPHARM PHARMACY receipt for 24.90 issued? | 05/Apr/2018 12:49:11 pm | 05/04/2018 | **FP** |
| 11 | When was the OLIVE9 PHARMACY SDN BHD receipt for 12.15 issued? | 31/03/2017 | 31/03/2017 | **TP** |
| 12 | When was the MR. D.I.Y. (M) SDN BHD receipt for 59.20 issued? | 10-01-16 | 10/01/2016 | **TP** |
| 13 | When was the MR. D.I.Y. (KUCHAI) SDN BHD receipt for 26.40 issued? | 01-02-16 | 01/02/2016 | **TP** |
| 14 | When was the BENS INDEPENDENT GROCER SDN. BHD receipt for 81.00 issue… | 08/03/18 | 08/03/2018 | **TP** |
| 15 | When was the 99 SPEED MART S/B receipt for 98.90 issued? | 28-01-18 | 28/01/2018 | **TP** |
| 16 | When was the SYARIKAT PERNIAGAAN GIN KEE receipt for 190.80 issued? | 25/01/2018 | 25/01/2018 | **TP** |
| 17 | When was the KEDAI UBAT & RUNCIT HONG NING SDN. BHD. receipt for 676.… | 02/02/16 | 02/02/2016 | **TP** |
| 18 | When was the UNIHAKKA INTERNATIONAL SDN BHD receipt for 7.80 issued? | 11 May 2018 18:54 | 11/05/2018 | **TP** |
| 19 | When was the GARDENIA BAKERIES (KL) SDN BHD receipt for 55.14 issued? | 20/08/2017 | 20/08/2017 | **TP** |
| 20 | When was the UNIHAKKA INTERNATIONAL SDN BHD receipt for 6.60 issued? | 20 Mar 2018 17:55 | 20/03/2018 | **TP** |
| 21 | When was the MODERN DEPOT SDN BHD receipt for 150.00 issued? | 04/12/2016 | 04/12/2016 | **TP** |
| 22 | When was the ROYALTEA receipt for 13.10 issued? | 02/06/2018 #1 2:57 PM | 02/06/2018 | **FP** |
| 23 | When was the RESTORAN WAN SHENG receipt for 2.30 issued? | 27-06-2018 12:15:15 | 27/06/2018 | **TP** |
| 24 | When was the GOLDEN ARCHES RESTAURANTS SDN BHD receipt for 14.70 issu… | 03/12/2016 | 03/12/2016 | **TP** |
| 25 | When was the AIK HUAT HARDWARE ENTERPRISE (SETIA ALAM) SDN BHD receip… | 22/09/2017 | 22/09/2017 | **TP** |
| 26 | When was the LA STATIONERY SDN BHD receipt for 56.80 issued? | 09/02/2018 09:21:19 AM | 09/02/2018 | **FP** |
| 27 | When was the KING'S CONFECTIONERY S/B receipt for 9.90 issued? | 07/02/17 | 07/02/2017 | **TP** |
| 28 | When was the POPULAR BOOK CO. (M) SDN BHD receipt for 30.50 issued? | 05/03/18 | 05/03/2018 | **TP** |
| 29 | When was the UNIHAKKA INTERNATIONAL SDN BHD receipt for 7.30 issued? | 06 May 2018 | 06/05/2018 | **TP** |
| 30 | When was the UNIHAKKA INTERNATIONAL SDN BHD receipt for 8.20 issued? | There are multiple receipts for U… | 02/05/2018 | **FP** |
| 31 | When was the CHEF HENRY RIBS HOUSE receipt for 103.85 issued? | 05/06/2018 | 05/06/2018 | **TP** |
| 32 | When was the SANYU STATIONERY SHOP receipt for 8.70 issued? | 27/03/2017 | 14/10/2017 | **FP** |
| 33 | When was the RESTORAN HOETIM receipt for 83.00 issued? | 28/05/2018 | 28/05/2018 | **TP** |
| 34 | When was the F&P PHARMACY receipt for 31.90 issued? | 02/03/2018 | 02/03/2018 | **TP** |

**34 questions — TP 27 · FP 6 · FN 1 · TN n/a** &nbsp; accuracy 0.794 · precision of given answers 0.818 · answer rate 0.971

## Lookup · vendor address  (`address`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | What is the address on the SANYU STATIONERY SHOP receipt from 24 Octo… | NO. 31G&33G, JALAN SETIA INDAH X … | NO. 31G&33G, JALAN SETIA INDAH X … | **TP** |
| 2 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 3 | What is the address on the KOREAN DINE SDN BHD receipt from 19 June 2… | No 4, Jalan Permas 10/5, Bandar B… | NO 4, JALAN PERMAS 10/5. BANDAR B… | **TP** |
| 4 | What is the address on the UNIHAKKA INTERNATIONAL SDN BHD receipt fro… | 12, Jalan Tampoi 7/4,Kawasan Peri… | 12, JALAN TAMPOI 7/4,KAWASAN PERI… | **TP** |
| 5 | What is the address on the GUARDIAN HEALTH AND BEAUTY SDN BHD receipt… | 631 & G32, Grd Flr, 101 Mall, Ban… | G31 & G32, GRD FLR, IOI MALL BAND… | **FP** |
| 6 | What is the address on the UNIHAKKA INTERNATIONAL SDN BHD receipt fro… | 2, Jalan Tampoi 7/4,Kawasan Perin… | 2, JALAN TAMPOI 7/4,KAWASAN PERIN… | **TP** |
| 7 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 8 | What is the address on the KOH SENG HARDWARE receipt from 04 December… | NO. 939, BATU LIMA, JALAN IPOH, 5… | NO. 939, BATU LIMA, JALAN IPOH, 5… | **TP** |
| 9 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 10 | What is the address on the BHPETROL PERMAS JAYA 2 receipt from 08 May… | LOT PTD 101051 Jalan Permas 10/10… | LOT PTD 101051 JALAN PERMAS 10/10… | **TP** |
| 11 | What is the address on the 99 SPEED MART S/B receipt from 26 March 20… | LOT P.T. 2811. JALAN ANBSA, TAMAN… | LOT P.T. 2811, JALAN ANGSA, TAMAN… | **FP** |
| 12 | What is the address on the FIVE STAR CASH & CARRY receipt from 05 Jan… | G.23 & G.22, Plaza Seri Setia, No… | G.23 & G.22, PLAZA SERI SETIA, NO… | **TP** |
| 13 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 14 | What is the address on the ANEKA INTERTRADE MARKETING SDN BHD receipt… | 150B, Jalan TUDM, Kg Baru Subang,… | 150B, JALAN TUDM,KG BARU SUBANG 4… | **TP** |
| 15 | What is the address on the 99 SPEED MART S/B receipt from 19 October … | LOT P.T. 33198, BATU 4, JALAN KAP… | LOT P.T. 33198, BATU 4 JALAN KAPA… | **TP** |
| 16 | What is the address on the UNIHAKKA INTERNATIONAL SDN BHD receipt fro… | 12, Jalan Tampoi 7/4,1 Kawasan Pe… | 12, JALAN TAMPOI 7/4,KAWASAN PERI… | **FP** |
| 17 | What is the address on the SEN LEE HEONG RESTAURANT receipt from 12 J… | G-O-1,M.AVENUE NO.1,JLN 1/38A, SE… | G-0-1 ,M.AVENUE NO.1 ,JLN 1/38A, … | **FP** |
| 18 | What is the address on the THE TOAST F&B SDN BHD receipt from 19 May … | LOT 211082111 JALAN PERMAS UTARA,… | LOT 2110&2111 JALAN PERMAS UTARA … | **FP** |
| 19 | What is the address on the 118 MJ MOOKATA HOUSE receipt from 11 June … | NO.7G, Jalan Permas 11, Bandar Ba… | NO.7G,JALAN PERMAS 11, BANDAR BAR… | **TP** |
| 20 | What is the address on the AEON CO. (M) BHD receipt from 12 April 201… | 3RD FLR, AEON TAMAN MALURI SC, JL… | 3RD FLR, AEON TAMAN MALURI SC JLN… | **TP** |
| 21 | What is the address on the GERBANG ALAF RESTAURANTS SDN BHD receipt f… | No.3, Jalan SS21/39,47400 Petalin… | LEVEL 6, BANGUNAN TH, DAMANSARA U… | **TP** |
| 22 | What is the address on the CPI ROCKU SDN. BHD. receipt from 19 Februa… | LOT F355/356/357, FIRST FLOOR, RA… | LOT F355/356/357, FIRST FLOOR, RA… | **TP** |
| 23 | What is the address on the RESTAURANT JIAWEI JIAWEI HOUSE receipt fro… | 13, JLN TASIK UTAMA 8, MEDAN NIAG… | 13, JLN TASIK UTAMA 8 MEDAN NIAGA… | **TP** |
| 24 | What is the address on the ECONSAVE CASH & CARRY (FC) S/B receipt fro… | Lot G01, KL Festival City, No. 67… | LOT GO1, KL FESTIVAL CITY, NO. 67… | **FP** |
| 25 | What is the address on the GERBANG ALAF RESTAURANTS SDN BHD receipt f… | Level 6, Bangunan TH, Damansara U… | LEVEL 6, BANGUNAN TH, DAMANSARA U… | **TP** |
| 26 | What is the address on the SUPER NINETY NINE SDN.BHD. receipt from 31… | No.3343, Ground Floor, Jalan 18/3… | NO.3343, GROUND FLOOR, JALAN 18/3… | **TP** |
| 27 | What is the address on the HON HWA HARDWARE TRADING receipt from 11 F… | NO 37, JALAN MANIS 7. TAMAN SEGAR… | NO 37, JALAN MANIS 7, TAMAN SEGAR… | **TP** |
| 28 | What is the address on the SKL DAMANSARA ENTERPRISE receipt from 28 J… | 69G, JALAN SS21/60, DAMANSARA UTA… | 69G, JALAN SS21/60, DAMANSARA UTA… | **TP** |
| 29 | What is the address on the SANYU STATIONERY SHOP receipt from 23 Nove… | NO. 31G&33G, JALAN SETIA INDAH X … | NO. 31G&33G, JALAN SETIA INDAH X … | **TP** |
| 30 | What is the address on the EXQUISITE GARDEN SDN BHD receipt from 03 J… | LOT NO R134, Giant Hypermarket Pl… | LOT NO R134,GIANT HYPERMARKET PLE… | **FP** |
| 31 | What is the address on the RESTORAN WAN SHENG receipt from 17 March 2… | No.2, Jalan Temenggung 19/9, Seks… | NO.2, JALAN TEMENGGUNG 19/9, SEKS… | **TP** |
| 32 | What is the address on the KOH SENG HARDWARE receipt from 01 February… | NO. 939, BATU LIMA, JALAN IPOH, 5… | NO. 939, BATU LIMA, JALAN IPOH, 5… | **TP** |
| 33 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 34 | What is the address on the YIN MA (M) SDN.BHD. receipt from 31 Januar… | NO.2, JALAN UDANG SIAR 2, TAMAN S… | NO.2, JALAN UDANG SIAR 2, TAMAN S… | **TP** |
| 35 | What is the address on the UROKO JAPANESE CUISINE SDN BHD receipt fro… | 22A-1, JALAN 17/54, SECTION 17, 4… | 22A-1, JALAN 17/54, SECTION 17, 4… | **TP** |
| 36 | What is the address on the SWC ENTERPRISE SDN BHD receipt from 02 Jan… | 8 & 10, Jalan Ijok Permai 1, Pusa… | 8 & 10, JALAN IJOK PERMAI 1, PUSA… | **TP** |
| 37 | What is the address on the VIVOPAC MARKETING SDN BHD receipt from 17 … | 14 JALAN MANIS 4 TAMAN SEGAR 5610… | 14 JALAN MANIS 4 TAMAN SEGAR 5610… | **TP** |
| 38 | What is the address on the MOONLIGHT CAKE HOUSE SDN BHD receipt from … | No.1, Jalan Permas 10/5, Bandar B… | NO.1, JALAN PERMAS 10/5, BANDAR B… | **TP** |
| 39 | What is the address on the DEWINA HOST SDN BHD receipt from 27 March … | NOT_FOUND | LOT SATMZ 23, MEZZANINE LEVEL SAT… | **FN** |
| 40 | What is the address on the MR. D.I.Y. (M) SDN BHD receipt from 06 May… | LOT 1851 -A & 1851-B, JALAN KPB 6… | LOT 1851-A & 1851-B, JALAN KPB 6,… | **TP** |
| 41 | What is the address on the CROSS CHANNEL NETWORK SDN. BHD. receipt fr… | 47, JALAN MERANTI 1, SEK. 3, BAND… | 47, JALAN MERANTI 1, SEK. 3, BAND… | **TP** |
| 42 | What is the address on the MR. D.I.Y. (M) SDN BHD receipt from 21 Mar… | LOT 1851-A & 1851-B, JALAN KPB 6,… | LOT 1851-A & 1851-B, JALAN KPB 6,… | **TP** |
| 43 | What is the address on the NSK TRADE CITY-SELAYANG receipt from 27 Oc… | LOT 4674 & 4675 SELAYANG BATU 8, … | LOT 4674 & 4675 SELAYANG BATU 8 J… | **TP** |
| 44 | What is the address on the KEDAI UBAT & RUNCIT HONG NING SDN. BHD. re… | NO.8, JALAN LANG KUNING, KEPONG B… | NO.8,JALAN LANG KUNING, KEPONG BA… | **TP** |
| 45 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 46 | What is the address on the SYARIKAT KAM LAI SEONG SDN BHD receipt fro… | NO. 1442, JALAN SK 11/5, SERI KEM… | NO. 1442, JALAN SK 11/5, SERI KEM… | **TP** |
| 47 | What is the address on the KEDAI BUKU NEW ACHEIVERS receipt from 28 D… | NO. 12 & 14, JALAN JINJANG 27/54 … | NO. 12 & 14, JALAN JINJANG 27/54 … | **TP** |
| 48 | What is the address on the GUARDIAN HEALTH AND BEAUTY SDN BHD receipt… | NO.8 & 10, GF SOLARIS MOUNT KUARA… | SOLARIS MOUNT KIARA NO.8 & 10, GF… | **TP** |
| 49 | What is the address on the FLORISM DE ART receipt from 14 January 201… | LOT 2.70.00, LEVEL 2, PAVILION KU… | LOT 2.70.00,LEVEL 2, PAVILION KUA… | **TP** |
| 50 | What is the address on the GREAT ZONE HOUSEHOLD CENTRE SDN BHD receip… | 60 & 62, Jalan Ciku, 86000, KLUANG | 60 & 62. JALAN CIKU, 86000,KLUANG | **TP** |
| 51 | What is the address on the MAXINCOME RESOURCES SDN BHD receipt from 1… | No 16A, Jalan Astaka U8/83, Bukit… | NO 16A, JALAN ASTAKA U8/83, BUKIT… | **TP** |
| 52 | What is the address on the 99 SPEED MART S/B receipt from 20 Septembe… | LOT P.T. 33198, BATU 4, JALAN KAP… | LOT P.T. 33198, BATU 4 JALAN KAPA… | **TP** |
| 53 | What is the address on the PREMIO STATIONERY SDN BHD receipt from 20 … | No 57, Jalan SS 3/29, 47300 Petal… | NO 57, JALAN SS 3/29, 47300 PETAL… | **TP** |
| 54 | What is the address on the UNIHAKKA INTERNATIONAL SDN BHD receipt fro… | 12, Jalan Tampoi 7/4,Kawasan Peri… | 12, JALAN TAMPOI 7/4,KAWASAN PERI… | **TP** |
| 55 | What is the address on the AEON CO. (M) BHD receipt from 09 July 2017? | 3RD FLR, AEON TAMAN MALURI SC, JL… | 3RD FLR, AEON TAMAN MALURI SC JLN… | **TP** |
| 56 | What is the address on the EVERGREEN LIGHT SDN BHD receipt from 10 Ju… | NO.7-1, JALAN PUTERI 7/11, BANDAR… | NO.7-1, JALAN PUTERI 7/11, BANDAR… | **TP** |
| 57 | What is the address on the POTTERS GARDEN SDN BHD receipt from 04 Jan… | Batu 11. Sg Buloh, 47000 Selangor. | BATU 11 . SG BULOH , 47000 SELANG… | **TP** |
| 58 | What is the address on the KEDAI PAPAN YEW CHUAN receipt from 16 Marc… | LOT 276 JALAN BANTING 43800 DENGK… | LOT 276 JALAN BANTING 43800 DENGK… | **TP** |
| 59 | What is the address on the UNIHAKKA INTERNATIONAL SDN BHD receipt fro… | 12, Jalan Tampoi 7/4,Kawasan Peri… | 12, JALAN TAMPOI 7/4,KAWASAN PERI… | **TP** |
| 60 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 61 | What is the address on the 99 SPEED MART S/B receipt from 10 March 20… | LOT P.T. 2811, JALAN ANGSA, TAMAN… | LOT P.T. 2811, JALAN ANGSA, TAMAN… | **TP** |
| 62 | What is the address on the STAR GROCER SDN BHD receipt from 25 March … | No 4, Desa Pandan, Off kampong Pa… | NO 4, DESA PANDAN, OFF KAMPONG PA… | **TP** |
| 63 | What is the address on the OJC MARKETING SDN BHD receipt from 15 Janu… | NO 2 & 4, JALAN BAYU 4, BANDAR SE… | NO 2 & 4, JALAN BAYU 4, BANDAR SE… | **FP** |
| 64 | What is the address on the SUPER SEVEN CASH & CARRY SDN BHD receipt f… | NO. 1 Jalan Euro 1, Off Jalan Bat… | NO. 1 JALAN EURO 1 OFF JALAN BATU… | **TP** |
| 65 | What is the address on the MOONLIGHT CAKE HOUSE SDN BHD receipt from … | No.1, Jalan Permas 10/5, Bandar B… | NO.1, JALAN PERMAS 10/5, BANDAR B… | **TP** |
| 66 | What is the address on the RESTORAN WAN SHENG receipt from 11 May 201… | No.2, Jalan Temenggung 19/9, Seks… | NO.2, JALAN TEMENGGUNG 19/9, SEKS… | **TP** |
| 67 | What is the address on the GARDENIA BAKERIES (KL) SDN BHD receipt fro… | Lot 3, Jalan Pelabur 23/1, 40300 … | LOT 3, JALAN PELABUR 23/1, 40300 … | **TP** |
| 68 | What is the address on the AEON CO. (M) BHD receipt from 17 June 2018? | 3RD FLR, AEON TAMAN MALURI SC, JL… | 3RD FLR, AEON TAMAN MALURI SC JLN… | **TP** |

**68 questions — TP 59 · FP 8 · FN 1 · TN n/a** &nbsp; accuracy 0.868 · precision of given answers 0.881 · answer rate 0.985

## Aggregate · total spend at a vendor  (`sum_total`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | How much did I spend at GERBANG ALAF RESTAURANTS SDN BHD in total? | 262.20 | 262.20 | **TP** |
| 2 | How much did I spend at SWC ENTERPRISE SDN BHD in total? | 16.20 | 16.20 | **TP** |
| 3 | How much did I spend at 99 SPEED MART S/B in total? | 779.15 | 779.15 | **TP** |
| 4 | How much did I spend at UNIHAKKA INTERNATIONAL SDN BHD in total? | 228.90 | 228.90 | **TP** |
| 5 | How much did I spend at PASARAYA BORONG PINTAR SDN BHD in total? | 15.35 | 15.35 | **TP** |
| 6 | How much did I spend at POPULAR BOOK CO. (M) SDN BHD in total? | 86.35 | 227.00 | **FP** |
| 7 | How much did I spend at HON HWA HARDWARE TRADING in total? | 42.20 | 42.20 | **TP** |
| 8 | How much did I spend at AEON CO. (M) BHD in total? | 723.10 | 870.55 | **FP** |
| 9 | How much did I spend at LIM SENG THO HARDWARE TRADING in total? | 25.90 | 25.90 | **TP** |
| 10 | How much did I spend at SYARIKAT PERNIAGAAN GIN KEE in total? | 822.35 | 822.35 | **TP** |
| 11 | How much did I spend at WESTERN EASTERN STATIONERY SDN. BHD in total? | 57.72 | 57.72 | **TP** |
| 12 | How much did I spend at RESTORAN WAN SHENG in total? | 92.30 | 92.30 | **TP** |
| 13 | How much did I spend at KEDAI PAPAN YEW CHUAN in total? | 1391.25 | 1391.25 | **TP** |
| 14 | How much did I spend at YONG CEN ENTERPRISE in total? | 257.50 | 258.50 | **TP** |
| 15 | How much did I spend at SUPER SEVEN CASH & CARRY SDN BHD in total? | 119.40 | 527.85 | **FP** |
| 16 | How much did I spend at SEGI CASH & CARRY SDN. BHD. in total? | 929.65 | 929.65 | **TP** |
| 17 | How much did I spend at GUARDIAN HEALTH AND BEAUTY SDN BHD in total? | 140.61 | 140.61 | **TP** |
| 18 | How much did I spend at MR. D.I.Y. (M) SDN BHD in total? | 451.60 | 363.80 | **FP** |
| 19 | How much did I spend at PRINT EXPERT SDN BHD in total? | 550.35 | 550.35 | **TP** |
| 20 | How much did I spend at KING'S CONFECTIONERY S/B in total? | 114.60 | 114.60 | **TP** |
| 21 | How much did I spend at BEMED (SP) SDN. BHD. in total? | 841.70 | 841.70 | **TP** |
| 22 | How much did I spend at GARDENIA BAKERIES (KL) SDN BHD in total? | 1104.55 | 1104.55 | **TP** |
| 23 | How much did I spend at YIN MA (M) SDN.BHD. in total? | 52.60 | 52.60 | **TP** |
| 24 | How much did I spend at KEDAI UBAT & RUNCIT HONG NING SDN. BHD. in to… | 738.65 | 738.65 | **TP** |
| 25 | How much did I spend at MR. D.I.Y. (KUCHAI) SDN BHD in total? | 451.60 | 87.80 | **FP** |
| 26 | How much did I spend at SANYU STATIONERY SHOP in total? | 99.50 | 99.50 | **TP** |
| 27 | How much did I spend at MOONLIGHT CAKE HOUSE SDN BHD in total? | 59.40 | 59.40 | **TP** |

**27 questions — TP 22 · FP 5 · FN 0 · TN n/a** &nbsp; accuracy 0.815 · precision of given answers 0.815 · answer rate 1.000

## Aggregate · receipt count at a vendor  (`count`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | How many receipts do I have from GERBANG ALAF RESTAURANTS SDN BHD? | 7 | 7 | **TP** |
| 2 | How many receipts do I have from SWC ENTERPRISE SDN BHD? | 3 | 3 | **TP** |
| 3 | How many receipts do I have from 99 SPEED MART S/B? | 13 | 13 | **TP** |
| 4 | How many receipts do I have from UNIHAKKA INTERNATIONAL SDN BHD? | 28 | 28 | **TP** |
| 5 | How many receipts do I have from PASARAYA BORONG PINTAR SDN BHD? | 5 | 5 | **TP** |
| 6 | How many receipts do I have from POPULAR BOOK CO. (M) SDN BHD? | 5 | 6 | **FP** |
| 7 | How many receipts do I have from HON HWA HARDWARE TRADING? | 4 | 4 | **TP** |
| 8 | How many receipts do I have from AEON CO. (M) BHD? | 8 | 9 | **FP** |
| 9 | How many receipts do I have from LIM SENG THO HARDWARE TRADING? | 3 | 3 | **TP** |
| 10 | How many receipts do I have from SYARIKAT PERNIAGAAN GIN KEE? | 11 | 11 | **TP** |
| 11 | How many receipts do I have from WESTERN EASTERN STATIONERY SDN. BHD? | 3 | 3 | **TP** |
| 12 | How many receipts do I have from RESTORAN WAN SHENG? | 14 | 14 | **TP** |
| 13 | How many receipts do I have from KEDAI PAPAN YEW CHUAN? | 8 | 8 | **TP** |
| 14 | How many receipts do I have from YONG CEN ENTERPRISE? | 4 | 4 | **TP** |
| 15 | How many receipts do I have from SUPER SEVEN CASH & CARRY SDN BHD? | 3 | 4 | **FP** |
| 16 | How many receipts do I have from SEGI CASH & CARRY SDN. BHD.? | 4 | 4 | **TP** |
| 17 | How many receipts do I have from GUARDIAN HEALTH AND BEAUTY SDN BHD? | 3 | 3 | **TP** |
| 18 | How many receipts do I have from MR. D.I.Y. (M) SDN BHD? | 17 | 12 | **FP** |
| 19 | How many receipts do I have from PRINT EXPERT SDN BHD? | 3 | 3 | **TP** |
| 20 | How many receipts do I have from KING'S CONFECTIONERY S/B? | 4 | 4 | **TP** |
| 21 | How many receipts do I have from BEMED (SP) SDN. BHD.? | 4 | 4 | **TP** |
| 22 | How many receipts do I have from GARDENIA BAKERIES (KL) SDN BHD? | 31 | 31 | **TP** |
| 23 | How many receipts do I have from YIN MA (M) SDN.BHD.? | 3 | 3 | **TP** |
| 24 | How many receipts do I have from KEDAI UBAT & RUNCIT HONG NING SDN. B… | 4 | 4 | **TP** |
| 25 | How many receipts do I have from MR. D.I.Y. (KUCHAI) SDN BHD? | 17 | 5 | **FP** |
| 26 | How many receipts do I have from SANYU STATIONERY SHOP? | 13 | 13 | **TP** |
| 27 | How many receipts do I have from MOONLIGHT CAKE HOUSE SDN BHD? | 3 | 3 | **TP** |

**27 questions — TP 22 · FP 5 · FN 0 · TN n/a** &nbsp; accuracy 0.815 · precision of given answers 0.815 · answer rate 1.000

## Aggregate · largest purchase at a vendor  (`max_total`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | What was my largest single purchase at GERBANG ALAF RESTAURANTS SDN B… | 109.05 | 109.05 | **TP** |
| 2 | What was my largest single purchase at SWC ENTERPRISE SDN BHD? | 8.00 | 8.00 | **TP** |
| 3 | What was my largest single purchase at 99 SPEED MART S/B? | 262.20 | 262.20 | **TP** |
| 4 | What was my largest single purchase at UNIHAKKA INTERNATIONAL SDN BHD? | 12.20 | 12.20 | **TP** |
| 5 | What was my largest single purchase at PASARAYA BORONG PINTAR SDN BHD? | 10.40 | 10.40 | **TP** |
| 6 | What was my largest single purchase at POPULAR BOOK CO. (M) SDN BHD? | 30.70 | 140.65 | **FP** |
| 7 | What was my largest single purchase at HON HWA HARDWARE TRADING? | 19.60 | 19.60 | **TP** |
| 8 | What was my largest single purchase at AEON CO. (M) BHD? | 458.55 | 458.55 | **TP** |
| 9 | What was my largest single purchase at LIM SENG THO HARDWARE TRADING? | 10.50 | 10.50 | **TP** |
| 10 | What was my largest single purchase at SYARIKAT PERNIAGAAN GIN KEE? | 190.80 | 190.80 | **TP** |
| 11 | What was my largest single purchase at WESTERN EASTERN STATIONERY SDN… | 42.88 | 42.88 | **TP** |
| 12 | What was my largest single purchase at RESTORAN WAN SHENG? | 17.60 | 17.60 | **TP** |
| 13 | What was my largest single purchase at KEDAI PAPAN YEW CHUAN? | 312.70 | 312.70 | **TP** |
| 14 | What was my largest single purchase at YONG CEN ENTERPRISE? | 108.00 | 108.00 | **TP** |
| 15 | What was my largest single purchase at SUPER SEVEN CASH & CARRY SDN B… | 59.00 | 408.45 | **FP** |
| 16 | What was my largest single purchase at SEGI CASH & CARRY SDN. BHD.? | 674.00 | 674.00 | **TP** |
| 17 | What was my largest single purchase at GUARDIAN HEALTH AND BEAUTY SDN… | 108.21 | 108.21 | **TP** |
| 18 | What was my largest single purchase at MR. D.I.Y. (M) SDN BHD? | 96.90 | 96.90 | **TP** |
| 19 | What was my largest single purchase at PRINT EXPERT SDN BHD? | 226.60 | 226.60 | **TP** |
| 20 | What was my largest single purchase at KING'S CONFECTIONERY S/B? | 70.00 | 70.00 | **TP** |
| 21 | What was my largest single purchase at BEMED (SP) SDN. BHD.? | 308.70 | 308.70 | **TP** |
| 22 | What was my largest single purchase at GARDENIA BAKERIES (KL) SDN BHD? | 84.78 | 84.78 | **TP** |
| 23 | What was my largest single purchase at YIN MA (M) SDN.BHD.? | 32.70 | 32.70 | **TP** |
| 24 | What was my largest single purchase at KEDAI UBAT & RUNCIT HONG NING … | 676.00 | 676.00 | **TP** |
| 25 | What was my largest single purchase at MR. D.I.Y. (KUCHAI) SDN BHD? | 96.90 | 34.40 | **FP** |
| 26 | What was my largest single purchase at SANYU STATIONERY SHOP? | 21.90 | 21.90 | **TP** |
| 27 | What was my largest single purchase at MOONLIGHT CAKE HOUSE SDN BHD? | 28.20 | 28.20 | **TP** |

**27 questions — TP 24 · FP 3 · FN 0 · TN n/a** &nbsp; accuracy 0.889 · precision of given answers 0.889 · answer rate 1.000

## Aggregate · total spend in a year  (`sum_year`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | How much did I spend in total during 2016? | 1868.39 | 2267.75 | **FP** |
| 2 | How much did I spend in total during 2017? | 6058.01 | 6199.66 | **FP** |
| 3 | How much did I spend in total during 2018? | 11913.94 | 12500.91 | **FP** |

**3 questions — TP 0 · FP 3 · FN 0 · TN n/a** &nbsp; accuracy 0.000 · precision of given answers 0.000 · answer rate 1.000

## Aggregate · receipt count in a year  (`count_year`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | How many receipts do I have from 2016? | 30 | 31 | **FP** |
| 2 | How many receipts do I have from 2017? | 104 | 105 | **FP** |
| 3 | How many receipts do I have from 2018? | 200 | 203 | **FP** |

**3 questions — TP 0 · FP 3 · FN 0 · TN n/a** &nbsp; accuracy 0.000 · precision of given answers 0.000 · answer rate 1.000

## Part 1 summary

| | TP | FP | FN | TN | accuracy | precision | answer rate |
|---|---|---|---|---|---|---|---|
| all 287 | 243 | 36 | 8 | n/a | 0.847 | 0.871 | 0.972 |

---

# Part 2 — Multi-step questions (36), agent arm

The router scores 0.028 on this set (35/36 wrong): it issues one query per question and cannot express a comparison. Worse than its refusals, on three comparison questions it returned a bare single-vendor total as if it answered the question — a silent FP. The agent chains 2–3 tool calls.

## Compare two vendors  (`compare`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | Did I spend more at MR. D.I.Y. (M) or RESTORAN WAN SHENG? | MR. D.I.Y. (M) | MR. D.I.Y. (M) | **TP** |
| 2 | Did I spend more at YONG CEN ENTERPRISE or MOONLIGHT CAKE HOUSE? | YONG CEN ENTERPRISE | YONG CEN ENTERPRISE | **TP** |
| 3 | Did I spend more at WESTERN EASTERN STATIONERY or LIM SENG THO HARDWA… | WESTERN EASTERN STATIONERY | WESTERN EASTERN STATIONERY | **TP** |
| 4 | Did I spend more at PRINT EXPERT or SANYU STATIONERY SHOP? | PRINT EXPERT | PRINT EXPERT | **TP** |
| 5 | Did I spend more at AEON CO or Gerbang Alaf Restaurants? | AEON CO | AEON CO | **TP** |
| 6 | Did I spend more at WESTERN EASTERN STATIONERY or HON HWA HARDWARE TR… | WESTERN EASTERN STATIONERY | WESTERN EASTERN STATIONERY | **TP** |
| 7 | Did I spend more at POPULAR BOOK CO. (M) or LIM SENG THO HARDWARE TRA… | POPULAR BOOK CO. (M) | POPULAR BOOK CO. (M) | **TP** |
| 8 | Did I spend more at KEDAI UBAT & RUNCIT HONG NING or UNIHAKKA INTERNA… | KEDAI UBAT & RUNCIT HONG NING | KEDAI UBAT & RUNCIT HONG NING | **TP** |
| 9 | Did I spend more at KEDAI UBAT & RUNCIT HONG NING or SANYU STATIONERY… | KEDAI UBAT & RUNCIT HONG NING | KEDAI UBAT & RUNCIT HONG NING | **TP** |
| 10 | Did I spend more at BEMED (SP) or PRINT EXPERT? | BEMED (SP) | BEMED (SP) | **TP** |
| 11 | Did I spend more at SYARIKAT PERNIAGAAN GIN KEE or WESTERN EASTERN ST… | SYARIKAT PERNIAGAAN GIN KEE | SYARIKAT PERNIAGAAN GIN KEE | **TP** |
| 12 | Did I spend more at MR. D.I.Y. (M) or YIN MA (M) SDN.BHD? | MR. D.I.Y. (M) | MR. D.I.Y. (M) | **TP** |
| 13 | Did I spend more at YONG CEN ENTERPRISE or SWC ENTERPRISE? | YONG CEN ENTERPRISE | YONG CEN ENTERPRISE | **TP** |
| 14 | Do I have more receipts from AEON CO or LIM SENG THO HARDWARE TRADING? | AEON CO | AEON CO | **TP** |
| 15 | Do I have more receipts from KING'S CONFECTIONERY or LIM SENG THO HAR… | KING'S CONFECTIONERY | KING'S CONFECTIONERY | **TP** |
| 16 | Do I have more receipts from GARDENIA BAKERIES (KL) or MOONLIGHT CAKE… | GARDENIA BAKERIES (KL) | GARDENIA BAKERIES (KL) | **TP** |
| 17 | Do I have more receipts from PRINT EXPERT or RESTORAN WAN SHENG? | RESTORAN WAN SHENG | RESTORAN WAN SHENG | **TP** |
| 18 | Do I have more receipts from PRINT EXPERT or MR. D.I.Y. (M)? | MR. D.I.Y. (M) | MR. D.I.Y. (M) | **TP** |

**18 questions — TP 18 · FP 0 · FN 0 · TN n/a** &nbsp; accuracy 1.000 · precision of given answers 1.000 · answer rate 1.000

## Rank vendors  (`rank`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | Which vendor did I spend the most with? | KEDAI PAPAN YEW CHUAN | KEDAI PAPAN YEW CHUAN | **TP** |
| 2 | Which vendor do I have the most receipts from? | GARDENIA BAKERIES (KL) SDN BHD | GARDENIA BAKERIES (KL) | **TP** |
| 3 | What are my top three vendors by total spend? | ['KEDAI PAPAN YEW CHUAN', 'GARDEN… | KEDAI PAPAN YEW CHUAN; GARDENIA B… | **TP** |

**3 questions — TP 3 · FP 0 · FN 0 · TN n/a** &nbsp; accuracy 1.000 · precision of given answers 1.000 · answer rate 1.000

## Extreme in a period  (`extreme`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | What was my biggest single purchase in 2017? | 848.0 | 848.00 | **TP** |
| 2 | Which vendor did I make my biggest 2017 purchase from? | KOH SENG HARDWARE | KOH SENG HARDWARE | **TP** |
| 3 | What was my biggest single purchase in 2018? | FLORISM DE ART | 750.00 | **FP** |
| 4 | Which vendor did I make my biggest 2018 purchase from? | FLORISM DE ART | FLORISM DE ART | **TP** |
| 5 | What is my average receipt value? | 60.08 | 60.08 | **TP** |

**5 questions — TP 4 · FP 1 · FN 0 · TN n/a** &nbsp; accuracy 0.800 · precision of given answers 0.800 · answer rate 1.000

## Chained lookup (find receipt → read field)  (`chain`)

| # | Question | Model predicted | Ground truth | Outcome |
|---|---|---|---|---|
| 1 | What is the address of the vendor I spent 181.55 with on 19/02/2018? | LOT F355/356/357,FIRST FLOOR, RAI… | LOT F355/356/357,FIRST FLOOR, RAI… | **TP** |
| 2 | What is the address of the vendor I spent 393.00 with on 22/08/2017? | NO.2, JALAN METRO PERDANA TIMUR 1… | NO.2, JALAN METRO PERDANA TIMUR 1… | **TP** |
| 3 | What is the address of the vendor I spent 436.20 with on 09/02/2018? | NO.59 JALAN PERMAS 9/6 BANDAR BAR… | NO.59 JALAN PERMAS 9/6 BANDAR BAR… | **TP** |
| 4 | What is the address of the vendor I spent 412.90 with on 01/05/2018? | KM 4, Jln Ampang, Hulu Langat, 68… | KM 4, Jln Ampang, Hulu Langat, 68… | **TP** |
| 5 | What is the address of the vendor I spent 170.00 with on 02/01/2019? | NO 2 & 4, JALAN BAYU 4, BANDAR SE… | NO 2 & 4, JALAN BAYU 4, BANDAR SE… | **TP** |
| 6 | What is the address of the vendor I spent 278.80 with on 20/11/2017? | NO: 28, JALAN ASTANA 1C, BANDAR B… | NO: 28, JALAN ASTANA 1C, BANDAR B… | **TP** |
| 7 | On what date did I make my purchase of 190.00 from SYARIKAT KAM LAI S… | 08/03/2018, NO. 1442, JALAN SK 11… | 08/03/2018; NO. 1442, JALAN SK 11… | **TP** |
| 8 | On what date did I make my purchase of 150.00 from MODERN DEPOT, and … | 04/12/2016, NO.19, PT18685, JALAN… | 04/12/2016; NO.19, PT18685, JALAN… | **TP** |
| 9 | On what date did I make my purchase of 133.70 from B.I.G.- Ben's Inde… | 09/03/2018, Lot 6, Jalan Batai, P… | 09/03/2018; Lot 6, Jalan Batai, P… | **TP** |
| 10 | On what date did I make my purchase of 308.70 from BEMED (SP), and wh… | 27/03/2018, NO.19,JALAI DIBAR G U… | 27/03/2018; NO.19,JALAI DIBAR G U… | **TP** |

**10 questions — TP 10 · FP 0 · FN 0 · TN n/a** &nbsp; accuracy 1.000 · precision of given answers 1.000 · answer rate 1.000

## Part 2 summary

| | TP | FP | FN | TN | accuracy | precision | answer rate |
|---|---|---|---|---|---|---|---|
| all 36 | 35 | 1 | 0 | n/a | 0.972 | 0.972 | 1.000 |

---

## Reading the two parts together

- Lookup fields sit at 0.88–0.93 with precision above 0.9: when a receipt is resolved by vendor+date, reading a field from it is reliable.
- Vendor aggregates carry most of the FPs. These are Layer-1 inheritance: a receipt whose vendor or total was extracted wrong shifts the SQL result, and the system has no way to know.
- Year aggregates (n=6) are all FP: sums over 200+ receipts compound every extraction error. The 2018 sum is within 5% of gold yet scores zero under exact match — relative error is the honest metric for this class.
- The agent answers nearly everything it attempts correctly (precision 0.971) and abstains rarely; its one FP is a genuine LLM error (asked for an amount, returned the vendor name).

*Generated by `layer3/full_analysis.py` from the scored run files; aggregate gold is defined over canonical vendor identities (see README, Limitations).*
