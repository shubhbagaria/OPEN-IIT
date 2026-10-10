# Real-Time Fake-PTP Detection & In-Call Guidance
CreditNirvana Problem Statement 1: end-to-end solution on the provided sandbox extracts.

All numbers below come from `outputs/results.json`, `outputs/latency_*.json` and `outputs/slm_eval.json`, which are written by the code in this repository. Re-run with `python run_all.py` (add `--slm` for the GPU track).

---

## 0. What was delivered

| Deliverable | Where |
|---|---|
| Modular, executable Python package | `ptp_copilot/` (one module per phase, listed in §8) |
| Enriched output CSV, one row per PTP (3,696): predicted category, calibrated confidence, keep-probability at 3 stages, explanation, next utterance, guardrails, parking plan | `outputs/enriched_ptp_predictions.csv` |
| Ground-truth tables | `outputs/outcome_labels.csv`, `outputs/ground_truth_labels.csv` |
| Classification report, confusion matrix, AUC, ECE, calibration, streaming curve, audits, business case | `outputs/results.json`, `outputs/pipeline_log.txt`, `outputs/figures/*.png` |
| Live-call replays, turn by turn | `outputs/streaming_demo.md` |
| Latency benchmarks | `outputs/latency_tabular.json` (CPU copilot), `outputs/latency_slm.json` (CUDA SLM) |
| Agent integrity view | `outputs/agent_integrity_dashboard.csv` |
| SLM (LoRA) adapter + merged weights, eval | `outputs/slm_lora/`, `outputs/slm_merged/`, `outputs/slm_eval.json` |
| Full-stack web app (FastAPI + React) on top of the above | `webapp/` (see §10); run `python webapp/start.py` |

### Headline results (held-out test split: 360 accounts, 563 PTPs)

| Question from the problem statement | Result |
|---|---|
| AUC of the keep prediction at end of call | **0.763** (test); **0.785** out-of-fold across all 3,696 PTPs |
| Calibration of the keep prediction | ECE **0.037** (test); **0.025** out-of-fold |
| Share of broken PTPs flagged before the account is parked | **88.4%** (precision of the flag 84%) |
| Hardship cases misclassified as escape/repeat (the costliest error) | **1.6%** with cost-sensitive decoding (12.3% with plain argmax), then **0** routed to aggressive treatment after guardrails |
| Six-way category, end of call | macro-F1 **0.75** (cost-sensitive) / **0.79** (argmax); one-vs-rest AUC 0.94; **0.78** accuracy on the 27 human-annotated PTPs in the test split |
| Deciding before the call ends | the "heard enough" trigger fires on 88% of calls, at a median 2/3 of the way through and about 7.6 s before hang-up, and agrees with the final call 96% of the time |
| Live update latency | tabular copilot **60 ms p50 / 78 ms p95** per turn on CPU. SLM (Qwen2.5-0.5B + LoRA, 4-bit): **TTFT 47 ms p50 / 57 ms p95 at 0.74 GB peak VRAM** on an RTX 3050 Laptop, so both the < 100 ms and < 1 GB SLAs are met |
| Third-party debt disclosure by the copilot | **0** of 342 third-party calls. The historical baseline is 78% of third-party calls (334/428), where the agent named an amount or PTP *after* a relative identified themself |

---

## 1. Data and what it allowed

| File | Rows | Use |
|---|---|---|
| `ptps.csv` | 3,696 PTPs (3,091 call, 605 field) | unit of prediction |
| `call_transcripts.csv` | 39,194 turns / 3,367 calls, Hinglish, Kanglish, English, with `pause_before_ms` and ASR confidence | conversation + acoustic signals |
| `payments.csv` | 2,162 payments, 7 channels | ground truth |
| `accounts.csv` | 2,400 accounts (salary day for 762, ability-to-pay for 1,673) | pre-call features |
| `dial_attempts.csv` | 51,105 attempts with agent remarks | dialler history, agent remarks |
| `post_call_events.csv` | link sent/opened, WhatsApp reply, inbound call | post-call stage |
| `agents.csv`, `lenders.csv` | 30 agents incl. 1 bot, 6 lenders | agent cohorts |
| `annotated_ptp_sample.csv` | 200 human-labelled PTPs (105 high / 80 medium / 15 low confidence) | gold labels |
| `splits.csv` | account-level train 70% / validation 15% / test 15% | all evaluation is account-disjoint |

The spec named five generic files. The real extract has ten. The mapping is direct; the extra files (events, annotations, splits) were all used.

---

## 2. Phase 1: Ground truth

### 2.1 "Kept": the business definition (`labels.outcome_labels`)

```
paid(t)  = Σ payments on the account, ALL channels (UPI, NACH, NEFT, BBPS, branch, field cash),
           from captured_ts up to t, and before the next PTP on the same account (a new promise supersedes)
kept     = paid(promised_date + 2 days grace, 23:59:59)  ≥ 0.90 × promised_amount
cured    = paid(promised_date + 14 days cure, 23:59:59)  ≥ 0.90 × promised_amount
status   = kept_on_time | kept_late (cured) | partial (≥25%) | broken
```

* **+2 days grace** absorbs NEFT/NACH clearing and weekend posting.
* **+14 days cure** keeps observation inside the 30-day billing cycle, so a payment from the next cycle is never credited to this promise.
* **90% threshold** absorbs ₹250–500 bounce charges and GST/rounding differences.
* **Supersession** stops one payment from being counted twice when an agent re-negotiates (226 PTPs; 7% of them were kept).
* **Censoring** is flagged: 0.03% of grace windows and 0.3% of cure windows run past the last observed payment (24 Jul).

| status | n | share |
|---|---|---|
| kept on time | 1,023 | 27.7% |
| kept late (cure window) | 260 | 7.0% |
| partial (25–90%) | 436 | 11.8% |
| broken | 1,977 | 53.5% |

The model target is `kept` (27.7%). `cured` (34.7%) is reported for recovery accounting. This matches the "more than half of PTPs break" premise.

### 2.2 Six behavioural categories (`labels.rule_labels`, `labels.bootstrap_categories`)

**Step 1: a transparent hindsight decision list (20 rules, R01–R20).** It may use the outcome, because ground truth is labelled after the fact. The in-call model never sees the outcome. Order encodes regulatory priority:

1. **Third party** (relation phrase in the transcript, or a non-self answerer) with a date → `third_party`. With no commitment at all → `agent_recorded_or_pushed`.
2. **Agent-pushed**: the PTP was recorded, the borrower never stated a date, and there was agent pressure (CIBIL/escalation/"last reminder") → `agent_recorded_or_pushed`. Payment data cannot separate this from escape, so only the conversation can.
3. Kept, or cured with firm language → `genuine_feasible`.
4. **Severe hardship** (medical, job loss, company shut), or hardship together with a partial offer or a condition → `genuine_infeasible`. This sits *before* the repeat and escape rules, so hardship is never overridden by history.
5. ≥2 prior broken promises (bureau-style history plus in-window history) and the topic was raised → `repeat_promiser`.
6. Evasion or exit phrases with no borrower-stated date → `escape`; the remaining rules are softer fall-backs.

Every PTP carries its `rule_id` in the output for audit.

**Step 2: confidence-gated self-training to 100%.** LightGBM is seeded with the 200 gold labels (weight 3 × annotator confidence) plus rule labels with confidence ≥0.75. Each round fits a temperature on out-of-fold gold predictions, then accepts pseudo-labels with calibrated confidence ≥0.85 that agree with the rules (or ≥0.95). Three rounds. Result: 200 human / 1,694 high-confidence pseudo / 1,802 low-confidence pseudo labels, each with `category_source` and `ground_truth_conf`, so downstream training down-weights the weak ones.

**Honest quality estimate.** With 5-fold cross-validation over the gold set (held-out folds never seen by the bootstrapper):

| label pipeline | accuracy vs human | macro-F1 |
|---|---|---|
| rules only | 0.67 | 0.69 |
| rules + self-training | 0.67 | 0.69 |

Self-training gives full coverage and calibrated confidences, but it does **not** improve agreement with annotators; the rules carry the signal. The annotations are noisy themselves: 47% are medium or low confidence, and e.g. 7 of 51 "genuine_infeasible" PTPs were paid on time. Agreement among annotators is the real ceiling, and CN should measure it (double-annotate 100 calls) before any further label work.

| category | n | share |
|---|---|---|
| genuine_feasible | 1,012 | 27% |
| repeat_promiser | 832 | 23% |
| genuine_infeasible | 708 | 19% |
| escape | 492 | 13% |
| third_party | 345 | 9% |
| agent_recorded_or_pushed | 307 | 8% |

---

## 3. Phase 2: Features (`features.py`, `lexicon.py`)

Every block is **time-aware**: nothing observed after a decision point enters the model used at that point.

| Block | Signals (examples) | Known at |
|---|---|---|
| **Conversation** (per-turn lexicon, then cumulative over the prefix) | hedging (`dekhta hoon`, `try karunga`, `nodtini`, `let me see`), exit (`baad mein call karo`, `meeting nalli idini`, `I have to go`), conditional (`salary aayi toh`, `party payment kottre`), vague dates (`agle hafte`, `mundina vaara`, `month end`), firm (`pakka`, `khud kar dunga`), partial offers, hardship / severe hardship, third-party identity (`main unka bhai`, `naanu avra amma`, `this is her mother`), link accept/refuse, prior-break excuses, minimal acks | every turn |
| **Attribution** | who first stated a date; `recorded_without_cust_date`; `recorded_without_cust_amount`; `recorded_without_commit`; customer replies to an agent date proposal and how long they paused before them; commitment within 2 turns of pressure; whether the customer stated the discussed day | every turn |
| **Acoustic / telephony** | customer mean and max pause, long hesitations (>1.2 s), ASR confidence mean/min, `[inaudible]` count, talk-to-listen ratio by words and by speaking time | every turn |
| **Financial / behavioural** | `is_salary_window` (promise 0–5 days after salary credit), `is_pre_salary_hazard` (1–4 days before); salary day stated in the call ("salary 7 tak aati hai") fills the 68% of accounts without one on file; amount vs overdue / EMI / ability-to-pay; bureau band; bounce reason; prior-PTP broken ratio (bureau + in-window, closed windows only); historical settlement latency; 30-day dial, answer, refusal and hardship dispositions; 30-day payments | pre-call |
| **Agent integrity** | the agent's keep rate on *earlier* PTPs whose grace window had closed, shrunk to the channel cohort; lift vs peers; share of PTPs at shift end; share of generic copy-pasted remarks ("PTP liya, customer cooperative"); PTPs per day | pre-call |
| **Post-call (≤36 h and before the due date)** | link opened and hours to open, WhatsApp yes/no, inbound call; agent-entered answerer and disposition (excluded from in-call so the score cannot be gamed by what the agent types) | parking decision |

**A leakage bug caught and fixed:** the first version used leave-one-out target encoding for the agent's keep rate. That leaks the label: within an agent, a kept PTP always gets a slightly lower rate than a broken one. Out-of-fold AUC was a suspicious 0.94 and the encoding was the #1 SHAP feature. The time-aware version brought AUC to an honest 0.785.

---

## 4. Phase 3: Models (`models.py`, `pipeline.py`)

### 4.1 Keep (credibility) models: LightGBM, three stages

| Stage | Inputs | Test AUC | ECE | Brier |
|---|---|---|---|---|
| Pre-call | account + history + agent | 0.567 | 0.032 | 0.201 |
| **In-call** (trained on every customer-turn prefix) | + PTP terms + conversation + acoustics | **0.763** | **0.037** | 0.159 |
| Post-call (≤36 h) | + agent-entered fields + engagement | 0.765 | 0.035 | 0.160 |

Out-of-fold over all 3,696 PTPs (account-grouped 5-fold cross-fitting, which is also what fills the CSV): pre 0.579, **in-call 0.785**, post 0.783.

Readings:
* **The conversation is where the signal is.** Account data alone barely ranks PTPs (0.57). The six largest SHAP drivers are days-to-promise, the borrower's firmness and hedging/exit scores, the longest customer pause, promised day-of-month and promise size vs ability to pay (`outputs/figures/shap_in_call.png`).
* **Post-call engagement adds almost nothing on top of the call** in this data. The parking decision can therefore be made at hang-up, and engagement used only to re-open parked accounts early.
* **Calibration:** global Platt plus a per-portfolio intercept shrunk towards zero (n/(n+150)). It is rank-preserving within a portfolio and low-variance on a 561-row validation set. Isotonic was tried first and dropped: it cost 0.025 AUC. Per-portfolio test ECE ranges 0.04–0.11 except MSME (0.18 on only 55 PTPs); this needs more validation volume before go-live.

### 4.2 Category model: cost-sensitive multiclass

LightGBM multiclass on the in-call features, class-balanced and temperature-calibrated. It is decoded by **Bayes risk** against `config.COST_MATRIX`: genuine hardship predicted as escape or repeat costs 8×, third party predicted as anything that discusses the debt costs 5×.

Test split, end of call (ground truth = gold + pseudo):

| category | precision | recall | F1 | support |
|---|---|---|---|---|
| genuine_feasible | 0.738 | 0.472 | 0.576 | 161 |
| genuine_infeasible | 0.515 | 0.820 | 0.633 | 122 |
| escape | 0.807 | 0.630 | 0.708 | 73 |
| third_party | 0.933 | 0.982 | 0.957 | 57 |
| repeat_promiser | 0.760 | 0.709 | 0.734 | 103 |
| agent_recorded_or_pushed | 0.830 | 0.936 | 0.880 | 47 |
| **macro avg** | **0.764** | **0.758** | **0.748** | 563 |
| weighted avg | 0.730 | 0.702 | 0.698 | 563 |

Confusion matrix (rows = true, columns = predicted, same order as above):
```
[[ 76  48   7   2  20   8]
 [ 20 100   2   0   0   0]
 [  1  21  46   1   3   1]
 [  0   1   0  56   0   0]
 [  4  24   2   0  73   0]
 [  2   0   0   1   0  44]]
```
Plain argmax scores higher (accuracy 0.760, macro-F1 0.791) but sends **12.3%** of hardship cases to escape/repeat. The cost-sensitive decision cuts that to **1.6%** (2 of 122 PTPs), at the price of over-calling hardship among genuine-feasible PTPs, which is the safe direction (they get a softer, not a harsher, treatment). One-vs-rest AUC 0.944; top-1 ECE 0.064. On the 27 human-annotated PTPs in the test split: accuracy 0.78, macro-F1 0.77.

### 4.3 Explainability

Every row of the enriched CSV carries:
* a **signal explanation** in plain words ("borrower never stated a date; agent proposed it; 2 pressure phrase(s) before the promise; hardship mentioned"), and
* the **top-3 exact TreeSHAP contributions** (LightGBM `pred_contrib`) with values and signs, readable by an agent, a supervisor or an auditor.

---

## 5. Phase 3b/4: Real-time in-call copilot and latency

### 5.1 Tabular streaming copilot (`streaming.py`), the production path

`InCallCopilot.on_turn(...)` runs once per ASR-finalised utterance. It computes lexical signals for the new turn only, updates cumulative prefix features with *the same functions used in training* (no train/serve skew), then runs both models, guardrails, the script and the compliance filter. It also sets `heard_enough` once a commitment or record event has happened and the score has been stable (|Δp| < 0.05, same category, confidence ≥0.5) over consecutive turns.

| quality vs share of call heard (test) | keep AUC | category macro-F1 |
|---|---|---|
| 0–25% | 0.690 | 0.22 |
| 25–50% | 0.724 | 0.60 |
| 50–75% | 0.776 | 0.68 |
| 75–99% | 0.776 | 0.55 |
| end of call | 0.787 | 0.72 |

The "heard enough" trigger fires on 87.8% of calls, at a median 66.7% of the call and 7.6 s before hang-up. At the trigger, keep AUC is 0.768, category accuracy 0.72, and 95.8% agree with the end-of-call category, so the prompt reaches the agent while the borrower is still on the line.

**Latency** (120 test calls, 1,432 turns, CPU, single thread): **p50 60 ms, p95 78 ms, p99 96 ms** per update. The first version took 145 ms; profiling showed pandas column-insert overhead, and batching column construction plus incremental signal extraction fixed it. This fits comfortably inside the 600–1,000 ms human conversational pause.

`outputs/streaming_demo.md` replays one held-out call per category, turn by turn. Example: the escape call is flagged after the borrower's first utterance ("I forgot about it"), and at "okay okay, I will pay, I have to go" the copilot fires *heard enough* and proposes a ₹2,000 link-payment ask before the call ends.

### 5.2 SLM track (`slm.py`)

**Design.** Qwen2.5-0.5B-Instruct, LoRA r=16 / α=32 on all attention and MLP projections (8.8 M trainable parameters, 1.75%). It was trained on 2,400 dialogue prefixes (taken from ≥40% into the call, plus the full call) for 2 epochs, in 25 min at 1.45 GB peak VRAM on an RTX 3050 Laptop (4 GB).

Output format: `<code>|<category>|<next agent line>`. The **first generated token is a single digit 1–6**, so:
* the category and its confidence are available at **time-to-first-token**;
* confidence = softmax over the six digit logits, temperature-scaled on validation;
* the utterance streams afterwards.

Targets: Phase-1 category plus the approved, compliance-filtered policy script (policy distillation: the SLM is taught to *say only pre-approved FPC/DPDP-safe lines*).

**A training bug caught:** the first run collapsed to predicting `genuine_feasible` for every call (macro-F1 0.07). The class code is 1 of ~45 target tokens, so the loss was dominated by the easy templated utterance. Giving the code token its own loss weight (5:1) fixed it.

**Quality (test split, 484 calls, text + compact context line only):**

| | accuracy | macro-F1 | top-1 ECE (calibrated) |
|---|---|---|---|
| SLM fp16, end of call | 0.645 | 0.673 | 0.042 |
| SLM fp16, mid-call prefix | 0.641 | 0.669 | 0.055 |
| SLM NF4 4-bit, end of call | 0.605 | 0.642 | 0.039 |
| *Tabular LightGBM (for reference)* | *0.702* | *0.748* | *0.064* |

Per class, the SLM matches the tabular model on third-party (F1 0.97) and agent-pushed (0.85), which are conversation-only phenomena. It is weaker on repeat-promiser (0.51) and the feasible/infeasible split, which depend on account history the SLM sees only as a one-line summary.

Generated text (150 calls): 100% valid format; 37% exact match to the policy script; **0 compliance violations** both raw and after the serving guardrail.

The raw SLM sends **17%** of hardship cases to escape/repeat, which is exactly why, in the deployed design, **the SLM never decides treatment on its own:**
1. Its category probabilities are an input next to the tabular model's.
2. Deterministic lexical guardrails can only soften the choice.
3. The utterance actually shown or spoken is re-rendered from the approved template for the final, guard-railed category. In 1 of the 5 inspected samples the SLM had paired a repeat-promiser code with a hardship script.

**CUDA-synchronised latency** (`torch.cuda.synchronize()` + `time.perf_counter()`, 40 held-out calls, mean prompt 330 tokens, batch 1, HF eager, no compilation, RTX 3050 Laptop 4 GB):

| precision | weights | **peak VRAM** | cold TTFT p50 / p95 | **incremental TTFT p50 / p95** | decode | 48-token E2E p50 |
|---|---|---|---|---|---|---|
| fp16 | 0.996 GB | 1.279 GB ✗ (>1 GB) | 33.0 / 40.5 ms | 27.4 / 38.2 ms ✓ | 46.2 tok/s | 1,034 ms |
| **NF4 4-bit** | 0.548 GB | **0.740 GB ✓** | 56.4 / 69.4 ms | **46.7 / 56.6 ms ✓** | 29.6 tok/s | 1,630 ms |

* *Cold* re-encodes the whole prefix every turn. *Incremental* keeps the call's KV cache and pre-fills only the new utterance plus the 5-token assistant header, then crops the header back off; the incremental class code matches the cold one in 97.5% (fp16) / 90% (NF4) of calls.
* **Only NF4 meets both SLAs** (TTFT < 100 ms and < 1 GB VRAM), at a cost of ~4 points of accuracy. fp16 meets TTFT but not VRAM.
* **TTFT is the number that matters** here: the category and confidence come out on the first token, inside the 600–1,000 ms conversational pause. The full utterance is the slow part in eager mode (1.0–1.7 s for 48 tokens). That is acceptable for an *agent desktop* prompt the agent reads, and the voice bot does not need generation at all because the line is rendered from the approved template.

**Path to production serving (projection, not measured here).** The prototype is uncompiled HF eager mode at batch 1, where Python overhead dominates decode, about 20–35 ms per token. A serving engine removes most of that:
* vLLM (PagedAttention, continuous batching, CUDA graphs) or TensorRT-LLM (fused kernels, in-flight batching; INT4-AWQ/FP8 weights);
* plus prefix caching of the shared system + context block.

Publicly reported numbers for 0.5B-class models on such engines are hundreds of tokens/s per stream, which would bring a 48-token reply to the stated ~120–180 ms while serving dozens of concurrent calls per GPU. The same benchmark (`slm.benchmark`) should be re-run on the target serving stack before committing to that SLA.

---

## 6. Guidance, guardrails and dynamic parking (`policy.py`)

Fixed processing order. Guardrails can only **soften** treatment, never harden it:

1. **Guardrails.**
   * Hardship signal (in-call, or an `rpc_hardship` disposition in the last 30 days) with a predicted escape/repeat → re-routed to `genuine_infeasible`.
   * Severe hardship → `hardship_desk_human`.
   * Third-party or wrong-number signal → `third_party` script.
2. **Script.** Six scripts × 3 languages (Hinglish, Kanglish, English), parameterised with amount, a token ask (₹1,000–2,000, never above 40% of the promise) and a post-salary date (salary day + 2). Until the borrower has committed to anything, the copilot only coaches the agent to *ask for the borrower's own date*.
3. **Compliance filter.** Deterministic regex check. It blocks threats or abuse (police, jail, family or neighbour shaming), any debt word in a third-party script (DPDP), and contact outside 08:00–19:00 (RBI FPC).
4. **Disposition and parking engine.** Replaces "park until the PTP date" with a schedule by category × calibrated p(keep):

| category / condition | parking | reminders | queue |
|---|---|---|---|
| genuine_feasible, p ≥ 0.6 | until due date | D-1 | park |
| genuine_feasible, 0.35–0.6 | due date − 1 | D-2 WhatsApp, D-1 | park_with_check |
| genuine_feasible, p < 0.35 | ≤ 3 days | D+0 link, D-1 | early_followup |
| genuine_infeasible | until the (rescheduled) date | D-1, salary-day nudge | restructuring_review; no escalation |
| severe hardship | ≤ 3 days | soft D-1 | hardship_desk_human |
| escape | 1–2 days | D+0 token link, D+1 | early_followup |
| repeat_promiser | ≤ 2 days | D+0 link, D+1 | senior_caller |
| third_party | 0, call the borrower within 24 h | none | rpc_confirmation |
| agent_recorded_or_pushed | 1 day | WhatsApp self-confirmation | integrity_review; independent confirm call |

### 6.1 Regulatory audit (all 3,696 PTPs)

| check | result |
|---|---|
| PTPs with any hardship signal | 1,596 |
| ... routed to an aggressive queue (senior caller / escape script) after guardrails | **0** |
| ... softened by a guardrail | 452 |
| Third-party PTPs; copilot utterances that mention the debt | 342; **0** |
| Copilot utterances with threatening or abusive language | **0** |
| *Historical baseline:* third-party calls where the agent named an amount or PTP after a relative identified themself | 334 / 428 (**78%**) |

The last row is a finding for CN in its own right. In the historical data, agents routinely log "₹6,000 on 16 May, I am recording it" to a borrower's mother. The copilot's third-party script removes that exposure, and the `rpc_confirmation` queue makes sure the real borrower is the one who confirms.

### 6.2 DPDP handling of voice

Voice-derived features (pauses, ASR confidence) are used only as one input, never alone (pause features have moderate SHAP weight; no rule uses them). Raw audio is not stored by this service: only per-turn numeric features and the transcript tokens needed for audit are kept, with purpose limited to collections credibility. The retention period should be set by CN's DPO.

---

## 7. Phase 5: Business impact (`pipeline.business_impact`)

**Observed on held-out PTPs** (no assumptions):
* 88.4% of broken PTPs are flagged before parking. 84% of flags are true breaks. 43% of kept PTPs also get an early, cheap touch (bot or WhatsApp, ₹2.25), i.e. ₹272 per 1,000 PTPs.
* Average parking falls from 7.26 to 4.77 days. That frees **1,893 contact-days per 1,000 PTPs** on promises that were going to break anyway.

**Projected** (needs the RCT; the scenario is the share of flagged-broken PTPs that in-call firming or early follow-up converts into a payment):

| scenario | keep rate | broken-rate reduction | escalation cost avoided / 1,000 PTPs | recovery lift / 1,000 PTPs |
|---|---|---|---|---|
| baseline | 28.2% | n/a | n/a | n/a |
| conservative (8%) | 33.3% | −5.1 pp | ₹8.3k | ₹0.70 L (token only) to ₹7.9 L (full promise) |
| base (15%) | 37.8% | −9.5 pp | ₹15.6k | ₹1.32 L to ₹14.8 L |
| optimistic (22%) | 42.2% | −14.0 pp | ₹22.8k | ₹1.93 L to ₹21.7 L |

Escalation cost assumes a broken PTP costs 3 tele connects (₹22.5) plus a 35% chance of a field visit (₹275); every assumption is in `config.py`.

**Measuring real value (randomised trial).** The prompts change the outcome the model predicts, so accuracy on old calls is not enough. Randomise at **account** level and stratify by portfolio × bucket × channel; arms: (A) score only, used for parking; (B) score + in-call prompts; (C) control. Detecting a +3 pp keep-rate lift from 28.2% (α 0.05, power 0.8) needs **3,644 PTPs per arm**, about 4–6 weeks at sandbox volumes. Primary metric: keep rate and ₹ recovered at +14 days. Guardrail metrics: complaint rate, hardship-desk volume, and third-party disclosure incidents (target 0).

### Agent integrity (`outputs/agent_integrity_dashboard.csv`)

The dashboard reports quality-weighted PTPs (Σ p_keep) next to raw counts for every agent and the bot, plus pushed share, generic-remark share, shift-end share and keep lift vs peers, combined into a z-score. Two agents are flagged:
* **TA013:** 55% copy-pasted generic remarks, keep rate 11 pp below peers.
* **TA004:** 30% of its PTPs classified agent-pushed.

These are outcome-independent flags, so they also work for new agents.

### Staying hard to game and drift

* The credibility score uses no agent-typed field in-call. Agent remarks are used only *against* the agent (integrity), never as positive evidence.
* PSI between train and test on the top 15 drivers is ≤0.10 for all of them (largest: `salary_credit_day` at 0.098). Production should alert at PSI > 0.2 and re-check whether agents learn to say the "firm" phrases (watch for a rising `firmness_score` with a falling keep rate).

---

## 8. Code map and reproduction

```
ptp_copilot/
  config.py          business constants (grace, cure, 90%, costs, cost matrix, SLA)
  data.py            loaders
  labels.py          kept definition, 20-rule hindsight labeller, self-training bootstrapper
  lexicon.py         code-mixed Hinglish/Kanglish/English regex lexicons, date/amount/salary extraction
  features.py        per-turn signals -> prefix features; account, agent-integrity, post-call blocks
  models.py          LightGBM keep models, portfolio calibrator, cost-sensitive category model, ECE, SHAP
  policy.py          guardrails, 6x3 scripts, compliance filter, dynamic parking engine
  pipeline.py        end-to-end: build -> labels -> train -> evaluate -> cross-fit -> decide -> audit -> business
  streaming.py       InCallCopilot: live per-turn engine
  demo_streaming.py  replays + CPU latency benchmark
  slm.py             SFT data, LoRA fine-tune, eval, CUDA-synchronised latency benchmark
  figures.py         PNG charts
run_all.py           python run_all.py [--slm]
```
Dependencies: Python 3.11, pandas 3, numpy, scikit-learn, lightgbm, scipy, matplotlib; for `--slm`: torch (CUDA), transformers ≥5, peft, bitsandbytes, plus the base weights in `outputs/hf/Qwen2.5-0.5B-Instruct` (`huggingface_hub.snapshot_download`).

*Model weights are not in the git repository* (each `model.safetensors` is 942 MB; GitHub's limit is 100 MB per file). They are published on the Hugging Face Hub at **[Shubhbagaria12/QWEN_OPENIIT](https://huggingface.co/Shubhbagaria12/QWEN_OPENIIT)**: merged fine-tuned weights at the repo root, the LoRA adapter in `lora_adapter/`, plus a model card with usage and results. **No manual step is needed:** `ptp_copilot.slm.load` (used by the evaluation, the benchmark and the web app) downloads the merged model into `outputs/slm_merged/` on first use, and `slm.train` downloads the base model `Qwen/Qwen2.5-0.5B-Instruct` into `outputs/hf/`. Override the source with `PTP_HF_MODEL_REPO=<user/repo>`. The LoRA adapter is also committed in git (`outputs/slm_lora/`, 34 MB).

*Environment note:* the 2.6 GB CUDA torch wheel could not be downloaded over this machine's connection. The GPU runs therefore borrowed an existing local CUDA torch 2.14.0+cu126 build *read-only* via `PTP_TORCH_SITE=<other env>/Lib/site-packages`; `slm.py` appends that path *after* its own, so this environment's packages always take priority. With torch installed normally, leave the variable unset.

## 9. Limitations and next steps

1. **Labels.** 200 gold labels and annotator noise cap the category metrics; the 0.67 rule/human agreement is the real ceiling. Next step: double annotation and a written labelling guide.
2. **Synthetic sandbox.** The transcripts are templated, so the lexicons fit them closely. Real ASR will need the lexicons extended from production n-grams, or the SLM track to carry more weight.
3. **Counterfactual bias.** Outcomes reflect the incumbent rule-based policy. Only 5% of accounts sit in the `random_contact_point` arm, and they were not used for re-weighting; the RCT in §7 is the fix.
4. **Calibration volume.** Per-portfolio calibration has only 55–163 validation PTPs per portfolio. Re-fit monthly on a rolling window.
5. **Field-visit PTPs** (16%) have no transcript and are scored only on account, agent and PTP terms.

---

## 10. Web application: PTP Intelligence console (`webapp/`)

A full-stack app (FastAPI + React/Vite) built *on top of* the existing `ptp_copilot` package. It does not retrain or replace any model.

### 10.1 How it is wired

```
csv_folder/*.csv  ─┐
outputs/*.csv/json ┼─> webapp/backend/services.py ── reuses ──> ptp_copilot.pipeline.build (feature store; cached to outputs/app_cache.pkl)
                   │                                           ptp_copilot.features (turn signals + prefix features)
                   │                                           outputs/models/models.pkl (LightGBM in-call keep + category models)
                   │                                           ptp_copilot.policy (guardrails, approved scripts, compliance, parking)
                   │                                           ptp_copilot.slm.load / code_logits (fine-tuned Qwen, outputs/slm_merged)
                   └─> webapp/backend/app.py (REST API; also serves the built frontend) ──> webapp/frontend (React)
```

| What the UI shows | Where it comes from | Badge |
|---|---|---|
| Credibility (P(kept)) | LightGBM in-call model, calibrated, run live on the transcript prefix | MODEL OUTPUT |
| Behaviour category (tabular) | LightGBM category model, cost-sensitive decision | MODEL OUTPUT |
| Qwen category + confidence | fine-tuned Qwen2.5-0.5B (LoRA merged); first-token digit logits, temperature-calibrated | MODEL OUTPUT |
| Final category, safeguards, treatment, script | `policy.apply_guardrails` → `guidance` → `compliance_check` → `parking_plan` | MODEL OUTPUT |
| Commitment ownership, evidence quotes | existing lexicon applied to the revealed turns only | MODEL OUTPUT |
| Lists (live calls, customers, agents) | out-of-fold predictions from `enriched_ptp_predictions.csv` | STORED MODEL OUTPUT |
| Outcomes, payments, DPD, remarks | supplied CSVs | ACTUAL |
| AUC, ECE, flag rate, benchmark latency | `results.json`, `latency_*.json`, `slm_eval.json` | MEASURED |
| Recovery lift, cost avoided | business scenarios in `results.json` | PROJECTED |
| AI-vs-control comparison | no experiment exists in the data | UNAVAILABLE |

**Which secondary model runs (Qwen vs Gemini).** The **fine-tuned Qwen SLM is the project's secondary model**: every reported result, the evaluation, the report and the deck use it. Hosting Qwen on a public website would need a paid GPU, so the **deployed website uses Google Gemini (`gemini-3.5-flash-lite`, API, prompted, not fine-tuned)** in the same slot instead. The choice is made at start-up by `PTP_LLM_PROVIDER` (or `python webapp/start.py --llm …`):
* `auto` (default): Qwen if a CUDA GPU is available, otherwise Gemini;
* `qwen`: force the local Qwen SLM;
* `gemini`: force Gemini, which needs `GEMINI_API_KEY` in the environment (never in code or git).

Gemini's early held-out score (124 of 484 test calls, `experiments/gemini_partial_score.py`; Qwen and LightGBM re-scored on the same calls):

| model | accuracy | macro-F1 | hardship → escape/repeat |
|---|---|---|---|
| LightGBM (deployed decision) | 0.694 | 0.678 | 3.1% |
| Qwen fine-tuned (fp16) | 0.653 | 0.649 | 21.9% |
| Gemini 3.5 Flash-Lite | 0.427 | 0.474 | 18.8% |

Gemini is weaker because it is only prompted, while the other two learned our labelling conventions. It also takes about 1.8 s per call, so the console shows LightGBM instantly and Gemini asynchronously ("as of turn N"). In deployment mode the Overview shows a notice that the reported model results come from Qwen. Transcripts are sent to Google in this mode: use a paid key and sandbox data only.

**Model loading.** Qwen is loaded once per backend process, in a background thread, through `ptp_copilot.slm.load`. The default is `nf4`, which meets the < 1 GB / < 100 ms SLA; set `PTP_SLM_PRECISION=fp16` for the more accurate fp16 weights (1.28 GB). It is warmed up once, and GPU calls are serialised with a lock. `/api/health` and the sidebar report the real status (`loading` / `ready` / `unavailable` with the error). If Qwen cannot load, every Qwen field shows *unavailable* and the tabular path keeps working. This was verified by starting the server without CUDA torch.

**Live calls are a replay** (badge `LIVE DEMO SIMULATION`). A recorded call from `call_transcripts.csv` is revealed utterance by utterance, paced by the call's own recorded timing ÷ speed (1x/2x/4x). After each utterance the backend re-scores **only the prefix seen so far** with both models. This is stateless prefix re-scoring; the KV-cache incremental mode exists only in the offline benchmark. The ground-truth category and payment outcome are revealed only after the last turn.

**Free-generated Qwen text is never shown to the agent.** "Generate raw Qwen draft" displays it, labelled, for transparency; the agent always sees the approved script for the guard-railed category.

### 10.2 Run it

Requirements: Python 3.11 with the packages in §8 plus `fastapi` and `uvicorn`. Node.js 18+ is needed only to build the frontend. A CUDA GPU is needed for Qwen; without one the app runs with Qwen marked unavailable.

```bash
# from the repository root: builds the frontend if needed, then serves UI + API on one port
python webapp/start.py --torch-site "<site-packages of an env with CUDA torch>"   # omit if torch is installed normally (Qwen)
python webapp/start.py --llm gemini     # deployment mode on a host without a GPU (set GEMINI_API_KEY first)
# open http://127.0.0.1:8000

# development (hot reload): backend on :8000, Vite dev server on :5173 proxying /api
python webapp/start.py --dev
cd webapp/frontend && npm install && npm run dev
```
First start takes about 15 s for data and 10 s for Qwen (the sidebar shows progress). Later starts reuse `outputs/app_cache.pkl`.

### 10.3 Judge demo path

1. **Overview:** KPIs with provenance badges, four charts, measured model quality.
2. **Live Calls → Demo scenarios.** Six held-out test calls, one per ground-truth category, all human-annotated (e.g. Third-Party `AT0024055`, Genuine-but-Infeasible `AT0017753`).
3. **Start (4x).** Watch the transcript stream. Credibility, both categories, commitment ownership, evidence (with the triggering quote), SHAP drivers and the treatment update after every utterance. The infeasible call raises **HARDSHIP SAFEGUARD ACTIVE**, the third-party call **PRIVACY SAFEGUARD ACTIVE**. At the end, the ground truth and an audit link appear.
4. **Customer profile:** account health, payment timeline, PTP history with outcome vs model output, dial remarks.
5. **Treatment Engine:** the five-input decision path → decision, plus a deterministic reasoning trace.
6. **Agent Quality → TA013:** why the "potential PTP inflation" signal appears, against the peer average.
7. **Analytics:** filters (lender, portfolio, channel, agent, category, date); calibration; category performance; the honest "no AI-vs-control experiment" note.
8. **Audit & Compliance:** batch audit (measured), plus every decision the app made (`outputs/app_audit.jsonl`), with detail pages.

### 10.4 API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | backend, data, tabular, Qwen (status, version, device, VRAM, measured latency) |
| GET | `/api/overview` | KPIs and chart data with provenance |
| GET | `/api/live-calls` | replayable calls; `q, category, risk, status, split, lang, sort, order, page, size` |
| GET | `/api/live-calls/scenarios` | demo scenarios |
| GET | `/api/live-calls/{call_id}` | call, account, PTP, transcript |
| POST | `/api/live-calls/{call_id}/step` | `{turn, qwen}` → assessment of prefix 0..turn (audited at the final turn) |
| POST | `/api/live-calls/{call_id}/predict` | full-call assessment (audited) |
| POST | `/api/live-calls/{call_id}/generate` | raw Qwen generation + compliance check (audited; never shown to the agent) |
| GET | `/api/customers`, `/api/customers/{account_id}` | accounts; profile, PTPs, payments, dials |
| GET | `/api/agents`, `/api/agents/{agent_id}` | integrity dashboard; per-agent evidence |
| GET | `/api/treatment/options`, `/api/treatment/{ptp_id}` | PTP picker; live decision path (audited) |
| GET | `/api/analytics` | filtered analytics |
| GET | `/api/audit`, `/api/audit/{audit_id}` | audit trail |

### 10.5 Tests

```bash
python webapp/tests/api_smoke_test.py      # 15 checks: model/tokenizer load, inference, parsing, confidence, prefix
                                           # scoring, determinism, safeguards, lookups, treatment, audit
node webapp/tests/e2e_browser.mjs <dir>    # needs `npm i puppeteer-core` and local Edge; 27 checks: every page,
                                           # two full live replays, safeguard banners, raw Qwen draft, audit detail,
                                           # 404, 390 px mobile, no console errors
```
Last run: API 15/15 pass; browser 27/27 pass. Measured live-step latency on the RTX 3050 Laptop: tabular 64–79 ms per prefix, and Qwen NF4 56–140 ms per prefix (cold prefill of the whole prefix, because the app re-scores statelessly).

### 10.6 Limitations of the app

* Calls are replays of recorded transcripts, not live telephony audio; there is no ASR in the loop.
* The supplied data has no customer names or contact details, so customers are shown by account ID.
* "Active PTPs" does not exist in historical data. The Overview shows all captured PTPs (1 is still inside its grace window at data end).
* Qwen is the weaker classifier (macro-F1 0.64 at NF4 vs 0.75 tabular). It is shown as a second opinion, and the decision uses the tabular model plus guardrails.
* The Qwen path needs CUDA torch; on this machine it borrows an existing local CUDA build via `--torch-site` (see §8).
* The audit log already contains the records created by the test runs above. Delete `outputs/app_audit.jsonl` for a clean demo.
