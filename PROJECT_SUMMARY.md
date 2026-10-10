# PTP Credibility Copilot: Project Summary

**Problem:** CreditNirvana Problem Statement 1, "Real-Time Fake PTP Detection & In-Call Guidance".
**In one line:** while a collections call is still happening, we estimate how likely the borrower's promise-to-pay (PTP) is to be kept and *why* it might break. We then tell the agent what to say next and set a follow-up plan based on how credible the promise is, with regulatory safeguards built in.

---

## 1. The problem

- More than half of PTPs never turn into payment. A broken PTP wastes days while the account is "parked" until the promised date, sends the account to costlier channels (tele-caller ₹15–30, field visit ₹150–400), and inflates agent PTP counts.
- CN defines **6 promise types**, each needing a different response:

| Type | Meaning | Right response |
|---|---|---|
| Genuine & Feasible | means it, can pay | park until due date, remind D-1 |
| Genuine but Infeasible | means it, cash flow doesn't fit (hardship) | date after salary / smaller amount / restructure; never pressure |
| Escape | says yes to end the call | small part-payment now, follow up in 24–48 h |
| Third Party | relative/colleague promises for the borrower | never disclose the debt; confirm with borrower |
| Repeat Promiser | many past broken PTPs | stronger treatment, short parking |
| Agent-Recorded / Pushed | agent forced or invented the date | integrity review, independent confirmation |

- Constraints: decide **before the call ends** within the conversational pause; never treat hardship harshly (RBI Fair Practices Code); never disclose debt to third parties (DPDP); every flag explainable.

## 2. Data (CN sandbox, `csv_folder/`)

| | |
|---|---|
| PTPs | 3,696 (3,091 from calls, 605 from field visits) |
| Call transcripts | 3,367 calls / 39,194 turns: Hinglish, Kanglish, English; with pause-before-turn and ASR confidence |
| Payments | 2,162 across 7 channels |
| Accounts / dial attempts | 2,400 accounts (6 lenders, 6 portfolios) / 51,105 attempts with agent remarks |
| Human labels | 200 annotated PTPs |
| Split | by account: 70% train / 15% validation / 15% test (563 test PTPs) |

## 3. Pipeline (`ptp_copilot/`)

```
CSV data → ① ground truth → ② features → ③ models → ④ calibration + evaluation → ⑤ policy/decisions → outputs/
```

1. **Ground truth** (`labels.py`)
   - **Kept** = at least 90% of the promised amount paid, across all channels, by promised date + 2 days of grace. A **cure** window of +14 days gives "kept late".
   - Result: 27.7% kept on time, 53.5% broken.
   - **The 6 types:** 20 transparent rules plus confidence-gated self-training (LightGBM) label all 3,696 PTPs, starting from the 200 human labels.
   - Agreement with the human labels: **0.67 accuracy / 0.69 macro-F1**. This caps how meaningful any category score can be.
2. **Features** (`features.py`, `lexicon.py`), all time-aware (nothing from after the decision point):
   - **What was said:** multilingual phrase lists for hedging ("dekhta hoon"), exit attempts, conditions ("salary aayi toh"), vague dates, firm language, hardship, third party ("main unki mummy hoon").
   - **Who drove the promise:** who stated the date or amount first, passive "haan" to an agent-proposed date, agent pressure.
   - **How it was said:** pauses, hesitations, ASR confidence.
   - **History and context:** past broken PTPs, salary date vs promised date, ability to pay, dialler history, the agent's past keep rate vs peers.
3. **Models:** see section 4.
4. **Calibration:** Platt scaling plus a shrunk per-portfolio offset, so a 0.6 means the same in every portfolio. SHAP gives an explanation for every prediction.
5. **Policy** (`policy.py`):
   - **Guardrails** can only soften treatment: hardship goes to restructuring or a human, third parties get the no-disclosure script.
   - **Scripts:** approved lines for each of the 6 types in 3 languages, with amount, token ask and post-salary date filled in.
   - **Compliance filter:** blocks threats, any debt mention to a third party, and calls outside 08:00–19:00.
   - **Parking engine:** sets parking, reminders and routing by type × credibility.

## 4. Models and scores (held-out test unless stated)

| Model | Library | Predicts | Used for |
|---|---|---|---|
| Pre-call keep | LightGBM | P(kept) from account + history + agent | planning |
| **In-call keep** | LightGBM | P(kept) from the conversation **so far** (trained on every turn prefix) | **live credibility score** |
| Post-call keep | LightGBM | + link opens / WhatsApp replies (≤36 h) | follow-up |
| **Category** | LightGBM, cost-sensitive (hardship mistakes weighted 8×) | which of the 6 types | **treatment decision** |
| Qwen SLM | Qwen2.5-0.5B-Instruct + LoRA (r=16), fine-tuned | type + confidence (first token = class code) + draft line | second opinion only |

**Will it be kept?**

| Model | AUC | ECE | Brier | AUC across all PTPs (out-of-fold) |
|---|---|---|---|---|
| Pre-call | 0.567 | 0.032 | 0.201 | 0.579 |
| **In-call** | **0.763** | **0.037** | **0.159** | **0.785** (ECE 0.025) |
| Post-call | 0.765 | 0.035 | 0.160 | 0.783 |

- **As the call progresses**, in-call AUC rises: 0.69 (first quarter) → 0.79 (full call).
- **Flagging:** 88.4% of broken PTPs are flagged before parking, and 84% of flags are correct.

**Which type?**

| Model | Accuracy | Macro-F1 | Hardship → escape/repeat |
|---|---|---|---|
| LightGBM, plain best guess | 0.760 | 0.791 | 12.3% |
| **LightGBM, cost-sensitive (deployed)** | 0.702 | 0.748 | **1.6%** (0 after guardrails) |
| Qwen fp16 | 0.645 | 0.673 | 17.2% |
| Qwen 4-bit (used in app) | 0.605 | 0.642 | n/a |

- Qwen matches LightGBM on third-party (F1 0.97) and agent-pushed (0.85), but is weaker on history-dependent types, so it is a second opinion only.
- Qwen's generated text: 100% valid format, 0 compliance violations.
- **XGBoost check** (`experiments/xgboost_comparison.py`): same features and setup give the same accuracy within noise (in-call AUC 0.761, category macro-F1 0.757). LightGBM was kept because it is 3× faster to train and about 8 ms faster per prediction.

**Speed** (one laptop: CPU + RTX 3050 4 GB):
- LightGBM live update: **60 ms** p50 per turn.
- Qwen 4-bit first token: **47 ms** at **0.74 GB** GPU memory. This meets the <100 ms and <1 GB targets; fp16 uses 1.28 GB and misses the memory target.

**"Heard enough" trigger:** fires on 88% of calls, about two-thirds of the way in and roughly 7.6 s before hang-up, and agrees with the end-of-call type 96% of the time.

## 5. Live architecture (one step per utterance)

```
CN ASR utterance → signals for the new turn → cumulative features (same code as training)
   → LightGBM credibility + category  ┐
   → Qwen category (second opinion)    ┘→ guardrails → approved script + compliance → parking / routing → audit log
```
We do **not** transcribe audio ourselves: the system consumes CN's ASR output. The demo replays recorded calls utterance by utterance at their real timing.

## 6. Safeguards (audited on all 3,696 PTPs)

- **Hardship:** 0 of 1,596 hardship PTPs routed to aggressive treatment.
- **Third parties:** 0 of 342 third-party scripts disclose the debt.
- **Language:** 0 scripts contain threatening language.
- **Finding for CN:** in historical calls, agents disclosed the amount or PTP to a relative in **78%** of third-party calls (334 of 428).
- **Agent integrity:** flags agents for review, e.g. TA013 (55% copy-pasted remarks, keep rate 11 points below peers).

## 7. Business impact

- **Measured:**
  - Average parking falls from 7.26 to 4.77 days.
  - 1,893 contact-days freed per 1,000 PTPs.
- **Projected** (needs a randomised trial of about 3,644 PTPs per arm), assuming 8–22% of flagged broken PTPs are converted:
  - keep rate 28% → 33–42%
  - escalation cost avoided: ₹8–23k per 1,000 PTPs

## 8. Web dashboard (`webapp/`: FastAPI + React)

| Page | What it shows |
|---|---|
| Overview | KPIs and charts, every number tagged Actual / Model output / Measured / Projected |
| Live Calls | 3,067 replayable calls + 6 demo scenarios (one per type) |
| **Live Call Console** | Start/Pause/Resume/Restart/Stop at 1–4× speed. A "Copilot now" panel next to the transcript shows credibility, type (LightGBM + Qwen), the suggested next line, action, parking, safeguards, commitment ownership and evidence, updated every utterance |
| Customers | account health, payments, PTP history, dial remarks |
| Agent Quality | quality-weighted PTPs, integrity flags with reasons |
| Treatment Engine | decision path for any PTP |
| Analytics | filterable trends, calibration, category performance |
| Audit & Compliance | batch audit + a log of every decision the app makes |

Tested: API 15/15 and browser end-to-end 27/27 pass.

**Secondary model: Qwen in results, Gemini in deployment.**
- **Qwen everywhere results are reported:** the fine-tuned Qwen SLM is the secondary model for all reported results, the report and the deck.
- **Gemini on the public website:** hosting Qwen would need a paid GPU, so the deployed site uses Google Gemini 3.5 Flash-Lite (API) in the same slot. `PTP_LLM_PROVIDER=auto` picks Qwen when a GPU is available, otherwise Gemini.
- **Gemini's early score:** on 124 held-out calls, Gemini reaches macro-F1 0.474, vs 0.649 for Qwen and 0.678 for LightGBM on the same calls. It is prompted, not trained on our labels, and takes about 1.8 s per call, so it runs asynchronously.

## 9. Deliverables and where they are

| Deliverable | Location |
|---|---|
| Pipeline code | `ptp_copilot/`, `run_all.py` |
| Enriched predictions for every PTP (type, confidence, explanation, next line, parking plan) | `outputs/enriched_ptp_predictions.csv` |
| All metrics, reports, confusion matrix | `outputs/results.json`, `outputs/pipeline_log.txt`, `outputs/figures/` |
| Latency benchmarks | `outputs/latency_tabular.json`, `outputs/latency_slm.json` |
| Fine-tuned Qwen model | Hugging Face [`Shubhbagaria12/QWEN_OPENIIT`](https://huggingface.co/Shubhbagaria12/QWEN_OPENIIT) (auto-downloaded); adapter in `outputs/slm_lora/` |
| Web app + tests | `webapp/` |
| Full documentation | `README.md` |

## 10. How to run

```bash
python run_all.py                 # pipeline + figures (CPU, ~4 min)
python run_all.py --slm           # + fine-tune / evaluate / benchmark Qwen (GPU)
python webapp/start.py            # dashboard at http://127.0.0.1:8000 (Qwen when a GPU is present)
python webapp/start.py --llm gemini   # deployment mode without a GPU (needs GEMINI_API_KEY)
```

## 11. Known limitations

- **Labels:** the 6-type labels agree with humans only 67% of the time, so category metrics have a ceiling.
- **Sandbox transcripts are templated:** real ASR text will need the phrase lists extended or the models retrained.
- **No experiment yet:** impact figures are projections until a randomised trial runs.
- **MSME calibration** is based on only 55 test PTPs.
