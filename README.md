# Phase 2 — Feature Engineering Engine

**What this is:** the code that turns raw call transcripts and account records into the
numbers Team C will train a model on.

**Who it's for:** anyone on the project. No machine-learning background assumed.

---

## 1. What Phase 2 actually has to do, in one paragraph

Phase 1 answers *"did this promise get kept?"* — one yes/no per promise. Phase 3 trains a
model to predict that answer. Phase 2 sits in between and answers a different question:
**"what do we know about this promise that might predict the outcome?"**

Every one of those things is a *feature* — a single number or label describing one promise.
"The borrower hedged three times." "The borrower never said a date." "The promise falls two
days before payday." Stack enough of them into a table, one row per promise, and that table
is what a model learns from.

This code produces that table. It is called the **feature store**.

---

## 2. The one decision everything else follows from

The product scores promises **while the call is still happening**. The agent needs the score
in time to say something before the borrower hangs up.

That has an awkward consequence for how features must be built. Suppose a call has ten turns.
If I only ever compute features from all ten, the model can only ever score a call that has
already ended — useless for a live prompt. So features have to be computable from a
**prefix**: turns 1 to 3, turns 1 to 5, and so on.

Everything in this design follows from that:

> **One function computes features from turns 0 to k. Nothing else is allowed to.**

The streaming table calls it at every borrower turn. The end-of-call table calls it once with
k set to the last turn. There is deliberately no separate "summarise the whole call" code path.

**Why it is worth being strict about.** If training features and live-serving features came
from two different pieces of code, they would drift apart over time — someone fixes a bug in
one and forgets the other. The model would then see slightly different numbers in production
than it saw in training, and quietly get worse. Nobody would get an error message. One
function makes that impossible, and there's a test asserting it.

---

## 3. Three tables, not one

The feature store is three separate files. The split is about **time** — specifically, when
each piece of information becomes known. All three answer the same question ("will this
promise be kept?"), but at different moments, and a model may only learn from information
that will actually exist at the moment it is used.

Take one call:

```
10:00  Bot: "Rs 4,200 is overdue. By which date can you pay?"
10:01  Borrower: "Next week… I have to go."           ← in-call
10:02  Call ends                                       ← end-call
Day 2  Borrower opens the payment link                ← post-call
Day 5  Promised date
```

| Table | One row per | Contains | Who uses it |
|---|---|---|---|
| `features_incall` | promise × borrower turn | only what has been said *so far* in the call | the live mid-call model, which warns the agent before the borrower hangs up; Phase 4 serves this |
| `features_endcall` | promise | everything knowable by the moment the call ends | Team C's main training table |
| `features_postcall` | promise | signals arriving *after* the call, up to the promised date | parking-window / follow-up model only |

### Why the in-call table has one row per turn

A live model has to work at **every point** in the call, not just the end. So each time the
borrower speaks, a new row is built from the turns up to that point only:

```
after turn 1  → features from turn 1 only
after turn 3  → features from turns 1–3
after turn 5  → features from turns 1–5
```

That is why it has about five rows per call (15,616 rows for 3,067 calls). The end-call row is
simply the last of these snapshots, made by the same function (see §2).

### Why the third table is quarantined

`post_call_events.csv` records things like the borrower opening the payment link. That is a
**very** strong predictor — someone who opens the link usually pays.

It is also completely unusable for the in-call model, because at the moment the agent needs a
prompt, it hasn't happened yet. If it leaked into the training data, the model would look
excellent in testing and then fail in production, because the single feature carrying all its
accuracy would always be blank when it mattered.

That's called **leakage**, and it is the most common way a project like this quietly fails.

So the separation is **physical, not a naming convention**. Post-call signals live in their
own file. `pipeline.py` checks on every run that no `post_` column has appeared in the other
two tables and crashes if one has. You cannot leak them by forgetting, only by deliberately
rewriting the guard.

### And `labels`?

`labels.csv` is not a feature table. It is the answer key (kept or broken), so it must never
be given to a model as an input. The pipeline only uses it to work out each agent's past keep
rate from outcomes already known before the call, and it crashes if `kept` ever appears in a
feature table.

---

## 4. What's in the box

### The pipeline, step by step

`python run.py` does this, in order:

1. **Read** every CSV in `csv_folder/` and stop if a required file or column is missing.
2. **Label** each promise as kept or broken. Phase 1's `phase1_labels.csv` is used if present;
   otherwise kept means at least 80% of the amount was paid by the borrower by the end of
   promised date + 3 days.
3. **Conversation features.** For each call: order the turns, clean the text, find phrases
   (`lexicons.py`) and dates/amounts (`extractors.py`), then compute features from turns
   0..k (`features.py`). This gives one row per borrower turn (in-call) and one per call
   (end-call). Field visits get a row with `has_transcript = 0`.
4. **Account features.** Join what was known about the account at the time: overdue amount,
   salary day, past promises.
5. **Agent features.** The agent's track record, using only outcomes already known before
   the call and only train-split accounts.
6. **Post-call features**, in their own table, counting events only up to the promised date.
7. **Safety checks.** Crash if a post-call, label or annotation column reached a feature
   table. Then save everything to `out/`.
8. **Reports.** `audit.py` writes coverage, leakage, feature dictionary and annotation
   reports.

```
csv_folder/ ─► read ─► labels ─┬─► conversation ─┐
                               ├─► account ──────┼─► endcall + incall ─┐
                               ├─► agent ────────┘                     ├─► checks ─► out/ ─► reports
                               └─► post-call ──────────────────────────┘
```

### Main folder

| File | What it does |
|---|---|
| `run.py` | **The file you run.** Reads the data, builds the tables, writes the reports. |
| `requirements.txt` | Libraries needed: pandas, numpy, pyarrow. |
| `README.md` | This file: what the project does, the design decisions, and how to run it. |
| `.gitignore` | Files Git should not upload (Python caches and similar). |
| `tests.py` | *Checking tool.* 32 checks that the pipeline is not silently wrong: "23 days" is not read as ₹23, a payment never keeps two promises, nothing from the future or the test split leaks in, one row per promise. Some use small made-up examples; the rest run the whole pipeline and inspect the real output. |
| `verify_real_samples.py` | *Checking tool.* Runs two real calls through the feature code and checks 19 things a person reading them would notice, e.g. "borrower mentioned hardship", "borrower named no amount". |
| `make_synthetic.py` | *Checking tool.* Writes fake CSVs in the real format, so the code can be tested without the real data (for teammates who may not see it). Its numbers mean nothing. |

The three checking tools are not part of the pipeline: `run.py` never uses them, and the
output is the same without them. Run `tests.py` before every commit, and
`verify_real_samples.py` after changing `lexicons.py` or `extractors.py`.

### The engine: `ptp_features/`

You never run these files directly; `run.py` calls `pipeline.py`, which calls the rest.
Listed in the order the data flows through them.

#### `config.py` — every setting in one place
- **Folders:** where to read (`csv_folder/`) and write (`out/`). Can also be set with the
  `PTP_DATA_DIR` / `PTP_OUT_DIR` environment variables.
- **Label rules:** which payment channels count as the borrower paying (`upi_app`, `upi_link`,
  `bbps`, `neft`, `branch_cash`, `cash_field`, but not `nach_represent`), the 3 grace days,
  and the 80% of the promised amount that must be paid.
- **Feature rules:** at least 20 past promises before an agent's keep rate is shown, money
  only counts as money from ₹100 up unless it has "Rs"/"₹", only train accounts feed agent
  history, post-call events count only up to the promised date.
- **Why it matters:** to change a rule, change it here. No number is hidden in the other files.

#### `lexicons.py` — what the borrower *said*
- **Contains:** 10 phrase lists in English, Hindi and Kannada, grouped by meaning: hedge,
  vague date, conditional, hardship, third party, termination, link refusal, over-agreement,
  refusal, firm commitment. Hardship and refusal are marked as **safety** categories.
- **`normalise()`** cleans a sentence first: lowercase, remove `[NAME]`-style placeholders and
  punctuation, keep `4,200` as `4200`, turn `₹` into `rs`.
- **Counting:** how many phrases from each list appear in a sentence, and which ones.
- **Spoken language:** counts common words per language ("hai", "illa", "the") to see which
  language was actually spoken, whatever the account says.
- **Tools:** `mine_candidate_terms()` suggests new phrases from the data; `coverage_report()`
  measures how many sentences match no list at all.

#### `extractors.py` — dates and money
- **`find_amounts()`:** finds money such as `Rs 12,200`, `4200 rupees`, `5 hazaar`. It rejects
  numbers that aren't money ("7 days", "2 loans", "5 baje"), and digits that belong to a date.
- **`find_dates()`:** finds dates and marks each as **exact** (3 April, kal, friday, 5 tarikh)
  or **vague** (next week, month end, amele).
- **Bot behaviour:** `is_date_question()` spots the bot asking "by which date?";
  `is_minimum_ask()` spots it pushing for "at least" a part payment.

#### `features.py` — the heart: conversation features
- **`prepare_turns()`** takes one call's transcript rows, puts them in order, and runs the
  two files above on every turn once. Phrases are counted on borrower turns only; dates and
  amounts on both sides, because *who said it first* matters.
- **`extract_prefix_features(records, k)`** turns those tags into about 125 numbers, using
  turns 0..k only. Groups:
  - *Volume:* turns so far, how much the borrower said.
  - *Phrases:* count, yes/no, rate, variety, and "in the latest turn", for each list.
  - *Language:* which language dominated and how mixed it was.
  - *Structure:* did the borrower ever state a date or amount, who said it first, the
    promise turn, counteroffers, back-and-forth.
  - *Timing:* pauses, speech rate, speech-to-text confidence, especially on the promise turn.
- The same function makes both the in-call rows and the end-call row (see §2).

#### `context.py` — account and agent features
- **`build_account_features()`** (columns start with `acct_`): joins `accounts.csv` and
  `lenders.csv` to each promise. Promise size vs overdue, EMI and ability to pay; days
  before salary day (counted around the month end); past promises and how many broke; time
  and day of the call. Missing inputs stay blank, never 0.
- **`build_agent_features()`** (columns start with `agent_`): joins `agents.csv`, then the
  agent's track record from **before** the call only: past keep rate, promises per connected
  call, share of promises logged after 6 pm, share of copy-pasted remarks. Only train-split
  accounts count.

#### `pipeline.py` — the conductor
- **`load_raw()`** reads the CSVs and checks every required column is there.
- **`load_or_build_labels()`** uses Phase 1's labels if present, otherwise works out kept or
  broken from `payments.csv`. Each payment counts toward one promise only.
- **`build_conversational_tables()`** links each promise to its call and builds the in-call
  and end-call rows.
- **`build_postcall_features()`** counts link opens, replies and callbacks up to the
  promised date.
- **`run()`** joins everything, runs the safety checks (no post-call, label or annotation
  column in a feature table), and saves each table as CSV and Parquet.

#### `audit.py` — the reports
- Sorts every column into *pre-call*, *in-call*, *post-call* or *identifier*.
- Writes `coverage_report`, `leakage_audit`, `feature_dictionary` and, when
  `annotated_ptp_sample.csv` is present, `annotation_check` (see §7).

### Input: `csv_folder/`

| File | Contains |
|---|---|
| `ptps.csv` | **One row per promise:** who, how much, by when, call or field visit. Everything is built around it. |
| `call_transcripts.csv` | Each call turn by turn, with timings and speech-to-text confidence. |
| `accounts.csv` | Per account: overdue, EMI, salary day, past promises, credit band. |
| `agents.csv` | Per agent or bot: channel, language team, experience. |
| `dial_attempts.csv` | Every call attempt and agent remark. Used for agent features. |
| `payments.csv` | Every payment and channel. Used for the labels. |
| `post_call_events.csv` | Link sent/opened, WhatsApp replies, callbacks. |
| `splits.csv` | Train / validation / test, per account. |
| `lenders.csv` | Lender type. |
| `annotated_ptp_sample.csv` | *(Optional)* 200 promises labelled by type. Used only to check features, never as one. |

### Output: `out/`

Rebuilt on every run.

| File | Contains |
|---|---|
| `features_endcall` | One row per promise, everything known at hangup. Team C's main training table. |
| `features_incall` | One row per borrower turn, only what was known at that moment. For the live model. |
| `features_postcall` | Post-call signals up to the promised date. Follow-up model only. |
| `labels` | Kept or broken, plus `label_source`. The answer key, never a model input. |
| `coverage_report` | How much data was covered, phrase coverage per language, emptiest features. |
| `leakage_audit.md` | Checks that nothing from the future or the test set leaked in. |
| `feature_dictionary` | Every column: meaning, when available, blank rate, caveats. |
| `annotation_check` | Features compared with the human labels (train + validation only). |

Tables are written as `.csv` and `.parquet`, reports as `.md` (some also `.json`).

### Running it

```bash
pip install -r requirements.txt
python run.py                      # reads csv_folder/, writes out/
python tests.py                    # 32 tests against the real data
python verify_real_samples.py      # 19 checks on two real calls
```

`--data-dir` and `--out-dir` override the folders. Takes about ten seconds. Writes the
three tables plus the label table (CSV and Parquet), and the reports, all into `out/`.

To test without the real data:

```bash
python make_synthetic.py --out-dir data_synthetic
python tests.py --data-dir data_synthetic
```

`annotated_ptp_sample.csv` is optional. When it is present, the run also writes
`annotation_check.md` (see §7). It is a human label, so it is only ever used to *check*
features and never enters a feature table. The pipeline crashes if it ever does.

---

## 5. The features, in plain words

About 180 columns. They fall into six families.

**How text becomes numbers.** `lexicons.py` and `extractors.py` decide nothing on their own.
They only *tag* each turn: "1 hedge, 1 termination cue", "vague date: next week, no
amount". `features.py` then combines the tags across turns 0..k into numbers: counts
(`lex_hedge_count`), yes/no flags (`cust_stated_any_amount`), who said something first
(`first_amount_speaker`), and the promise turn itself (the first borrower turn with an exact
date or an amount), whose pause and speech confidence become features too. A model then
learns which combinations predict a broken promise.

### 5a. What the borrower said — the lexicons

Dictionaries of phrases grouped by what they mean, not what language they're in:

| Category | What it catches | Examples across the three languages |
|---|---|---|
| `hedge` | non-committal softening | *I'll try*, *dekhta hoon*, *nodtini* |
| `vague_date` | a date you can't put in a calendar | *next week*, *agle hafte*, *munde vaara* |
| `conditional` | promise depends on something else | *if I get paid*, *salary aayi toh*, *party payment kottre* |
| `hardship` | genuine inability | *business down*, *paise nahi hai*, *business tumba down ide* |
| `third_party` | not actually the borrower | *he's not here*, *bhai ne*, *avaru illa* |
| `termination` | trying to end the call | *I have to go*, *mujhe jaana hai*, *naanu hogbeku* |
| `link_refusal` | refusing the payment link | *don't send it*, *link mat bhejo*, *link beda* |
| `over_agreement` | agreeing too readily | *okay okay*, *haan haan*, *haudu haudu* |
| `refusal` | explicit no | *not possible*, *nahi hoga*, *aagalla* |
| `concrete_commitment` | firm language | *definitely*, *pakka*, *khanditha* |

**The important design decision.** The example code you were given looked up the borrower's
`preferred_language` and applied only that one dictionary. **That is a bug**, and it would
have been invisible — no error, just quietly missing most of the signal.

Two reasons. First, `preferred_language` is what the borrower *registered* with, not what
they actually spoke. Second, the transcripts are code-mixed: real borrowers mix Hindi,
Kannada and English inside a single sentence. A borrower tagged `english` saying *"dekhta
hoon"* would score zero hedging under the gated version.

So **every dictionary runs on every utterance.** `preferred_language` is kept as a feature
the model can use; it is never used as a filter. There's a test that asserts the function
doesn't even *accept* a language argument, so the bug can't be reintroduced.

Lexicons only run on **borrower** turns. The bot's lines are a fixed script — counting them
would add the same constant to every call.

### 5b. Who said what first — the structural features

This is where the most decision-relevant signal lives.

- **Who first said a date?** Borrower or bot.
- **Who first said an amount?** If the bot proposed the number and the borrower just agreed,
  that's a different promise from one the borrower offered.
- **Did the borrower *ever* state a date or an amount?** — `cust_stated_neither_date_nor_amount`

That last one is close to the whole agent-pushed detector. If across an entire call the
borrower never uttered a date or a figure, then whatever got logged as a promise was
constructed by the agent or the bot. The borrower never made it. No model needed — it's a
fact about the transcript.

Also here: did the borrower counteroffer a different number (compared with the house figure
*in force at that moment*, not the bot's opening overdue statement, so a borrower who takes
the bot's minimum ask is agreeing rather than counteroffering); how many real back-and-forth
exchanges happened; where in the call the promise landed; and whether the promise arrived
immediately after the bot pushed for a minimum payment.

### 5c. Timing — standing in for the audio we don't have

The planning document asks Phase 3 to analyse "live audio streams" for hesitation and stress.
**There is no audio.** `call_transcripts.csv` is text only.

**This needs flagging to whoever wrote the plan.** It kills the PyTorch multi-modal branch of
Phase 3 as written. It also removes the DPDP voice-biometric exposure the brief warns about —
so it saves effort *and* reduces compliance risk.

What we have instead, per turn: `pause_before_ms`, `start_ms`, `end_ms`, `asr_confidence`.
That's enough to proxy hesitation:

- **Pause before the promise turn**, z-scored against that borrower's own pauses in the same
  call — so it's comparable between naturally fast and naturally slow talkers.
- **Latency to the date question** specifically. "By which date can you pay?" is the single
  most important question in the call; how long someone takes to answer it is informative in
  a way other pauses aren't.
- **Speech rate** — characters per second of speaking time.
- **ASR confidence dip** on the promise turn. Mumbled, hesitant speech transcribes worse.

### 5d. Account context

Mostly joins from `accounts.csv`, plus a few derived ratios. The one worth understanding:

**`acct_promise_before_payday`.** `accounts.csv` has `salary_credit_day`. If a borrower
promises money three days *before* their salary lands, that promise is structurally hard to
keep regardless of how sincere they sound.

The arithmetic has to wrap around the month. A promise on the 29th against a salary day of
the 2nd is **three days before payday**, not twenty-seven days after it. Getting this wrong
would invert the flag for every promise near a month boundary. There's a test for it.

This is also the **zero-model quick win**: it needs no training data, no labels, and is fully
explainable to an auditor in one sentence.

Two rules hold across all account flags:

- **Unknown stays unknown.** 69% of accounts have no `salary_credit_day`. For those,
  `acct_promise_before_payday` is blank (`<NA>`), not 0. A 0 would tell the model "known
  not to be before payday". The same applies to the part-payment, repeat-promiser and
  late-shift flags.
- **Prefer running history to snapshots.** `prev_ptp_count` is a snapshot from the start of
  the window (see §7), so it misses promises made inside the window.
  `acct_n_prior_ptps_in_data` counts the account's earlier promises in `ptps.csv`, strictly
  before this one.

### 5e. Agent context — where leakage hides

"Agent TA005's historical promise-to-payment rate" is a powerful feature and a trap.

The naive version — `groupby('agent_id').mean()` — uses outcomes from the *future*. Two
guards are applied:

1. **Time.** A statistic attached to a call on 12 April uses only promises whose outcome was
   already *knowable* by 12 April. A promise due next Tuesday tells you nothing today, even
   though it sits earlier in the file. "Knowable" means the end of the grace window: midnight
   after promised date + 3 days. The label uses the same window, defined once in
   `config.py`.
2. **Split.** Only training-set accounts contribute history at all, so nothing crosses from
   validation or test into a training row.

Guard 1 is the one most pipelines get wrong. There's a test that recomputes the rate from
scratch for sixty promises and checks the pipeline matches to nine decimal places.

A subtler version of the same trap: "promises per connected call, before this call" must
measure from when the call **started**, not when the promise was logged a few minutes later.
Otherwise the call's own promise counts as history. The pipeline uses the call's own
dial-attempt timestamp.

Also here, the two integrity signals from the brief: **late-shift bunching** (promises
clustered after 6pm — what hitting a daily target looks like in data) and **remark reuse**
(the share of an agent's notes that are copy-pastes). Both computed over the agent's
*previous* days only, never their current one.

### 5f. Post-call engagement

Link opened, WhatsApp confirmation replied to (yes *or* no), borrower called back.
Quarantined, per §3.

These are also **cut off at the promised date**. In the real data, 29% of link opens happen
on or after the promised date, usually because the borrower opened the link to pay. Counting
those would give the parking-window model the answer. Only events between capture and the
promised date count (`postcall_horizon_days` in `config.py`).

---

## 6. What the tests found

The first version had 22 pipeline tests and 19 checks against real transcript rows from your
actual data (calls `AT0000026` and `AT0000051`). **They caught four real bugs**, which is the
point of writing them. A later review on the full real data found more. Both rounds are
below.

### Bug 1 — the serious one: `Rs 12,200` was being read as `Rs 12`

Text normalisation strips punctuation. Commas are punctuation. So `4,200` became `4 200`,
and the amount detector read two separate numbers: `4` (discarded as too small) and `200`.

Your real transcripts use comma grouping throughout — `4,200`, `12,200`, `2,800`. **Every
grouped amount would have been silently corrupted.** No error, no warning. The amount
features would just have been wrong, and Team C would have spent a week wondering why they
carried no signal.

Fixed by joining digit groups *before* stripping punctuation, and by preserving separators
that sit between two digits so dates and decimals survive intact.

### Bug 2 — year read as a rupee amount

Having preserved `-` between digits for dates, `2026-04-29` then produced an amount of
Rs 2,026. Fixed by masking date spans before scanning for money: a number inside a date is
not money.

### Bug 3 — decimals truncated

`4200.50` came out as `4200`. Minor, now correct.

### Bug 4 — a hardship phrase the lexicon missed

This one only showed up against the **real** transcript. The actual borrower said
*"business tumba down ide"*. My dictionary had `"business down ide"` — but *tumba* ("very")
splits the phrase, so it didn't match. The same hole would have hit Hindi *"business bahut
down hai"*.

This matters more than the others because **hardship drives the safety gate**. A missed
hardship cue means a struggling borrower could be routed to harsher treatment — exactly the
outcome the brief says must never happen. Fixed by adding intensifier-tolerant forms.

**The general lesson:** the hand-written dictionaries are a *seed*, not the finished article.
`lexicons.mine_candidate_terms()` ranks phrases from the training split by how strongly they
separate kept from broken promises. Run it, review the output with someone who knows
collections, and fold confirmed terms back in. The coverage report tracks progress: it prints
the share of borrower utterances where *no* category fired, broken down by language.

### Second round: review against the full real data

Running on the real CSVs (3,696 promises, 39,194 transcript turns) turned up the following.
Each is fixed and has a test.

**Bug 5 — the label ignored the most common payment channel.** The fallback "kept" label
counted only `upi_app`, `bbps`, `neft` and `branch_cash`. It left out `upi_link` (paying
through the link sent on the call: 744 payments, the **largest** channel) and `cash_field`
(cash handed to a field agent). The kept rate went from **10.9% to 24.8%**. Every agent
kept-rate feature sits on this label. `nach_represent` is still excluded, on purpose.

**Bug 6 — one payment could keep two promises.** When a borrower re-promised before the first
promise fell due, a single payment landing in both windows marked both as kept (226 promises
overlap like this; 17 payments were double-counted). Each payment now counts toward one
promise only: the most recent one whose window contains it.

**Bug 7 — the grace window ended a day early.** "Promised date + 3 days" was computed as
midnight at the *start* of day 3, so payments made during that day did not count (118 payments
in the data). The window now runs through the end of the last grace day.

**Bug 8 — `agent_id` appeared twice as a feature.** The agent table and the account table
both carried it, so the join produced `agent_id_x` and `agent_id_y`. The dictionary then
labelled both as `pre_call` model inputs. Now there is one `agent_id`, classed as an
identifier.

**Bug 9 — "unknown" became "no".** Flags like `acct_promise_before_payday` were 0 when the
salary day was missing (2,534 promises). They now stay blank (see §5d).

**Bug 10 — the ₹ sign was deleted.** 2,529 transcript turns write amounts as `₹8,800`.
Punctuation stripping removed the `₹`, so small amounts like `₹50` were lost. `₹` is now
read as a currency marker.

**Bug 11 — post-call features saw the outcome.** See §5f.

**Bug 12 — the call's own promise counted as agent history.** See §5e.

**Lexicon and date fixes.** One Kannada term (`maga`) had a Cyrillic `а` in it, so it could
never match. `ivattte` was a typo. Bare `toh` (a Hindi filler) was counting as
"conditional", and bare `beda` ("don't want") as "link refusal". Both were removed; the
specific phrases (`salary aayi toh`, `link beda`) stay. The weekday short forms `sun`, `sat`,
`mon` and `wed` were counting as concrete dates, but in code-mixed speech they are usually
words ("sun lo" means "listen"), so they were removed.

**Counteroffer.** See §5b.

**Test bugs.** The field-visit test looked for `source == "visit"`, but the real data says
`field_visit`, so it failed on real data. It now uses `visit_id`. The "end-call row equals
final prefix" test compared the extractor with itself, so it could never fail. It now
checks rows the pipeline actually wrote against a fresh recomputation from the transcript.
`make_synthetic.py` crashed on Windows (`strftime("%-d")`).

**Smaller.** Parquet errors are reported instead of silently skipped. `run.py` reads the CSVs
once, not twice. The default data folder is `csv_folder`. Agent bunching and remark-reuse
use the agent's most recent earlier day, even when the agent had no activity on the call day. The leak check reads train labels
only, so the audit never looks at test outcomes.

### Verification status, honestly stated

| | |
|---|---|
| Logic, joins, leakage guards, as-of arithmetic | **Verified** — 32 tests, passing on both the real data and synthetic data |
| Lexicons, date/amount extraction, timing features | **Verified against real rows** — 19 checks on two actual calls |
| Coverage rates, language mix, whether `prev_ptp_count` leaks | **Measured on the real data** — see §7 |
| Features vs. human taxonomy | **Measured** on 173 annotated train/validation promises — see §7 |

Every number in the reports now comes from the real CSVs in `csv_folder/`.

---

## 7. The reports it generates

All reports are written to `out/` on every run. They are generated from the code and data,
so edit the code, not the report files.

### `coverage_report.md`
How much data the pipeline reached: how many promises have transcripts, how many are
field-sourced with none, turns per call, decision points generated. Then the **lexicon
coverage** section — the share of borrower utterances where no cue fired, overall and per
language. That's the number tracking the "100% language coverage" objective. If one language
is much worse, the dictionary has a hole there.

### `leakage_audit.md`
The main items:

1. Confirmation the post-call quarantine held, and that no label or annotation column
   reached a feature table.
2. **An empirical test of whether `accounts.csv` columns are as-of-capture.** `bucket_start`,
   `dpd_start` and `overdue_start` say "start" in the name. `prev_ptp_count`,
   `prev_ptp_broken` and `ability_to_pay_estimate` don't — so nobody knows whether they were
   computed at the start of the window or over the whole thing. If it's the whole window,
   **they leak**. This test compares the column against a running count recomputed from
   `ptps.csv` and prints a verdict. On the real data: **safe snapshot** (see below).
3. A **leak canary** — any conversational feature correlating with the label above
   |r| = 0.6 on the **train split** gets flagged for a human to look at. A smoke alarm, not a
   rule; a genuinely strong feature can trip it.
4. The agent-history settings, and the post-call cut-off date.

### `feature_dictionary.csv` / `.md`
Every feature with its group, **time-availability class**, null rate, an **as-of caveat**,
and a plain-English definition. Generated from the code, so it cannot drift from what
actually shipped. The caveat column marks every feature built on an `accounts.csv` field
whose timing is undocumented (`ability_to_pay_estimate`, `bureau_score_band`, the
`prev_ptp_*` family and so on), so Team C knows which ones to treat with care.

Team C needs this more than they need extra features.

### `annotation_check.md` (new)
Compares the features with the human taxonomy in `annotated_ptp_sample.csv` (200 promises).
It uses only the 173 in train and validation. The 27 test-split promises are excluded so
nobody tunes lexicons on the test set. It shows three things:

1. **Kept rate by class.** This sanity-checks the label: `genuine_feasible` is kept 55% of the
   time, against 0–25% for every other class.
2. **A simple rule per class**, with recall and precision. For example, "answerer not self
   or third-party cue" finds 75% of `third_party` promises at 86% precision.
3. **Feature means by class**, side by side.

### What the real data says

- **Reach.** 3,067 of 3,696 promises (83%) have a transcript. 605 are field visits with no
  call. 24 are call-sourced but their call has no transcript rows.
- **Lexicon coverage is the weak spot.** 54% of borrower turns trigger no cue category (50%
  Kanglish, 54% English, 56% Hinglish). That is the main target for `mine_candidate_terms`.
- **`prev_ptp_count` does not leak.** It is constant per account (a window-start snapshot)
  and uncorrelated with the account's total promises (r = 0.02). It is safe, but it goes
  stale inside the window; `acct_n_prior_ptps_in_data` fills that gap.
- **Leak canary:** nothing above |r| = 0.6 on the train split.
- **From the annotations:**
  - The agent-pushed detector (`cust_stated_neither_date_nor_amount`) finds only 58% of
    annotated `agent_recorded_or_pushed` promises. The other 42% are borrowers who said only
    a *vague* date ("next week"), which currently counts as "stated a date". "No concrete
    date and no amount" is worth trying as the detector.
  - The escape rule fires on 120 of 173 promises (precision 22%). Termination cues like
    "ok ok" are common in every class, so on their own they don't separate classes.
  - `repeat_promiser` is found only 42% of the time from the accounts snapshot.

---

## 8. Three things to raise with the group

**1. There is no audio.** Phase 3 as written plans a PyTorch multi-modal model over "live
audio streams". The data is text plus timing. Raise it now, not in week four.

**2. Agent features are double-edged.** Including agent identity makes the model partly
predict *the agent* rather than *the promise*, and the brief explicitly requires the scorer
stay independent of the agents it measures. They're all grouped under the `agent_` prefix so
Team C can ablate them in one line. Ask for performance reported both ways.

**3. Phase 1's input list is missing `accounts.csv`.** The planner lists only `ptps`,
`payments`, `post_call_events` and `annotated_ptp_sample` for Phase 1 — but the taxonomy
rules need `salary_credit_day` (for *genuine_infeasible*) and `prev_ptp_broken` (for
*repeat_promiser*), both of which live in `accounts.csv`. Either Phase 1 pulls from `shared/`
too, or those rules move to Phase 2.

---

## 9. What Team C receives

- `features_endcall.parquet` — main training table, one row per promise
- `features_incall.parquet` — streaming table for the mid-call model
- `features_postcall.parquet` — quarantined, parking-window model only
- `labels.parquet` — the label behind the agent-history features; `label_source` says
  whether it came from Phase 1 or the Team B fallback. A Phase 1 file
  (`phase1_labels.csv` in `csv_folder/` or `out/`) always wins.
- `feature_dictionary.csv` — what every column means, when it's available, and any as-of caveat
- `coverage_report.md`, `leakage_audit.md`, `annotation_check.md`

Two notes to pass on with it:

- **Use `splits.csv` as given.** It's account-level, which is deliberate — the same account
  can have several promises, and splitting at promise level would let an account's history
  appear in both training and test. Don't re-split.
- **Also run a time-based holdout** (train on the earlier months, test on the later ones)
  alongside the given split. If the two disagree sharply, there's drift — and that's a
  finding worth reporting rather than hiding.

---

## 10. Next steps

1. ~~Run it on the real CSVs.~~ Done: outputs and reports are in `out/`.
2. **Close the lexicon gap.** 54% of borrower turns trigger no cue. Run
   `mine_candidate_terms` on the train split, starting with Hinglish (56% uncovered), and
   have a collections SME review the candidates.
3. ~~Tell Team C whether `prev_ptp_count` leaks.~~ It doesn't: it's a window-start
   snapshot. Point them to `acct_n_prior_ptps_in_data` as well.
4. **Agree the label with Phase 1.** Confirm the channel list (now including `upi_link` and
   `cash_field`), the 3-day grace window, and one-payment-per-promise. Once Phase 1 publishes
   `phase1_labels.csv`, re-run; it replaces the fallback automatically.
5. **Try a tighter agent-pushed detector** (no concrete date and no amount) and check it
   against `annotation_check.md`.
6. Hand Team C the three tables, the dictionary and the annotation check.
7. Raise the three points in §8.
