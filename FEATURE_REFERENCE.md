# Feature Reference

Every column in every output file in `out/`, in plain words.

| File | One row per | Rows | Columns | Use |
|---|---|---|---|---|
| `features_endcall.csv` | promise | 3,696 | 183 | Team C's main training table: everything known when the call ended |
| `features_incall.csv` | borrower turn | 15,616 | 184 | Live mid-call model: only what was known at that turn |
| `features_postcall.csv` | promise | 3,696 | 17 | Follow-up model only: signals after the call, up to the promised date |
| `labels.csv` | promise | 3,696 | 5 | The answer key. **Never** a model input |

**The end-call and in-call tables have the same feature columns.** The only differences:
end-call has `has_transcript`; in-call has `decision_point` and `is_final_turn`. In the
in-call table every conversation column is computed **only from the turns up to that row's
borrower turn**; in the end-call table, from the whole call.

**How to read the tables below**

- **Type:** `0/1` = yes/no flag · `0–1` = share or rate · `ms` = milliseconds ·
  `₹` = rupees · `category` = text label · `number` = count or ratio.
- **Blank** means "unknown" or "not applicable", not 0. For example, every conversation
  column is blank for field visits, and promise-turn columns are blank until the promise
  turn happens.
- **"Borrower"** = the customer; **"house"** = the lender's side (bot or agent).
- **⚠ As-of date undocumented** = comes from an `accounts.csv` field whose timing is not
  documented. Use with care; see `out/leakage_audit.md`.
- Columns are grouped by topic below; the order inside the CSV files differs.

---

## 1. `features_endcall.csv` and `features_incall.csv`

### 1.1 Identifiers and bookkeeping (not model inputs)

| Column | Type | What it means |
|---|---|---|
| `ptp_id` | text | Promise ID from `ptps.csv`. The key that joins every table. |
| `call_id` | text | The call this promise came from (= `attempt_id` in `ptps.csv`). Blank for field visits. |
| `account_id` | text | Loan account the promise belongs to. |
| `agent_id` | text | Agent or bot that logged the promise (TA = tele agent, FA = field agent, BOT = voice bot). |
| `split` | text | `train`, `validation` or `test`, from `splits.csv`. Decided per account. |
| `has_transcript` | 0/1 | 1 if the promise has a call transcript. 0 for field visits and the 24 calls with no transcript rows; their conversation columns are blank. *(end-call only)* |
| `decision_point` | number | Which borrower turn this row is: 1 = after the borrower's 1st turn, 2 = after the 2nd, and so on. *(in-call only)* |
| `is_final_turn` | 0/1 | 1 if this borrower turn was the last turn of the whole call. *(in-call only)* |

### 1.2 How much was said

Known: at each turn (in-call) or at hangup (end-call).

| Column | Type | What it means |
|---|---|---|
| `k_turn_idx` | number | Turn number (`turn_idx`) of the last turn included. In the end-call table, the call's last turn. |
| `n_turns_so_far` | number | Turns so far, both speakers. |
| `n_customer_turns` | number | Borrower turns so far. |
| `n_house_turns` | number | Bot or agent turns so far ("house" = the lender's side). |
| `call_elapsed_ms` | ms | Time from call start to the end of the latest turn. |
| `cust_chars_total` | number | Characters the borrower has said (after cleaning). |
| `cust_words_total` | number | Words the borrower has said. |
| `cust_chars_per_turn` | number | Average characters per borrower turn. |
| `cust_words_per_turn` | number | Average words per borrower turn. |

### 1.3 What the borrower said: phrase lists

Phrases are counted on **borrower turns only** (the bot's lines are a fixed script). Each of
the 10 phrase lists produces the same five columns:

| Suffix | Type | Meaning |
|---|---|---|
| `_count` | number | Total phrases from this list so far |
| `_any` | 0/1 | 1 if any phrase from this list so far |
| `_rate` | number | Phrases per borrower turn |
| `_distinct` | number | How many *different* phrases from this list |
| `_in_last_turn` | 0/1 | 1 if the latest borrower turn had one |

The 10 lists:

| List | Catches |
|---|---|

| `lex_hedge_*` | non-committal softening (I'll try, dekhta hoon, nodtini) |
| `lex_vague_date_*` | a date you can't put in a calendar (next week, agle hafte, amele) |
| `lex_conditional_*` | promise depends on something else (if I get paid, salary aayi toh, kottre) |
| `lex_hardship_*` | genuine inability: job loss, illness, business down (paise nahi hai, kashta) |
| `lex_third_party_*` | speaker is not the borrower or speaks for someone (my husband, avaru illa) |
| `lex_termination_*` | trying to end the call (I have to go, busy, phone rakh) |
| `lex_link_refusal_*` | refusing the payment link (don't send, link mat bhejo, link beda) |
| `lex_over_agreement_*` | agreeing too readily (okay okay, haan haan, sari sari) |
| `lex_refusal_*` | explicit no (not possible, nahi hoga, aagalla) |
| `lex_concrete_commitment_*` | firm language (definitely, pakka, khanditha) |

Every column, plus the two combined columns:

| Column | Type | What it means |
|---|---|---|
| `lex_hedge_count` | number | Total hedge phrases said by the borrower so far. |
| `lex_hedge_any` | 0/1 | 1 if the borrower used any hedge phrase so far. |
| `lex_hedge_rate` | number | Hedge phrases per borrower turn (count ÷ borrower turns). |
| `lex_hedge_distinct` | number | How many *different* hedge phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_hedge_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a hedge phrase. |
| `lex_vague_date_count` | number | Total vague date phrases said by the borrower so far. |
| `lex_vague_date_any` | 0/1 | 1 if the borrower used any vague date phrase so far. |
| `lex_vague_date_rate` | number | Vague date phrases per borrower turn (count ÷ borrower turns). |
| `lex_vague_date_distinct` | number | How many *different* vague date phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_vague_date_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a vague date phrase. |
| `lex_conditional_count` | number | Total conditional phrases said by the borrower so far. |
| `lex_conditional_any` | 0/1 | 1 if the borrower used any conditional phrase so far. |
| `lex_conditional_rate` | number | Conditional phrases per borrower turn (count ÷ borrower turns). |
| `lex_conditional_distinct` | number | How many *different* conditional phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_conditional_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a conditional phrase. |
| `lex_hardship_count` | number | Total hardship phrases said by the borrower so far. |
| `lex_hardship_any` | 0/1 | 1 if the borrower used any hardship phrase so far. |
| `lex_hardship_rate` | number | Hardship phrases per borrower turn (count ÷ borrower turns). |
| `lex_hardship_distinct` | number | How many *different* hardship phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_hardship_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a hardship phrase. |
| `lex_third_party_count` | number | Total third party phrases said by the borrower so far. |
| `lex_third_party_any` | 0/1 | 1 if the borrower used any third party phrase so far. |
| `lex_third_party_rate` | number | Third party phrases per borrower turn (count ÷ borrower turns). |
| `lex_third_party_distinct` | number | How many *different* third party phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_third_party_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a third party phrase. |
| `lex_termination_count` | number | Total termination phrases said by the borrower so far. |
| `lex_termination_any` | 0/1 | 1 if the borrower used any termination phrase so far. |
| `lex_termination_rate` | number | Termination phrases per borrower turn (count ÷ borrower turns). |
| `lex_termination_distinct` | number | How many *different* termination phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_termination_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a termination phrase. |
| `lex_link_refusal_count` | number | Total link refusal phrases said by the borrower so far. |
| `lex_link_refusal_any` | 0/1 | 1 if the borrower used any link refusal phrase so far. |
| `lex_link_refusal_rate` | number | Link refusal phrases per borrower turn (count ÷ borrower turns). |
| `lex_link_refusal_distinct` | number | How many *different* link refusal phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_link_refusal_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a link refusal phrase. |
| `lex_over_agreement_count` | number | Total over agreement phrases said by the borrower so far. |
| `lex_over_agreement_any` | 0/1 | 1 if the borrower used any over agreement phrase so far. |
| `lex_over_agreement_rate` | number | Over agreement phrases per borrower turn (count ÷ borrower turns). |
| `lex_over_agreement_distinct` | number | How many *different* over agreement phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_over_agreement_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a over agreement phrase. |
| `lex_refusal_count` | number | Total refusal phrases said by the borrower so far. |
| `lex_refusal_any` | 0/1 | 1 if the borrower used any refusal phrase so far. |
| `lex_refusal_rate` | number | Refusal phrases per borrower turn (count ÷ borrower turns). |
| `lex_refusal_distinct` | number | How many *different* refusal phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_refusal_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a refusal phrase. |
| `lex_concrete_commitment_count` | number | Total concrete commitment phrases said by the borrower so far. |
| `lex_concrete_commitment_any` | 0/1 | 1 if the borrower used any concrete commitment phrase so far. |
| `lex_concrete_commitment_rate` | number | Concrete commitment phrases per borrower turn (count ÷ borrower turns). |
| `lex_concrete_commitment_distinct` | number | How many *different* concrete commitment phrases were used. Five different ones is a stronger signal than one repeated. |
| `lex_concrete_commitment_in_last_turn` | 0/1 | 1 if the borrower's most recent turn had a concrete commitment phrase. |
| `lex_hedge_minus_commitment` | number | Hedge count minus firm-commitment count. Above 0 means more wobbling than commitment. |
| `safety_signal_any` | 0/1 | 1 if any hardship or refusal phrase appeared. **Safety flag:** such a borrower must be routed to help or restructuring, never pushed harder. |

### 1.4 Language actually spoken

Counts common words of each language in the borrower's speech, whatever the account's
registered language says.

| Column | Type | What it means |
|---|---|---|
| `obs_lang_hindi_markers` | number | Common Hindi words heard from the borrower (e.g. hai, nahi, abhi). |
| `obs_lang_hindi_share` | 0–1 | Hindi share of all language marker words. |
| `obs_lang_kannada_markers` | number | Common Kannada words heard from the borrower (e.g. ide, illa, naanu). |
| `obs_lang_kannada_share` | 0–1 | Kannada share of all language marker words. |
| `obs_lang_english_markers` | number | Common English words heard from the borrower (e.g. the, will, please). |
| `obs_lang_english_share` | 0–1 | English share of all language marker words. |
| `obs_lang_dominant` | category | Language the borrower mostly spoke: `hindi`, `kannada`, `english`, or `unknown` (no markers). Based on what was said, not the account's registered language. |
| `obs_lang_mix` | 0–1 | How mixed the languages were. 0 = one language only; higher = more code-mixing. |

### 1.5 Dates, amounts and who said them first

Dates and amounts are found on **both** sides, because who proposed them matters. The
**promise turn** is the first borrower turn containing an exact date or an amount.

| Column | Type | What it means |
|---|---|---|
| `n_cust_date_mentions` | number | Dates the borrower mentioned (exact + vague). |
| `n_cust_concrete_date_mentions` | number | Exact dates the borrower mentioned (3 April, kal, friday). |
| `n_cust_vague_date_mentions` | number | Vague dates the borrower mentioned (next week, month end). |
| `n_cust_amount_mentions` | number | Money amounts the borrower mentioned. |
| `n_house_date_mentions` | number | Dates the bot/agent mentioned. |
| `n_house_amount_mentions` | number | Money amounts the bot/agent mentioned. |
| `cust_stated_any_date` | 0/1 | 1 if the borrower stated any date. |
| `cust_stated_concrete_date` | 0/1 | 1 if the borrower stated an exact date. |
| `cust_stated_vague_date_only` | 0/1 | 1 if the borrower gave only vague dates, never an exact one. Typical of escape promises. |
| `cust_stated_any_amount` | 0/1 | 1 if the borrower stated an amount. |
| `cust_stated_neither_date_nor_amount` | 0/1 | 1 if the borrower never stated a date or an amount. **Agent-pushed detector:** the "promise" was made up by the bot/agent. |
| `first_date_speaker` | category | Who mentioned a date first: `customer`, `house` or `none`. |
| `first_concrete_date_speaker` | category | Who mentioned an exact date first: `customer`, `house` or `none`. |
| `first_amount_speaker` | category | Who mentioned an amount first. `house` = the number was proposed to the borrower, not offered by them. |
| `house_asked_date` | 0/1 | 1 if the bot/agent asked "by which date can you pay?" (or similar). |
| `n_house_date_questions` | number | How many times the bot/agent asked for a date. |
| `house_made_minimum_ask` | 0/1 | 1 if the bot/agent pushed for a minimum or part payment ("at least", "kam se kam", "kanishta"). |
| `turns_since_date_question` | number | Turns since the first date question. Blank if none was asked. |
| `cust_answered_date_q_with_concrete` | 0/1 | 1 if the borrower's reply to the first date question had an exact date. Blank if no question or no reply yet. |
| `latency_to_date_question_ms` | ms | How long the borrower paused before answering the first date question. The most informative pause in the call. |
| `promise_seen` | 0/1 | 1 once a **promise turn** has happened: the first borrower turn with an exact date or an amount. |
| `promise_turn_idx` | number | Turn number of the promise turn. Blank if none. |
| `promise_position_ratio` | 0–1 | Where the promise turn sits in the call so far: 0 = start, 1 = latest turn. Promises at the very end often go with trying to hang up. |
| `turns_after_promise` | number | Turns since the promise turn. |
| `promise_follows_minimum_ask` | 0/1 | 1 if the turn just before the promise was the bot/agent pushing for a minimum. The number may be theirs, not the borrower's. |
| `promise_follows_date_question` | 0/1 | 1 if the turn just before the promise was a date question. |
| `cust_counteroffered` | 0/1 | 1 if the borrower named a different amount from the house figure in force at that moment. Real negotiation. |
| `cust_to_house_amount_ratio` | number | Borrower's first amount ÷ the house amount just before it. Below 1 = borrower offered less. |
| `n_negotiation_exchanges` | number | Borrower turns with a date or amount after the house first proposed one. Back-and-forth. |
| `cust_max_amount_stated` | ₹ | Largest amount the borrower said. |
| `cust_min_amount_stated` | ₹ | Smallest amount the borrower said. |
| `cust_first_amount_stated` | ₹ | First amount the borrower said (where they opened). |
| `cust_last_amount_stated` | ₹ | Last amount the borrower said (where they landed). |

### 1.6 Timing and hesitation

There is no audio, so pauses, speaking speed and speech-to-text confidence stand in for
hesitation. Columns ending `_on_promise` are blank until the promise turn happens.

| Column | Type | What it means |
|---|---|---|
| `cust_pause_mean_ms` | ms | Average silence before the borrower's turns. |
| `cust_pause_max_ms` | ms | Longest silence before a borrower turn. |
| `cust_pause_min_ms` | ms | Shortest silence before a borrower turn. |
| `cust_pause_std_ms` | ms | How much the borrower's pauses vary. Needs 2+ turns. |
| `cust_pause_last_ms` | ms | Silence before the borrower's latest turn. |
| `cust_turn_dur_mean_ms` | ms | Average length of a borrower turn. |
| `cust_turn_dur_std_ms` | ms | How much borrower turn length varies. |
| `cust_speech_rate_mean` | chars/s | Average speaking speed (characters per second). Stands in for hesitation, since there is no audio. |
| `cust_speech_rate_std` | chars/s | How much speaking speed varies. |
| `cust_speech_rate_last` | chars/s | Speaking speed in the latest borrower turn. |
| `cust_asr_conf_mean` | 0–1 | Average speech-to-text confidence on borrower turns. Mumbled speech scores lower. |
| `cust_asr_conf_min` | 0–1 | Lowest speech-to-text confidence on a borrower turn. |
| `cust_asr_conf_last` | 0–1 | Speech-to-text confidence on the latest borrower turn. |
| `cust_talk_ms` | ms | Total time the borrower spoke. |
| `house_talk_ms` | ms | Total time the bot/agent spoke. |
| `cust_talk_share` | 0–1 | Borrower's share of the talking time. |
| `silence_share` | 0–1 | Share of the call so far that was silence between turns. |
| `pause_on_promise_ms` | ms | Silence before the promise turn. |
| `pause_on_promise_z` | number | That pause compared with this borrower's own pauses (z-score). Above 0 = longer than usual for them; fair across fast and slow talkers. |
| `asr_conf_on_promise` | 0–1 | Speech-to-text confidence on the promise turn. |
| `asr_conf_dip_on_promise` | number | Borrower's average confidence minus confidence on the promise turn. Positive = less clear than usual when promising. |
| `speech_rate_on_promise` | chars/s | Speaking speed on the promise turn. |
| `promise_turn_words` | number | Words in the promise turn. |

### 1.7 Account details (copied from `accounts.csv`, `lenders.csv`, `ptps.csv`)

Known before the call starts, so the same value appears on every in-call row of a promise.

| Column | Type | What it means |
|---|---|---|
| `portfolio` | category | Loan product: `two_wheeler`, `pl_salaried`, `mfi_jlg`, `consumer_durable`, `credit_card`, `msme`. |
| `income_type` | category | `salaried`, `self_employed` or `daily_wage`. |
| `town_id` | category | Borrower's town (T1–T3). |
| `preferred_language` | category | Language registered on the account: `hinglish`, `kanglish`, `english`. What they *registered*, not necessarily what they spoke. |
| `bucket_start` | category | Delinquency bucket at window start: `X` (current), `1-30`, `31-60`, `61-90`, `90+` days past due. |
| `dpd_start` | number | Days past due at window start. |
| `emi_amount` | ₹ | Monthly instalment. |
| `outstanding` | ₹ | Total loan balance outstanding. |
| `bureau_score_band` | category | Credit score band (e.g. `550-649`). ⚠ As-of date undocumented. |
| `other_active_loans` | number | Other loans the borrower has. ⚠ As-of date undocumented. |
| `paid_other_lenders_30d` | true/false | Paid another lender in the last 30 days. ⚠ As-of date undocumented. |
| `last_bounce_reason` | category | Why the last auto-debit failed: `insufficient_funds`, `mandate_not_registered`, `stop_payment`, `technical`. ⚠ As-of date undocumented. |
| `ability_to_pay_estimate` | 0–1 | Lender's estimate of ability to pay, as a fraction of the EMI. ⚠ As-of date undocumented. |
| `lender_id` | category | Lender (L01–L06). |
| `lender_type` | category | `Private bank`, `NBFC`, `Small finance bank`, `Fintech NBFC`, `NBFC-MFI`, `NBFC (MSME)`. |
| `dialling_arm` | category | How the account was chosen for dialling: `rule_based` or `random_contact_point`. |
| `channel` | category | How the promise was taken: `voice_bot`, `tele_agent` or `field`. |
| `source` | category | `call` or `field_visit`. |

### 1.8 Account features (calculated)

Also known before the call starts.

| Column | Type | What it means |
|---|---|---|
| `acct_promise_horizon_days` | days | Days from the call date to the promised date. |
| `acct_promised_amount` | ₹ | Amount promised (from `ptps.csv`). |
| `acct_overdue_at_capture` | ₹ | Amount overdue when the promise was taken. |
| `acct_promise_to_overdue_ratio` | number | Promised ÷ overdue. Below 1 = part payment, which is not a bad sign by itself. |
| `acct_is_part_payment_promise` | 0/1 | 1 if less than 99% of the overdue was promised. |
| `acct_promise_to_ability_ratio` | number | Promised ÷ (ability-to-pay estimate × EMI). Above 1 = promised more than they are estimated to manage. ⚠ Built on an undocumented column. |
| `acct_promise_to_emi_ratio` | number | Promised ÷ monthly EMI. |
| `acct_has_salary_day` | 0/1 | 1 if the account has a salary credit day (about 31% of promises do). |
| `acct_days_promised_before_salary` | days | Days from the promised date forward to the next salary day, wrapping round the month end. 0 = on payday; 1–5 = just before payday. |
| `acct_promise_before_payday` | 0/1 | 1 if the promise falls 1–5 days before payday: structurally hard to keep. The no-model quick win. |
| `acct_prev_ptp_count` | number | Earlier promises on the account (snapshot from `accounts.csv`, taken at window start). |
| `acct_prev_ptp_broken` | number | How many of those earlier promises broke (snapshot). |
| `acct_prev_ptp_broken_rate` | 0–1 | Broken ÷ earlier promises. |
| `acct_is_repeat_promiser` | 0/1 | 1 if 2+ earlier promises and 2+ broken. |
| `acct_n_prior_ptps_in_data` | number | Promises on this account in `ptps.csv` *before* this one. Running count, unlike the snapshot above. |
| `acct_answerer_is_self` | 0/1 | 1 if the person who answered was recorded as the borrower. 0 strongly suggests a third-party promise. |
| `acct_answerer` | category | Who answered: `self`, `family_member` or `third_party`. |
| `acct_payment_link_sent` | 0/1 | 1 if a payment link was sent for this promise (recorded on the promise). |
| `acct_capture_hour` | 0–23 | Hour the promise was logged. |
| `acct_capture_dow` | 0–6 | Day of week the promise was logged (0 = Monday). |
| `acct_capture_is_late_shift` | 0/1 | 1 if logged at 6 pm or later. |

### 1.9 Agent features

Agent details from `agents.csv`, and the agent's track record using **only data from before
this call** and **only train-split accounts**, so nothing from the future or the test set
leaks in. Field agents never appear in `dial_attempts.csv`, so their track-record columns
are blank.

| Column | Type | What it means |
|---|---|---|
| `agent_channel` | category | Agent type: `voice_bot`, `tele` or `field`. |
| `agent_language_team` | category | Agent's language team: `hinglish`, `kanglish`, `english` or `all`. |
| `agent_town_id` | category | Agent's town. Only field agents have one, so mostly blank. |
| `agent_tenure_months` | months | How long the agent has worked. |
| `agent_shift` | category | `day` or `all`. |
| `agent_language_match` | 0/1 | 1 if the agent's language team matches the borrower's registered language (`all` always matches). |
| `agent_is_bot` | 0/1 | 1 if the promise was taken by the voice bot. |
| `agent_hist_kept_rate` | 0–1 | Share of this agent's past promises that were kept. Uses only promises whose outcome was known **before** this call, train-split accounts only. Blank if under 20. |
| `agent_hist_n_resolved` | number | How many past promises the rate above is based on. |
| `agent_hist_ptp_per_connect` | 0–1 | Promises logged per answered call, over calls before this one. Unusually high = possibly pushing promises. Blank if under 20 answered calls. |
| `agent_hist_n_connects` | number | How many past answered calls the rate above is based on. |
| `agent_late_shift_bunching` | 0–1 | Average share of the agent's promises logged after 6 pm, over previous days. Bunching late looks like chasing a daily target. |
| `agent_remark_reuse_rate` | 0–1 | Average share of the agent's call remarks that were copies of their other remarks that day, over previous days. Copy-paste remarks go with promises never really taken. |

---

## 2. `features_postcall.csv`

Things that happened **after** the call, counted only from the call until the promised date
(events after that usually accompany the payment itself). **Never use these in the in-call
or end-call model**: at call time they haven't happened yet.

| Column | Type | What it means |
|---|---|---|
| `ptp_id` | text | Promise ID. |
| `account_id` | text | Loan account. |
| `split` | text | `train`, `validation` or `test`. |
| `post_payment_link_sent_n` | number | Times: payment link sent to the borrower. |
| `post_payment_link_sent_any` | 0/1 | 1 if: payment link sent to the borrower. |
| `post_payment_link_opened_n` | number | Times: borrower opened the payment link. |
| `post_payment_link_opened_any` | 0/1 | 1 if: borrower opened the payment link. |
| `post_whatsapp_confirmation_sent_n` | number | Times: WhatsApp confirmation sent. |
| `post_whatsapp_confirmation_sent_any` | 0/1 | 1 if: WhatsApp confirmation sent. |
| `post_confirmation_reply_yes_n` | number | Times: borrower replied "yes" to the confirmation. |
| `post_confirmation_reply_yes_any` | 0/1 | 1 if: borrower replied "yes" to the confirmation. |
| `post_confirmation_reply_no_n` | number | Times: borrower replied "no" to the confirmation. |
| `post_confirmation_reply_no_any` | 0/1 | 1 if: borrower replied "no" to the confirmation. |
| `post_inbound_call_from_customer_n` | number | Times: borrower called back. |
| `post_inbound_call_from_customer_any` | 0/1 | 1 if: borrower called back. |
| `post_hours_to_link_open` | hours | Hours from the call to the first link open. Blank if not opened before the promised date. |
| `post_engagement_score` | 0–3 | Link opened + replied yes + called back (each 0/1). Higher = more engaged. |

---

## 3. `labels.csv`

The answer key. The pipeline uses it only to calculate the agent's past keep rate (from
outcomes known before each call). It must never be given to a model as an input, and the
pipeline stops with an error if `kept` appears in any feature table.

| Column | Type | What it means |
|---|---|---|
| `ptp_id` | text | Promise ID. |
| `kept` | 0/1 | **The outcome.** 1 if the borrower paid at least 80% of the promised amount, through a borrower-chosen channel (`upi_app`, `upi_link`, `bbps`, `neft`, `branch_cash`, `cash_field`; not `nach_represent`), between the call and the end of promised date + 3 days. Each payment counts toward one promise only. |
| `amount_ratio` | number | Amount paid in the window ÷ amount promised. |
| `paid` | ₹ | Amount paid in the window and credited to this promise. |
| `label_source` | text | Where the label came from: `team_b_fallback` (worked out here) or `phase1:<file>` (Phase 1's official labels, used automatically when present). |
