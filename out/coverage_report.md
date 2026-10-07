# Coverage report

## Reach

- Promises in ptps.csv: **3,696**
- With a usable transcript: **3,067** (83.0%)
- Sourced from calls: 3,091
- Sourced from field visits (no transcript): 605
- Distinct calls in transcripts: 3,367
- In-call decision points generated: **15,616** (mean 5.09 per promise)

- Turns per call: min 3, median 12.0, max 26

## Lexicon coverage (the Phase 2 objective)

Borrower utterances with no cue of any category: **54.0%** of 16,223

| Category | % of borrower utterances |
| --- | --- |
| concrete_commitment | 15.09% |
| conditional | 3.54% |
| hardship | 3.27% |
| hedge | 9.20% |
| link_refusal | 4.17% |
| over_agreement | 6.89% |
| refusal | 4.12% |
| termination | 14.87% |
| third_party | 2.30% |
| vague_date | 10.84% |

### Uncovered rate by registered language

| Language | Uncovered | Utterances |
| --- | --- | --- |
| english | 54.0% | 3,719 |
| hinglish | 56.0% | 8,269 |
| kanglish | 50.1% | 4,235 |

> A high uncovered rate for one language means the lexicon has a hole there. Re-run `lexicons.mine_candidate_terms` on that subset.

## Gaps

- Call-sourced promises whose call has no transcript rows: **24** (kept with has_transcript = 0)

## Splits

- train: 1,680 accounts
- validation: 360 accounts
- test: 360 accounts
- Accounts appearing in more than one split: **0** (must be 0)

## Highest null-rate features

| Feature | Null rate |
| --- | --- |
| agent_town_id | 83.6% |
| acct_promise_before_payday | 68.6% |
| acct_days_promised_before_salary | 68.6% |
| cust_to_house_amount_ratio | 64.6% |
| cust_max_amount_stated | 64.3% |
| cust_min_amount_stated | 64.3% |
| cust_last_amount_stated | 64.3% |
| cust_first_amount_stated | 64.3% |
| promise_follows_minimum_ask | 53.8% |
| turns_after_promise | 53.8% |
| pause_on_promise_ms | 53.8% |
| promise_turn_words | 53.8% |
| pause_on_promise_z | 53.8% |
| promise_turn_idx | 53.8% |
| promise_follows_date_question | 53.8% |
| promise_position_ratio | 53.8% |
| speech_rate_on_promise | 53.8% |
| asr_conf_on_promise | 53.8% |
| asr_conf_dip_on_promise | 53.8% |
| cust_answered_date_q_with_concrete | 50.2% |
| turns_since_date_question | 50.2% |
| latency_to_date_question_ms | 50.2% |
| agent_hist_kept_rate | 38.9% |
| acct_promise_to_ability_ratio | 31.7% |
| ability_to_pay_estimate | 30.1% |