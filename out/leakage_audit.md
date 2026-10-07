# Leakage audit

## 1. Structural guarantee

Post-call signals live in `features_postcall` and nowhere else. `pipeline.run()` asserts this and raises if any `post_` column appears in the in-call or end-call tables, so the separation cannot be defeated by accident.

## 2. Are accounts.csv history columns as-of-capture?

- `corr_with_running_prior`: 0.0152
- `corr_with_total_in_window`: 0.0236
- `pct_equal_to_total_in_window`: 0.1648
- `constant_within_account`: True
- `columns_with_undocumented_as_of_semantics`: ['prev_ptp_count', 'prev_ptp_broken', 'ability_to_pay_estimate', 'paid_other_lenders_30d', 'other_active_loans', 'last_bounce_reason', 'bureau_score_band']

**Verdict:** SAFE (snapshot) -- constant per account, consistent with a window-start snapshot. Prefer a recomputed running prior for promises after the first.

## 3. Leak canary

Conversational features correlating with the label above |r| = 0.6 (train split only; label source: `team_b_fallback`):

None. (Smoke alarm did not fire.)

## 4. Agent features

- History pool: **train** accounts only
- Outcome-known-time guard: **True**
- Minimum prior observations before a rate is emitted: **20**

- Outcome known at: end of day promised_date + 3 (the same window the label uses)

Agent rates use only promises already resolved before the current call. A promise due next week contributes nothing today, even though it sits earlier in the file. Promise-per-connect counts only attempts that started before this call started, so the call's own PTP never counts.

## 5. Post-call table cut-off

`features_postcall` counts only events between capture and promised_date + 0 day(s). Events after the promise falls due mostly accompany the payment itself, so counting them would leak the outcome into the parking-window model.

## 6. Labels and annotations never enter a feature table

`pipeline.run()` raises if `kept`, `paid`, `amount_ratio`, `label_source`, `annotated_type` or `annotator_confidence` appears in any feature table.

## 7. Known caveats for Team C

Agent-identity features make the model partly predict *the agent* rather than *the promise*. The brief requires the scorer stay independent of the agents it measures. They are grouped under the `agent_` prefix so they can be ablated in one line -- please report performance with and without them.

Columns built on `accounts.csv` fields with undocumented as-of semantics carry an `as_of_caveat` in `feature_dictionary.csv`. `acct_n_prior_ptps_in_data` is the safe running alternative to `acct_prev_ptp_count`.