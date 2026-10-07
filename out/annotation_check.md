# Annotation check

How the engineered features line up with the human taxonomy in `annotated_ptp_sample.csv`. The annotation is a label: it is used here for validation only and never enters a feature table.

- Annotated promises: **200** (by split: {'train': 139, 'validation': 34, 'test': 27})
- Annotated ptp_ids not found in ptps.csv: 0
- Used here (train + validation; **test excluded** so nobody tunes on it): **173**
- Outcome label source: `team_b_fallback`

## Classes

| Type | n | Kept rate |
| --- | --- | --- |
| agent_recorded_or_pushed | 12 | 0.0% |
| escape | 31 | 6.5% |
| genuine_feasible | 47 | 55.3% |
| genuine_infeasible | 43 | 14.0% |
| repeat_promiser | 24 | 0.0% |
| third_party | 16 | 25.0% |

Annotator confidence: {'high': 96, 'medium': 64, 'low': 13}

## Rule-level detectors

Each taxonomy class against the simple feature rule that should fire on it. Low recall means the features miss the class; low precision means the rule also fires on other classes.

| Class | Rule | n class | n fired | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| agent_recorded_or_pushed | borrower never stated a date or an amount | 12 | 42 | 58% | 17% |
| third_party | third-party cue, or answerer not recorded as self | 16 | 14 | 75% | 86% |
| genuine_infeasible | promise just before payday, hardship/refusal cue, or conditional cue | 43 | 78 | 72% | 40% |
| repeat_promiser | 2+ earlier promises and 2+ broken (accounts snapshot) | 24 | 33 | 42% | 30% |
| escape | termination cue, link refusal, or only a vague date | 31 | 120 | 87% | 22% |

## Feature means by class

| Feature | agent_recorded_or_pushed | escape | genuine_feasible | genuine_infeasible | repeat_promiser | third_party |
| --- | --- | --- | --- | --- | --- | --- |
| `has_transcript` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `cust_stated_neither_date_nor_amount` | 0.58 | 0.39 | 0.11 | 0.07 | 0.42 | 0.31 |
| `cust_stated_concrete_date` | 0.00 | 0.13 | 0.51 | 0.70 | 0.25 | 0.12 |
| `cust_stated_vague_date_only` | 0.42 | 0.48 | 0.19 | 0.07 | 0.33 | 0.56 |
| `lex_hedge_any` | 0.58 | 0.45 | 0.21 | 0.47 | 0.54 | 0.00 |
| `lex_conditional_any` | 0.00 | 0.23 | 0.13 | 0.42 | 0.08 | 0.00 |
| `lex_hardship_any` | 0.25 | 0.10 | 0.06 | 0.30 | 0.08 | 0.00 |
| `lex_third_party_any` | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.44 |
| `lex_termination_any` | 0.75 | 0.74 | 0.60 | 0.33 | 0.50 | 0.00 |
| `lex_link_refusal_any` | 0.00 | 0.19 | 0.30 | 0.16 | 0.12 | 0.00 |
| `safety_signal_any` | 0.42 | 0.13 | 0.30 | 0.51 | 0.12 | 0.00 |
| `acct_answerer_is_self` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 |
| `acct_promise_before_payday` | 0.00 | 0.19 | 0.25 | 0.17 | 0.20 | 0.00 |
| `acct_is_repeat_promiser` | 0.17 | 0.00 | 0.17 | 0.23 | 0.42 | 0.19 |
| `acct_prev_ptp_broken_rate` | 0.50 | 0.43 | 0.39 | 0.54 | 0.68 | 0.54 |

> The sample is small (tens per class). Treat these as direction checks, not performance estimates.