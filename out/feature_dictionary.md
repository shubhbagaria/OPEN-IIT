# Feature dictionary

Generated from the code, so it cannot drift from what is shipped.

## in_call

_Derived from turns 0..k. Safe at decision point k and later (in features_endcall, k is the last turn)._

| Table | Feature | Group | Nulls | Caveat | Definition |
| --- | --- | --- | --- | --- | --- |
| features_endcall | `asr_conf_dip_on_promise` | ASR confidence around the promise | 53.8% |  | Drop in ASR confidence on the promise turn versus the borrower's call average. A hesitation proxy, standing in for the audio we do not have. |
| features_endcall | `asr_conf_on_promise` | ASR confidence around the promise | 53.8% |  |  |
| features_endcall | `cust_answered_date_q_with_concrete` | Borrower speech volume and timing | 50.2% |  |  |
| features_endcall | `cust_asr_conf_last` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_asr_conf_mean` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_asr_conf_min` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_chars_per_turn` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_chars_total` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_counteroffered` | Borrower speech volume and timing | 17.0% |  | Borrower stated an amount different from the house figure in force at that moment (the most recent house amount before it). Evidence of real negotiation rather than passive agreement. |
| features_endcall | `cust_first_amount_stated` | Borrower speech volume and timing | 64.3% |  |  |
| features_endcall | `cust_last_amount_stated` | Borrower speech volume and timing | 64.3% |  |  |
| features_endcall | `cust_max_amount_stated` | Borrower speech volume and timing | 64.3% |  |  |
| features_endcall | `cust_min_amount_stated` | Borrower speech volume and timing | 64.3% |  |  |
| features_endcall | `cust_pause_last_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_pause_max_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_pause_mean_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_pause_min_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_pause_std_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_speech_rate_last` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_speech_rate_mean` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_speech_rate_std` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_stated_any_amount` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_stated_any_date` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_stated_concrete_date` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_stated_neither_date_nor_amount` | Borrower speech volume and timing | 17.0% |  | Borrower never uttered a date or an amount anywhere in the prefix. The core agent-recorded/agent-pushed detector: if this is 1 at end of call, whatever was logged as a promise was constructed by the house. |
| features_endcall | `cust_stated_vague_date_only` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_talk_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_talk_share` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_to_house_amount_ratio` | Borrower speech volume and timing | 64.6% |  | First borrower amount divided by the house amount in force just before it. |
| features_endcall | `cust_turn_dur_mean_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_turn_dur_std_ms` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_words_per_turn` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `cust_words_total` | Borrower speech volume and timing | 17.0% |  |  |
| features_endcall | `house_asked_date` | Bot / tele-agent behaviour | 17.0% |  |  |
| features_endcall | `house_made_minimum_ask` | Bot / tele-agent behaviour | 17.0% |  |  |
| features_endcall | `house_talk_ms` | Bot / tele-agent behaviour | 17.0% |  |  |
| features_endcall | `safety_signal_any` | Hardship / refusal safety signals | 17.0% |  | 1 if any hardship or refusal cue fired. A hit here must never be allowed to escalate treatment -- route to restructuring or a human. |
| features_endcall | `lex_concrete_commitment_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_concrete_commitment_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_concrete_commitment_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_concrete_commitment_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_concrete_commitment_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_conditional_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_conditional_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_conditional_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_conditional_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_conditional_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hardship_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hardship_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hardship_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hardship_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hardship_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_minus_commitment` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_hedge_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_link_refusal_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_link_refusal_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_link_refusal_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_link_refusal_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_link_refusal_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_over_agreement_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_over_agreement_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_over_agreement_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_over_agreement_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_over_agreement_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_refusal_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_refusal_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_refusal_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_refusal_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_refusal_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_termination_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_termination_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_termination_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_termination_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_termination_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_third_party_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_third_party_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_third_party_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_third_party_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_third_party_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_vague_date_any` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_vague_date_count` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_vague_date_distinct` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_vague_date_in_last_turn` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `lex_vague_date_rate` | Lexical cue counts from borrower speech | 17.0% |  |  |
| features_endcall | `n_cust_amount_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_cust_concrete_date_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_cust_date_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_cust_vague_date_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_customer_turns` | Mention counts | 17.0% |  |  |
| features_endcall | `n_house_amount_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_house_date_mentions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_house_date_questions` | Mention counts | 17.0% |  |  |
| features_endcall | `n_house_turns` | Mention counts | 17.0% |  |  |
| features_endcall | `n_negotiation_exchanges` | Mention counts | 17.0% |  |  |
| features_endcall | `n_turns_so_far` | Mention counts | 17.0% |  |  |
| features_endcall | `obs_lang_dominant` | Observed spoken-language markers | 17.0% |  |  |
| features_endcall | `obs_lang_english_markers` | Observed spoken-language markers | 17.0% |  |  |
| features_endcall | `obs_lang_english_share` | Observed spoken-language markers | 17.7% |  |  |
| features_endcall | `obs_lang_hindi_markers` | Observed spoken-language markers | 17.0% |  |  |
| features_endcall | `obs_lang_hindi_share` | Observed spoken-language markers | 17.7% |  |  |
| features_endcall | `obs_lang_kannada_markers` | Observed spoken-language markers | 17.0% |  |  |
| features_endcall | `obs_lang_kannada_share` | Observed spoken-language markers | 17.7% |  |  |
| features_endcall | `obs_lang_mix` | Observed spoken-language markers | 17.7% |  | 0 for monolingual speech, higher for more code-mixing. Derived from observed markers, not from the registered preferred_language. |
| features_endcall | `call_elapsed_ms` | Other | 17.0% |  |  |
| features_endcall | `promise_follows_date_question` | Promise-turn characteristics | 53.8% |  |  |
| features_endcall | `promise_follows_minimum_ask` | Promise-turn characteristics | 53.8% |  | The promise turn came directly after the house pushed for a minimum or part payment -- i.e. the number may be the house's, not the borrower's. |
| features_endcall | `promise_position_ratio` | Promise-turn characteristics | 53.8% |  | Where in the call the promise appeared, 0 (start) to 1 (end). Promises at the very end often accompany an attempt to exit the call. |
| features_endcall | `promise_seen` | Promise-turn characteristics | 17.0% |  |  |
| features_endcall | `promise_turn_idx` | Promise-turn characteristics | 53.8% |  |  |
| features_endcall | `promise_turn_words` | Promise-turn characteristics | 53.8% |  |  |
| features_endcall | `pause_on_promise_ms` | Promise-turn pause timing | 53.8% |  |  |
| features_endcall | `pause_on_promise_z` | Promise-turn pause timing | 53.8% |  | Pause before the promise turn, z-scored against the borrower's own pauses in this call, so it is comparable across fast and slow talkers. |
| features_endcall | `latency_to_date_question_ms` | Response latency | 50.2% |  | How long the borrower took to answer the 'by which date' question -- the single most informative pause in the call. |
| features_endcall | `silence_share` | Silence share | 17.0% |  |  |
| features_endcall | `speech_rate_on_promise` | Speech rate | 53.8% |  |  |
| features_endcall | `turns_after_promise` | Turn-position counters | 53.8% |  |  |
| features_endcall | `turns_since_date_question` | Turn-position counters | 50.2% |  |  |
| features_endcall | `first_amount_speaker` | Who proposed what, first | 17.0% |  | Who first mentioned a money amount. House-first suggests the number was pushed rather than offered. |
| features_endcall | `first_concrete_date_speaker` | Who proposed what, first | 17.0% |  |  |
| features_endcall | `first_date_speaker` | Who proposed what, first | 17.0% |  | Who first mentioned any date: customer, house (bot/agent), or none. |
| features_incall | `asr_conf_dip_on_promise` | ASR confidence around the promise | 66.1% |  | Drop in ASR confidence on the promise turn versus the borrower's call average. A hesitation proxy, standing in for the audio we do not have. |
| features_incall | `asr_conf_on_promise` | ASR confidence around the promise | 66.1% |  |  |
| features_incall | `cust_answered_date_q_with_concrete` | Borrower speech volume and timing | 60.8% |  |  |
| features_incall | `cust_asr_conf_last` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_asr_conf_mean` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_asr_conf_min` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_chars_per_turn` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_chars_total` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_counteroffered` | Borrower speech volume and timing | 0.0% |  | Borrower stated an amount different from the house figure in force at that moment (the most recent house amount before it). Evidence of real negotiation rather than passive agreement. |
| features_incall | `cust_first_amount_stated` | Borrower speech volume and timing | 73.5% |  |  |
| features_incall | `cust_last_amount_stated` | Borrower speech volume and timing | 73.5% |  |  |
| features_incall | `cust_max_amount_stated` | Borrower speech volume and timing | 73.5% |  |  |
| features_incall | `cust_min_amount_stated` | Borrower speech volume and timing | 73.5% |  |  |
| features_incall | `cust_pause_last_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_pause_max_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_pause_mean_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_pause_min_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_pause_std_ms` | Borrower speech volume and timing | 19.6% |  |  |
| features_incall | `cust_speech_rate_last` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_speech_rate_mean` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_speech_rate_std` | Borrower speech volume and timing | 19.6% |  |  |
| features_incall | `cust_stated_any_amount` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_stated_any_date` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_stated_concrete_date` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_stated_neither_date_nor_amount` | Borrower speech volume and timing | 0.0% |  | Borrower never uttered a date or an amount anywhere in the prefix. The core agent-recorded/agent-pushed detector: if this is 1 at end of call, whatever was logged as a promise was constructed by the house. |
| features_incall | `cust_stated_vague_date_only` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_talk_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_talk_share` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_to_house_amount_ratio` | Borrower speech volume and timing | 73.6% |  | First borrower amount divided by the house amount in force just before it. |
| features_incall | `cust_turn_dur_mean_ms` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_turn_dur_std_ms` | Borrower speech volume and timing | 19.6% |  |  |
| features_incall | `cust_words_per_turn` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `cust_words_total` | Borrower speech volume and timing | 0.0% |  |  |
| features_incall | `house_asked_date` | Bot / tele-agent behaviour | 0.0% |  |  |
| features_incall | `house_made_minimum_ask` | Bot / tele-agent behaviour | 0.0% |  |  |
| features_incall | `house_talk_ms` | Bot / tele-agent behaviour | 0.0% |  |  |
| features_incall | `safety_signal_any` | Hardship / refusal safety signals | 0.0% |  | 1 if any hardship or refusal cue fired. A hit here must never be allowed to escalate treatment -- route to restructuring or a human. |
| features_incall | `lex_concrete_commitment_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_concrete_commitment_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_concrete_commitment_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_concrete_commitment_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_concrete_commitment_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_conditional_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_conditional_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_conditional_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_conditional_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_conditional_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hardship_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hardship_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hardship_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hardship_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hardship_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_minus_commitment` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_hedge_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_link_refusal_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_link_refusal_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_link_refusal_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_link_refusal_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_link_refusal_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_over_agreement_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_over_agreement_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_over_agreement_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_over_agreement_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_over_agreement_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_refusal_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_refusal_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_refusal_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_refusal_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_refusal_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_termination_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_termination_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_termination_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_termination_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_termination_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_third_party_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_third_party_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_third_party_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_third_party_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_third_party_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_vague_date_any` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_vague_date_count` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_vague_date_distinct` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_vague_date_in_last_turn` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `lex_vague_date_rate` | Lexical cue counts from borrower speech | 0.0% |  |  |
| features_incall | `n_cust_amount_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_cust_concrete_date_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_cust_date_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_cust_vague_date_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_customer_turns` | Mention counts | 0.0% |  |  |
| features_incall | `n_house_amount_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_house_date_mentions` | Mention counts | 0.0% |  |  |
| features_incall | `n_house_date_questions` | Mention counts | 0.0% |  |  |
| features_incall | `n_house_turns` | Mention counts | 0.0% |  |  |
| features_incall | `n_negotiation_exchanges` | Mention counts | 0.0% |  |  |
| features_incall | `n_turns_so_far` | Mention counts | 0.0% |  |  |
| features_incall | `obs_lang_dominant` | Observed spoken-language markers | 0.0% |  |  |
| features_incall | `obs_lang_english_markers` | Observed spoken-language markers | 0.0% |  |  |
| features_incall | `obs_lang_english_share` | Observed spoken-language markers | 8.4% |  |  |
| features_incall | `obs_lang_hindi_markers` | Observed spoken-language markers | 0.0% |  |  |
| features_incall | `obs_lang_hindi_share` | Observed spoken-language markers | 8.4% |  |  |
| features_incall | `obs_lang_kannada_markers` | Observed spoken-language markers | 0.0% |  |  |
| features_incall | `obs_lang_kannada_share` | Observed spoken-language markers | 8.4% |  |  |
| features_incall | `obs_lang_mix` | Observed spoken-language markers | 8.4% |  | 0 for monolingual speech, higher for more code-mixing. Derived from observed markers, not from the registered preferred_language. |
| features_incall | `call_elapsed_ms` | Other | 0.0% |  |  |
| features_incall | `promise_follows_date_question` | Promise-turn characteristics | 66.1% |  |  |
| features_incall | `promise_follows_minimum_ask` | Promise-turn characteristics | 66.1% |  | The promise turn came directly after the house pushed for a minimum or part payment -- i.e. the number may be the house's, not the borrower's. |
| features_incall | `promise_position_ratio` | Promise-turn characteristics | 66.1% |  | Where in the call the promise appeared, 0 (start) to 1 (end). Promises at the very end often accompany an attempt to exit the call. |
| features_incall | `promise_seen` | Promise-turn characteristics | 0.0% |  |  |
| features_incall | `promise_turn_idx` | Promise-turn characteristics | 66.1% |  |  |
| features_incall | `promise_turn_words` | Promise-turn characteristics | 66.1% |  |  |
| features_incall | `pause_on_promise_ms` | Promise-turn pause timing | 66.1% |  |  |
| features_incall | `pause_on_promise_z` | Promise-turn pause timing | 66.2% |  | Pause before the promise turn, z-scored against the borrower's own pauses in this call, so it is comparable across fast and slow talkers. |
| features_incall | `latency_to_date_question_ms` | Response latency | 60.8% |  | How long the borrower took to answer the 'by which date' question -- the single most informative pause in the call. |
| features_incall | `silence_share` | Silence share | 0.0% |  |  |
| features_incall | `speech_rate_on_promise` | Speech rate | 66.1% |  |  |
| features_incall | `turns_after_promise` | Turn-position counters | 66.1% |  |  |
| features_incall | `turns_since_date_question` | Turn-position counters | 60.8% |  |  |
| features_incall | `first_amount_speaker` | Who proposed what, first | 0.0% |  | Who first mentioned a money amount. House-first suggests the number was pushed rather than offered. |
| features_incall | `first_concrete_date_speaker` | Who proposed what, first | 0.0% |  |  |
| features_incall | `first_date_speaker` | Who proposed what, first | 0.0% |  | Who first mentioned any date: customer, house (bot/agent), or none. |

## meta

_Identifier or bookkeeping column, not a model input._

| Table | Feature | Group | Nulls | Caveat | Definition |
| --- | --- | --- | --- | --- | --- |
| features_endcall | `agent_id` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_endcall | `account_id` | Other | 0.0% |  |  |
| features_endcall | `call_id` | Other | 17.0% |  |  |
| features_endcall | `has_transcript` | Other | 0.0% |  |  |
| features_endcall | `k_turn_idx` | Other | 17.0% |  |  |
| features_endcall | `ptp_id` | Other | 0.0% |  |  |
| features_endcall | `split` | Other | 0.0% |  |  |
| features_incall | `agent_id` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_incall | `account_id` | Other | 0.0% |  |  |
| features_incall | `call_id` | Other | 0.0% |  |  |
| features_incall | `decision_point` | Other | 0.0% |  |  |
| features_incall | `is_final_turn` | Other | 0.0% |  |  |
| features_incall | `k_turn_idx` | Other | 0.0% |  |  |
| features_incall | `ptp_id` | Other | 0.0% |  |  |
| features_incall | `split` | Other | 0.0% |  |  |
| features_postcall | `account_id` | Other | 0.0% |  |  |
| features_postcall | `ptp_id` | Other | 0.0% |  |  |
| features_postcall | `split` | Other | 0.0% |  |  |

## post_call

_Arrives after the call. NEVER safe for the in-call model._

| Table | Feature | Group | Nulls | Caveat | Definition |
| --- | --- | --- | --- | --- | --- |
| features_postcall | `post_confirmation_reply_no_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_confirmation_reply_no_n` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_confirmation_reply_yes_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_confirmation_reply_yes_n` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_engagement_score` | Post-call engagement (quarantined) | 0.0% |  | Link opened + confirmation 'yes' + inbound call, each 0/1, counted only for events before the promised date. |
| features_postcall | `post_hours_to_link_open` | Post-call engagement (quarantined) | 73.7% |  | Hours from capture to the first link open, if it happened before the promised date. |
| features_postcall | `post_inbound_call_from_customer_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_inbound_call_from_customer_n` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_payment_link_opened_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_payment_link_opened_n` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_payment_link_sent_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_payment_link_sent_n` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_whatsapp_confirmation_sent_any` | Post-call engagement (quarantined) | 0.0% |  |  |
| features_postcall | `post_whatsapp_confirmation_sent_n` | Post-call engagement (quarantined) | 0.0% |  |  |

## pre_call

_Known before the call starts. Safe everywhere._

| Table | Feature | Group | Nulls | Caveat | Definition |
| --- | --- | --- | --- | --- | --- |
| features_endcall | `acct_answerer` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_answerer_is_self` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_capture_dow` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_capture_hour` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_capture_is_late_shift` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_days_promised_before_salary` | Account context and promise-vs-capacity ratios | 68.6% |  | Days from the promised date forward to the salary credit day, computed circularly within the month. Small positive values (1-5) mean the promise falls just BEFORE payday and is structurally hard to keep. |
| features_endcall | `acct_has_salary_day` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_is_part_payment_promise` | Account context and promise-vs-capacity ratios | 0.0% |  | 1 when the promised amount is below 99% of the overdue; <NA> when either is missing. |
| features_endcall | `acct_is_repeat_promiser` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_count, prev_ptp_broken | 1 when the account had at least 2 earlier promises and at least 2 broken (from the accounts.csv snapshot). |
| features_endcall | `acct_n_prior_ptps_in_data` | Account context and promise-vs-capacity ratios | 0.0% |  | Promises on this account in ptps.csv captured strictly before this one. The running counterpart to the window-start snapshot prev_ptp_count. |
| features_endcall | `acct_overdue_at_capture` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_payment_link_sent` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_prev_ptp_broken` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_broken |  |
| features_endcall | `acct_prev_ptp_broken_rate` | Account context and promise-vs-capacity ratios | 28.2% | unverified as-of: prev_ptp_count, prev_ptp_broken |  |
| features_endcall | `acct_prev_ptp_count` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_count |  |
| features_endcall | `acct_promise_before_payday` | Account context and promise-vs-capacity ratios | 68.6% |  | 1 when the promise lands 1-5 days before salary credit. The zero-model infeasibility flag. |
| features_endcall | `acct_promise_horizon_days` | Account context and promise-vs-capacity ratios | 0.0% |  | Days from the capture date to the promised date. |
| features_endcall | `acct_promise_to_ability_ratio` | Account context and promise-vs-capacity ratios | 31.7% | unverified as-of: ability_to_pay_estimate | Promised amount divided by (ability_to_pay_estimate x EMI). Above 1 means the borrower promised more than their estimated capacity. |
| features_endcall | `acct_promise_to_emi_ratio` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `acct_promise_to_overdue_ratio` | Account context and promise-vs-capacity ratios | 0.0% |  | Promised amount as a share of what is overdue. Below 1 means a part-payment promise, which is not in itself a negative signal. |
| features_endcall | `acct_promised_amount` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_endcall | `agent_channel` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_endcall | `agent_hist_kept_rate` | Agent context (as-of, train-pool only) | 38.9% |  | Agent's historical PTP-to-payment rate, using ONLY promises whose outcome was knowable before this call (end of the grace window < now) and ONLY train-split accounts. |
| features_endcall | `agent_hist_n_connects` | Agent context (as-of, train-pool only) | 0.0% |  | Number of the agent's connected train-pool attempts behind agent_hist_ptp_per_connect. |
| features_endcall | `agent_hist_n_resolved` | Agent context (as-of, train-pool only) | 0.0% |  | Number of the agent's resolved train-pool promises behind agent_hist_kept_rate. |
| features_endcall | `agent_hist_ptp_per_connect` | Agent context (as-of, train-pool only) | 20.4% |  | Promises logged per connected call, over attempts that started strictly before this call started (the call's own attempt is excluded). An unusually high value is the first sign of agent-pushed promises. |
| features_endcall | `agent_is_bot` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_endcall | `agent_language_match` | Agent context (as-of, train-pool only) | 0.0% |  | 1 when the agent's language team matches the borrower's registered language (team 'all' always matches); <NA> when either is missing. |
| features_endcall | `agent_language_team` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_endcall | `agent_late_shift_bunching` | Agent context (as-of, train-pool only) | 18.0% |  | Share of the agent's promises logged after 18:00, averaged over their PREVIOUS days only. Target-chasing behaviour. |
| features_endcall | `agent_remark_reuse_rate` | Agent context (as-of, train-pool only) | 17.5% |  | Share of the agent's remarks that duplicate their other remarks, over previous days only. Copy-paste remarks accompany promises never taken. |
| features_endcall | `agent_shift` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_endcall | `agent_tenure_months` | Agent context (as-of, train-pool only) | 6.8% |  |  |
| features_endcall | `agent_town_id` | Agent context (as-of, train-pool only) | 83.6% |  |  |
| features_endcall | `ability_to_pay_estimate` | Other | 30.1% | unverified as-of: ability_to_pay_estimate |  |
| features_endcall | `bucket_start` | Other | 0.0% |  |  |
| features_endcall | `bureau_score_band` | Other | 0.0% | unverified as-of: bureau_score_band |  |
| features_endcall | `channel` | Other | 0.0% |  |  |
| features_endcall | `dialling_arm` | Other | 0.0% |  |  |
| features_endcall | `dpd_start` | Other | 0.0% |  |  |
| features_endcall | `emi_amount` | Other | 0.0% |  |  |
| features_endcall | `income_type` | Other | 0.0% |  |  |
| features_endcall | `last_bounce_reason` | Other | 0.0% | unverified as-of: last_bounce_reason |  |
| features_endcall | `lender_id` | Other | 0.0% |  |  |
| features_endcall | `lender_type` | Other | 0.0% |  |  |
| features_endcall | `other_active_loans` | Other | 0.0% | unverified as-of: other_active_loans |  |
| features_endcall | `outstanding` | Other | 0.0% |  |  |
| features_endcall | `paid_other_lenders_30d` | Other | 0.0% | unverified as-of: paid_other_lenders_30d |  |
| features_endcall | `portfolio` | Other | 0.0% |  |  |
| features_endcall | `preferred_language` | Other | 0.0% |  |  |
| features_endcall | `source` | Other | 0.0% |  |  |
| features_endcall | `town_id` | Other | 0.0% |  |  |
| features_incall | `acct_answerer` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_answerer_is_self` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_capture_dow` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_capture_hour` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_capture_is_late_shift` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_days_promised_before_salary` | Account context and promise-vs-capacity ratios | 67.1% |  | Days from the promised date forward to the salary credit day, computed circularly within the month. Small positive values (1-5) mean the promise falls just BEFORE payday and is structurally hard to keep. |
| features_incall | `acct_has_salary_day` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_is_part_payment_promise` | Account context and promise-vs-capacity ratios | 0.0% |  | 1 when the promised amount is below 99% of the overdue; <NA> when either is missing. |
| features_incall | `acct_is_repeat_promiser` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_count, prev_ptp_broken | 1 when the account had at least 2 earlier promises and at least 2 broken (from the accounts.csv snapshot). |
| features_incall | `acct_n_prior_ptps_in_data` | Account context and promise-vs-capacity ratios | 0.0% |  | Promises on this account in ptps.csv captured strictly before this one. The running counterpart to the window-start snapshot prev_ptp_count. |
| features_incall | `acct_overdue_at_capture` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_payment_link_sent` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_prev_ptp_broken` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_broken |  |
| features_incall | `acct_prev_ptp_broken_rate` | Account context and promise-vs-capacity ratios | 31.1% | unverified as-of: prev_ptp_count, prev_ptp_broken |  |
| features_incall | `acct_prev_ptp_count` | Account context and promise-vs-capacity ratios | 0.0% | unverified as-of: prev_ptp_count |  |
| features_incall | `acct_promise_before_payday` | Account context and promise-vs-capacity ratios | 67.1% |  | 1 when the promise lands 1-5 days before salary credit. The zero-model infeasibility flag. |
| features_incall | `acct_promise_horizon_days` | Account context and promise-vs-capacity ratios | 0.0% |  | Days from the capture date to the promised date. |
| features_incall | `acct_promise_to_ability_ratio` | Account context and promise-vs-capacity ratios | 31.0% | unverified as-of: ability_to_pay_estimate | Promised amount divided by (ability_to_pay_estimate x EMI). Above 1 means the borrower promised more than their estimated capacity. |
| features_incall | `acct_promise_to_emi_ratio` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `acct_promise_to_overdue_ratio` | Account context and promise-vs-capacity ratios | 0.0% |  | Promised amount as a share of what is overdue. Below 1 means a part-payment promise, which is not in itself a negative signal. |
| features_incall | `acct_promised_amount` | Account context and promise-vs-capacity ratios | 0.0% |  |  |
| features_incall | `agent_channel` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_incall | `agent_hist_kept_rate` | Agent context (as-of, train-pool only) | 36.7% |  | Agent's historical PTP-to-payment rate, using ONLY promises whose outcome was knowable before this call (end of the grace window < now) and ONLY train-split accounts. |
| features_incall | `agent_hist_n_connects` | Agent context (as-of, train-pool only) | 0.0% |  | Number of the agent's connected train-pool attempts behind agent_hist_ptp_per_connect. |
| features_incall | `agent_hist_n_resolved` | Agent context (as-of, train-pool only) | 0.0% |  | Number of the agent's resolved train-pool promises behind agent_hist_kept_rate. |
| features_incall | `agent_hist_ptp_per_connect` | Agent context (as-of, train-pool only) | 5.0% |  | Promises logged per connected call, over attempts that started strictly before this call started (the call's own attempt is excluded). An unusually high value is the first sign of agent-pushed promises. |
| features_incall | `agent_is_bot` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_incall | `agent_language_match` | Agent context (as-of, train-pool only) | 0.0% |  | 1 when the agent's language team matches the borrower's registered language (team 'all' always matches); <NA> when either is missing. |
| features_incall | `agent_language_team` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_incall | `agent_late_shift_bunching` | Agent context (as-of, train-pool only) | 1.8% |  | Share of the agent's promises logged after 18:00, averaged over their PREVIOUS days only. Target-chasing behaviour. |
| features_incall | `agent_remark_reuse_rate` | Agent context (as-of, train-pool only) | 1.2% |  | Share of the agent's remarks that duplicate their other remarks, over previous days only. Copy-paste remarks accompany promises never taken. |
| features_incall | `agent_shift` | Agent context (as-of, train-pool only) | 0.0% |  |  |
| features_incall | `agent_tenure_months` | Agent context (as-of, train-pool only) | 8.7% |  |  |
| features_incall | `agent_town_id` | Agent context (as-of, train-pool only) | 100.0% |  |  |
| features_incall | `ability_to_pay_estimate` | Other | 29.1% | unverified as-of: ability_to_pay_estimate |  |
| features_incall | `bucket_start` | Other | 0.0% |  |  |
| features_incall | `bureau_score_band` | Other | 0.0% | unverified as-of: bureau_score_band |  |
| features_incall | `channel` | Other | 0.0% |  |  |
| features_incall | `dialling_arm` | Other | 0.0% |  |  |
| features_incall | `dpd_start` | Other | 0.0% |  |  |
| features_incall | `emi_amount` | Other | 0.0% |  |  |
| features_incall | `income_type` | Other | 0.0% |  |  |
| features_incall | `last_bounce_reason` | Other | 0.0% | unverified as-of: last_bounce_reason |  |
| features_incall | `lender_id` | Other | 0.0% |  |  |
| features_incall | `lender_type` | Other | 0.0% |  |  |
| features_incall | `other_active_loans` | Other | 0.0% | unverified as-of: other_active_loans |  |
| features_incall | `outstanding` | Other | 0.0% |  |  |
| features_incall | `paid_other_lenders_30d` | Other | 0.0% | unverified as-of: paid_other_lenders_30d |  |
| features_incall | `portfolio` | Other | 0.0% |  |  |
| features_incall | `preferred_language` | Other | 0.0% |  |  |
| features_incall | `source` | Other | 0.0% |  |  |
| features_incall | `town_id` | Other | 0.0% |  |  |
