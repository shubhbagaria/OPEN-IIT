#!/usr/bin/env python3
"""
Sanity check against REAL rows from call_transcripts.csv.

These two calls were read directly out of the Drive file earlier. AT0000026 is
a Kannada/Kanglish call, AT0000051 an English one. Both are known cases: the
first is a conditional, part-payment-negotiated promise; the second is a
textbook escape promise ending in a link refusal.

This is the check that matters most, because synthetic text can only confirm
the pipeline agrees with its own templates.
"""

from ptp_features.features import extract_prefix_features, prepare_turns

# ---- REAL rows, verbatim from call_transcripts.csv -----------------------

AT0000026 = [
    (0, "bot", 1329, 5781, 466, "Namaskara, idu [LENDER] inda ondu automated call. Neevu [NAME] avra?", 1.0),
    (1, "customer", 6442, 7895, 661, "Haudu enu vishaya", 0.748),
    (2, "bot", 8341, 12017, 446, "Nimma account nalli 4,200 rupees payment 7 dinadinda baaki ide.", 1.0),
    (3, "customer", 12371, 13858, 354, "business tumba down ide.", 0.78),
    (4, "bot", 14326, 15874, 468, "Yaava dinaankadolage payment maadtira?", 1.0),
    (5, "customer", 18295, 21928, 2421, "Nodtini. Party payment kottre naadiddu olage maadtini. 4,200 [inaudible] poora kodtini.", 0.907),
    (6, "bot", 22458, 25325, 530, "Poora aagilla andre kanishta 2,800 rupees ivattu kattibidi.", 1.0),
    (7, "customer", 25620, 28406, 295, "Eega 2,800 rupees kodbahudu, baaki amele.", 0.804),
    (8, "bot", 28864, 31784, 458, "Confirm maadtiddini, 3 April olage 4,200 rupees.", 1.0),
    (9, "bot", 32224, 34531, 440, "Nimma number ge payment link barutte.", 1.0),
    (10, "customer", 34815, 36164, 284, "Haan, link kalsi.", 0.758),
    (11, "bot", 36477, 37841, 313, "Thank you sir.", 1.0),
    (12, "customer", 38242, 38867, 401, "Aytu.", 0.85),
]

AT0000051 = [
    (0, "bot", 1366, 6495, 638, "Hello, this is an automated call from [LENDER]. Am I speaking with [NAME]?", 1.0),
    (1, "customer", 6840, 7979, 345, "Yes, speaking.", 0.84),
    (2, "bot", 8534, 13951, 555, "An amount of Rs 12,200 has been due on your account for 23 days.", 1.0),
    (3, "customer", 14647, 16574, 696, "I forgot about it.", 0.9),
    (4, "bot", 16748, 19351, 174, "By which date can you make the payment?", 1.0),
    (5, "customer", 20706, 27661, 1355, "i will do it myself on next week, for sure. okay okay, i will pay, i have to go.", 0.855),
    (6, "bot", 27911, 31098, 250, "Okay, Rs 12,200 on 8 April, I am recording it.", 1.0),
    (7, "bot", 31515, 34493, 417, "You will get a payment link on this number.", 1.0),
    (8, "customer", 35080, 38369, 587, "Don't send a link I will do it myself", 0.886),
    (9, "bot", 38871, 40849, 502, "Thanks for your time.", 1.0),
]

COLS = ("turn_idx", "speaker", "start_ms", "end_ms", "pause_before_ms", "text", "asr_confidence")


def to_rows(tuples):
    return [dict(zip(COLS, t)) for t in tuples]


def show(call_id, tuples, interesting):
    recs = prepare_turns(to_rows(tuples))
    f = extract_prefix_features(recs, len(recs) - 1)
    print(f"\n{'=' * 70}\n{call_id}  ({len(recs)} turns)\n{'=' * 70}")
    for key in interesting:
        val = f[key]
        if isinstance(val, float):
            val = "nan" if val != val else f"{val:,.3f}"
        print(f"  {key:<42} {val}")
    return f


KEYS = [
    "cust_stated_any_amount", "cust_stated_concrete_date", "cust_stated_vague_date_only",
    "cust_stated_neither_date_nor_amount",
    "first_amount_speaker", "first_date_speaker", "first_concrete_date_speaker",
    "cust_max_amount_stated", "cust_counteroffered", "cust_to_house_amount_ratio",
    "n_negotiation_exchanges",
    "lex_hedge_count", "lex_conditional_count", "lex_hardship_count",
    "lex_vague_date_count", "lex_termination_count", "lex_link_refusal_count",
    "lex_over_agreement_count", "lex_concrete_commitment_count",
    "lex_hedge_minus_commitment", "safety_signal_any",
    "obs_lang_dominant", "obs_lang_mix",
    "house_asked_date", "house_made_minimum_ask", "latency_to_date_question_ms",
    "promise_seen", "promise_turn_idx", "promise_position_ratio",
    "promise_follows_minimum_ask",
    "pause_on_promise_ms", "pause_on_promise_z", "asr_conf_dip_on_promise",
    "cust_talk_share",
]

a = show("AT0000026 -- Kanglish, conditional + part-payment negotiation", AT0000026, KEYS)
b = show("AT0000051 -- English, escape promise + link refusal", AT0000051, KEYS)

print(f"\n{'=' * 70}\nExpectation checks\n{'=' * 70}")
checks = [
    ("AT26: borrower stated an amount (4,200 / 2,800)", a["cust_stated_any_amount"] == 1),
    ("AT26: conditional cue fired ('party payment kottre')", a["lex_conditional_count"] > 0),
    ("AT26: hardship cue fired ('business tumba down ide')", a["lex_hardship_count"] > 0),
    ("AT26: safety gate raised", a["safety_signal_any"] == 1),
    ("AT26: hedge fired ('Nodtini')", a["lex_hedge_count"] > 0),
    ("AT26: house made the minimum ask", a["house_made_minimum_ask"] == 1),
    ("AT26: long pause before promise (2421ms)", a["pause_on_promise_ms"] == 2421),
    ("AT26: dominant observed language is Kannada", a["obs_lang_dominant"] == "kannada"),
    ("AT26: real negotiation happened", a["n_negotiation_exchanges"] >= 1),
    ("AT51: borrower stated NO amount", b["cust_stated_any_amount"] == 0),
    ("AT51: borrower gave only a vague date ('next week')", b["cust_stated_vague_date_only"] == 1),
    ("AT51: house proposed the amount first", b["first_amount_speaker"] == "house"),
    ("AT51: termination cue fired ('i have to go')", b["lex_termination_count"] > 0),
    ("AT51: over-agreement fired ('okay okay')", b["lex_over_agreement_count"] > 0),
    ("AT51: link refusal fired", b["lex_link_refusal_count"] > 0),
    ("AT51: no hardship -> safety gate NOT raised", b["safety_signal_any"] == 0),
    ("AT51: slow answer to the date question (1355ms)", b["latency_to_date_question_ms"] == 1355),
    ("AT51: dominant observed language is English", b["obs_lang_dominant"] == "english"),
    ("AT51 vs AT26: weaker commitment language", b["lex_hedge_minus_commitment"] >= a["lex_hedge_minus_commitment"]),
]
bad = 0
for label, ok in checks:
    print(f"  {'PASS ' if ok else 'FAIL '} {label}")
    bad += (not ok)
print(f"\n  {len(checks) - bad} passed, {bad} failed")
raise SystemExit(1 if bad else 0)
