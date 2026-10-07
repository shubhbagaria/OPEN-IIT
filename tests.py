#!/usr/bin/env python3
"""
Phase 2 / Team B test suite.

These are not smoke tests. Each one targets a specific way this pipeline could
be silently wrong in a way that would not show up until Phase 3 or Phase 4:

  1. prefix/end-call equivalence   -> train/serve skew in Phase 4
  2. prefix monotonicity           -> future leaking into a mid-call score
  3. language-blind lexicons       -> recall collapse on code-mixed speech
  4. amount vs unit disambiguation -> "7 days" read as Rs 7
  5. agent-pushed detection        -> the integrity product silently failing
  6. post-call quarantine          -> the biggest leak available
  7. agent as-of correctness       -> target leakage via agent history
  8. split integrity               -> account history crossing folds
  9. circular salary arithmetic    -> infeasibility flag inverted at month ends
 10. NACH exclusion                -> a mandate-health model wearing a
                                      sincerity label

Run:  python tests.py                      (real data in csv_folder/)
      python tests.py --data-dir data_synthetic   (after make_synthetic.py)
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ptp_features import lexicons as lx
from ptp_features.config import Config, DEFAULT
from ptp_features.context import _days_before_salary, build_account_features
from ptp_features.extractors import find_amounts, find_dates
from ptp_features.features import extract_prefix_features, prepare_turns
from ptp_features.pipeline import (
    build_conversational_tables,
    load_raw,
    _recompute_labels,
)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    """Decorator turning a function into a reported test."""

    def wrap(fn):
        def runner(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                PASSED.append(name)
                print(f"  PASS  {name}")
            except AssertionError as exc:
                FAILED.append((name, str(exc)))
                print(f"  FAIL  {name}\n        {exc}")
            except Exception:
                FAILED.append((name, traceback.format_exc()))
                print(f"  ERROR {name}\n{traceback.format_exc()}")

        runner.__name__ = fn.__name__
        return runner

    return wrap


def turn(idx, speaker, text, pause=400, start=None, dur=1200, conf=0.9):
    start = idx * 2000 if start is None else start
    return {
        "turn_idx": idx, "speaker": speaker, "text": text,
        "start_ms": start, "end_ms": start + dur,
        "pause_before_ms": pause, "asr_confidence": conf,
    }


# ==========================================================================
# 1-2. The prefix invariant
# ==========================================================================

CALL_A = [
    turn(0, "bot", "Hello, this is an automated call from [LENDER]. Am I speaking with [NAME]?"),
    turn(1, "customer", "Yes, speaking."),
    turn(2, "bot", "An amount of Rs 12,200 has been due on your account for 23 days."),
    turn(3, "customer", "I forgot about it."),
    turn(4, "bot", "By which date can you make the payment?"),
    turn(5, "customer", "i will do it myself on next week, for sure. okay okay, i have to go.", pause=1355),
    turn(6, "bot", "Okay, Rs 12,200 on 8 April, I am recording it."),
    turn(7, "bot", "You will get a payment link on this number."),
    turn(8, "customer", "Don't send a link I will do it myself"),
    turn(9, "bot", "Thanks for your time."),
]


@check("extractor is deterministic and the final prefix differs from an early one")
def test_extractor_deterministic():
    recs = prepare_turns(CALL_A)
    last = extract_prefix_features(recs, len(recs) - 1)
    again = extract_prefix_features(recs, len(recs) - 1)
    assert last == again, "extractor is not deterministic"
    # And it must differ from an earlier prefix, or the test is vacuous.
    early = extract_prefix_features(recs, 3)
    assert early != last, "prefix at k=3 identical to end of call; test is vacuous"


def _same(a, b) -> bool:
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


@check("pipeline end-call row == final prefix recomputed from the transcript")
def test_endcall_matches_final_prefix(frames, outputs):
    """The real train/serve-skew check: what the pipeline wrote for a PTP must
    equal what the serving path (prepare_turns + extract_prefix_features at the
    last turn) computes from that call's transcript rows."""
    endcall = outputs["features_endcall"].set_index("ptp_id")
    transcripts = frames["call_transcripts"]
    sample = endcall[endcall["has_transcript"] == 1].head(40)
    assert len(sample) >= 10, "too few transcribed PTPs; test is vacuous"
    for ptp_id, row in sample.iterrows():
        rows = transcripts[transcripts["call_id"] == row["call_id"]].to_dict("records")
        recs = prepare_turns(rows, min_bare_amount=DEFAULT.features.min_bare_number_as_amount)
        expected = extract_prefix_features(recs, len(recs) - 1)
        diffs = [k for k, v in expected.items() if not _same(v, row[k])]
        assert not diffs, f"{ptp_id}: end-call table differs from final prefix on {diffs[:8]}"


@check("prefix at k never reads turns after k")
def test_prefix_ignores_future():
    recs_full = prepare_turns(CALL_A)
    for k in range(len(CALL_A)):
        truncated = prepare_turns(CALL_A[: k + 1])
        a = extract_prefix_features(recs_full, k)
        b = extract_prefix_features(truncated, k)
        diffs = {
            key for key in a
            if not (a[key] == b[key] or (
                isinstance(a[key], float) and isinstance(b[key], float)
                and math.isnan(a[key]) and math.isnan(b[key])
            ))
        }
        assert not diffs, (
            f"at k={k}, features differ when future turns are removed: {sorted(diffs)}. "
            "This means the extractor is reading ahead."
        )


@check("monotone counters never decrease as the prefix grows")
def test_monotonicity():
    recs = prepare_turns(CALL_A)
    monotone = [
        "n_turns_so_far", "n_customer_turns", "n_house_turns",
        "cust_chars_total", "cust_words_total",
        "n_cust_date_mentions", "n_cust_amount_mentions",
        "lex_hedge_count", "lex_termination_count",
    ]
    prev = None
    for k in range(len(recs)):
        cur = extract_prefix_features(recs, k)
        if prev is not None:
            for key in monotone:
                assert cur[key] >= prev[key], (
                    f"{key} decreased from {prev[key]} to {cur[key]} at k={k}"
                )
        prev = cur


# ==========================================================================
# 3. Language-blind lexicons
# ==========================================================================


@check("lexicons fire on all three languages regardless of registered language")
def test_lexicon_language_blind():
    cases = {
        "hedge": ["I will try", "dekhta hoon", "nodtini"],
        "vague_date": ["next week", "agle hafte", "munde vaara"],
        "hardship": ["business down", "paise nahi hai", "business down ide"],
        "termination": ["i have to go", "mujhe jaana hai", "naanu hogbeku"],
        "link_refusal": ["Don't send a link I will do it myself",
                         "link mat bhejo", "link beda naane maadtini"],
        "conditional": ["if I get paid", "salary aayi toh", "party payment kottre"],
    }
    for category, texts in cases.items():
        for text in texts:
            counts = lx.count_category_hits(lx.normalise(text))
            assert counts[category] > 0, (
                f"category {category!r} did not fire on {text!r}; "
                "lexicon has a hole for this language"
            )


@check("count_category_hits takes no language argument (cannot be gated)")
def test_no_language_gate():
    import inspect
    sig = inspect.signature(lx.count_category_hits)
    assert list(sig.parameters) == ["normalised_text"], (
        f"count_category_hits accepts {list(sig.parameters)}; a language parameter "
        "would allow the gating bug the sample code had"
    )


# ==========================================================================
# 4. Amount / date disambiguation
# ==========================================================================


@check("amounts: money detected, units rejected")
def test_amount_detection():
    money = {
        "4,200 poora kodtini": 4200.0,
        "an amount of rs 12,200 has been due": 12200.0,
        "eega 2,800 rupees kodbahudu": 2800.0,
        "i will pay 4200": 4200.0,
        "5 hazaar de dunga": 5000.0,
        "rs 50 abhi": 50.0,
        # The rupee sign must survive normalisation as a currency marker,
        # otherwise a small amount falls under the bare-number threshold.
        "₹50 abhi": 50.0,
        "aapke account mein ₹8,800 ka bhugtan": 8800.0,
    }
    for text, expected in money.items():
        found = find_amounts(lx.normalise(text))
        assert found, f"no amount found in {text!r}"
        assert any(abs(a.value - expected) < 1e-6 for a in found), (
            f"expected {expected} in {text!r}, got {[a.value for a in found]}"
        )

    not_money = [
        "payment 7 dinadinda baaki ide",   # 7 days
        "due for 23 days",                 # 23 days
        "i have 2 loans",                  # 2 loans
        "call me at 5 baje",               # 5 o'clock
        "3 mahine se",                     # 3 months
    ]
    for text in not_money:
        found = find_amounts(lx.normalise(text))
        assert not found, f"{text!r} wrongly parsed as money: {[a.value for a in found]}"


@check("dates: concrete vs vague separated correctly")
def test_date_detection():
    concrete = ["3 april", "8 April", "kal kar dunga", "naale maadtini",
                "friday tak", "5 tarikh", "03/04", "today itself"]
    for text in concrete:
        found = find_dates(lx.normalise(text))
        assert any(d.concrete for d in found), f"no concrete date in {text!r}"

    vague = ["next week", "month end", "agle hafte", "munde vaara", "kuch din mein"]
    for text in vague:
        found = find_dates(lx.normalise(text))
        assert found, f"no date at all in {text!r}"
        assert not any(d.concrete for d in found), (
            f"{text!r} wrongly marked concrete: {[d.raw for d in found]}"
        )

    # Short weekday forms that are ordinary words in code-mixed speech.
    for text in ["sun lo bhai", "hum sat log hain", "mon is fine", "haan toh bolo"]:
        found = find_dates(lx.normalise(text))
        assert not found, f"{text!r} wrongly parsed as a date: {[d.raw for d in found]}"


# ==========================================================================
# 5. The agent-pushed detector
# ==========================================================================


@check("agent-pushed detector: borrower who states nothing is flagged")
def test_agent_pushed_detection():
    pushed = [
        turn(0, "bot", "Am I speaking with [NAME]?"),
        turn(1, "customer", "Hmm"),
        turn(2, "bot", "Rs 4,200 has been due for 7 days."),
        turn(3, "customer", "Haan"),
        turn(4, "bot", "By which date can you make the payment?"),
        turn(5, "customer", "Accha"),
        turn(6, "bot", "Confirming, Rs 4,200 by 8 April."),
    ]
    f = extract_prefix_features(prepare_turns(pushed), 6)
    assert f["cust_stated_neither_date_nor_amount"] == 1, (
        "borrower stated no date and no amount but detector did not fire"
    )
    assert f["first_amount_speaker"] == "house"
    assert f["promise_seen"] == 0

    genuine = [
        turn(0, "bot", "By which date can you make the payment?"),
        turn(1, "customer", "I will pay 4200 on 8 April"),
    ]
    g = extract_prefix_features(prepare_turns(genuine), 1)
    assert g["cust_stated_neither_date_nor_amount"] == 0
    assert g["first_amount_speaker"] == "customer"
    assert g["cust_stated_concrete_date"] == 1


@check("who-proposed-first distinguishes house from borrower")
def test_first_speaker():
    house_first = [
        turn(0, "bot", "Can you pay Rs 5,000 by 10 April?"),
        turn(1, "customer", "haan 5,000 theek hai"),
    ]
    f = extract_prefix_features(prepare_turns(house_first), 1)
    assert f["first_amount_speaker"] == "house", f["first_amount_speaker"]
    assert f["first_date_speaker"] == "house", f["first_date_speaker"]


@check("counteroffer detected when borrower names a different number")
def test_counteroffer():
    rows = [
        turn(0, "bot", "please pay at least rs 2,800 today"),
        turn(1, "customer", "eega 1,500 rupees kodbahudu, baaki amele"),
    ]
    f = extract_prefix_features(prepare_turns(rows), 1)
    assert f["cust_counteroffered"] == 1
    assert abs(f["cust_to_house_amount_ratio"] - (1500 / 2800)) < 1e-6
    assert f["house_made_minimum_ask"] == 1

    # AT0000026 shape: the bot states the overdue (4,200), later asks for a
    # 2,800 minimum, and the borrower takes the 2,800. That is acceptance of
    # the house figure in force, not a counteroffer against the overdue.
    accepted = [
        turn(0, "bot", "nimma account nalli 4,200 rupees baaki ide"),
        turn(1, "customer", "4,200 poora kodtini"),
        turn(2, "bot", "kanishta 2,800 rupees ivattu kattibidi"),
        turn(3, "customer", "eega 2,800 rupees kodbahudu"),
    ]
    g = extract_prefix_features(prepare_turns(accepted), 3)
    assert g["cust_counteroffered"] == 0, "accepting the minimum ask scored as counteroffer"
    assert g["cust_last_amount_stated"] == 2800.0


# ==========================================================================
# 6-8. Leakage guards
# ==========================================================================


@check("post-call columns absent from in-call and end-call tables")
def test_postcall_quarantine(outputs):
    post_cols = {c for c in outputs["features_postcall"].columns if c.startswith("post_")}
    assert post_cols, "no post_ columns produced; test is vacuous"
    for name in ("features_endcall", "features_incall"):
        intruders = post_cols.intersection(outputs[name].columns)
        assert not intruders, f"{name} contains post-call columns: {sorted(intruders)}"


@check("agent as-of rates use no outcome resolved after the call")
def test_agent_asof(frames, outputs, cfg):
    """
    Direct check: for each PTP with a non-null agent_hist_kept_rate, recompute
    the rate from scratch using only promises resolved strictly before that
    call, and confirm it matches.
    """
    endcall = outputs["features_endcall"]
    labels = outputs["labels"]
    ptps = frames["ptps"].copy()
    ptps["captured_ts"] = pd.to_datetime(ptps["captured_ts"])
    ptps["promised_date"] = pd.to_datetime(ptps["promised_date"])

    splits = frames["splits"].set_index("account_id")["split"]
    train_accounts = set(splits[splits == "train"].index)

    hist = ptps.merge(labels[["ptp_id", "kept"]], on="ptp_id", how="left")
    hist = hist[hist["account_id"].isin(train_accounts)].dropna(subset=["kept"])
    hist["known_ts"] = cfg.label.window_end(hist["promised_date"])

    merged = endcall[["ptp_id", "agent_hist_kept_rate", "agent_hist_n_resolved"]].merge(
        ptps[["ptp_id", "agent_id", "captured_ts"]], on="ptp_id"
    )
    scored = merged.dropna(subset=["agent_hist_kept_rate"])
    assert len(scored) > 0, "no agent_hist_kept_rate values produced; test is vacuous"

    checked = 0
    for row in scored.head(60).itertuples(index=False):
        prior = hist[
            (hist["agent_id"] == row.agent_id) & (hist["known_ts"] < row.captured_ts)
        ]
        expected = prior["kept"].mean() if len(prior) else np.nan
        assert abs(row.agent_hist_kept_rate - expected) < 1e-9, (
            f"{row.ptp_id}: pipeline says {row.agent_hist_kept_rate:.6f}, "
            f"recomputed as-of says {expected:.6f} over {len(prior)} prior promises"
        )
        checked += 1
    assert checked >= 20, f"only {checked} rows verified"


@check("agent history ignores validation/test accounts entirely")
def test_agent_history_pool(frames, outputs, cfg):
    """
    Removing all non-train PTPs from the inputs must not change any agent
    history feature, because they were never supposed to contribute.
    """
    from ptp_features.context import build_agent_features

    splits = frames["splits"]
    train_accounts = set(splits[splits["split"] == "train"]["account_id"])
    labels = outputs["labels"]

    full = build_agent_features(
        frames["ptps"], frames["dial_attempts"], frames["agents"],
        frames["accounts"], splits, labels, cfg,
    )
    train_only_ptps = frames["ptps"][frames["ptps"]["account_id"].isin(train_accounts)]
    # Keep every PTP's OWN attempt row: it is the call itself (it says when the
    # call started), not history, and non-train ones are excluded from history
    # by the pool filter anyway.
    att = frames["dial_attempts"]
    train_only_attempts = att[
        att["account_id"].isin(train_accounts)
        | att["attempt_id"].isin(frames["ptps"]["attempt_id"].dropna())
    ]
    restricted = build_agent_features(
        frames["ptps"], train_only_attempts, frames["agents"],
        frames["accounts"], splits, labels, cfg,
    )

    for col in ("agent_hist_kept_rate", "agent_hist_ptp_per_connect",
                "agent_late_shift_bunching", "agent_remark_reuse_rate"):
        a = full.set_index("ptp_id")[col]
        b = restricted.set_index("ptp_id")[col]
        mismatch = (a.notna() != b.notna()) | (
            a.notna() & b.notna() & ((a - b).abs() > 1e-9)
        )
        assert not mismatch.any(), (
            f"{col} changed for {int(mismatch.sum())} rows when non-train accounts "
            "were removed -- history pool is leaking across splits"
        )


@check("no account appears in more than one split")
def test_split_integrity(frames):
    splits = frames["splits"]
    dupes = splits.groupby("account_id")["split"].nunique()
    assert (dupes <= 1).all(), f"{int((dupes > 1).sum())} accounts in multiple splits"


@check("end-call table has exactly one row per PTP")
def test_grain(frames, outputs):
    endcall = outputs["features_endcall"]
    assert len(endcall) == len(frames["ptps"]), (
        f"end-call has {len(endcall)} rows for {len(frames['ptps'])} PTPs"
    )
    assert endcall["ptp_id"].is_unique, "duplicate ptp_id in end-call table"


@check("field-sourced PTPs kept with has_transcript = 0")
def test_field_ptps(frames, outputs):
    endcall = outputs["features_endcall"]
    # Keyed on visit_id rather than a `source` spelling ("visit" in old
    # synthetic data, "field_visit" in the real ptps.csv).
    ptps = frames["ptps"]
    field_ids = set(ptps.loc[ptps["visit_id"].notna() & ptps["attempt_id"].isna(), "ptp_id"])
    assert field_ids, "no field PTPs in test data; test is vacuous"
    rows = endcall[endcall["ptp_id"].isin(field_ids)]
    assert (rows["has_transcript"] == 0).all(), "field PTPs marked as having transcripts"
    assert rows["lex_hedge_count"].isna().all(), (
        "field PTPs have conversational features; they have no call"
    )


# ==========================================================================
# 9-10. Domain correctness
# ==========================================================================


@check("salary-day arithmetic wraps correctly across month boundaries")
def test_circular_salary():
    # Promise on 29 April, salary on the 2nd -> 3 days before payday (wraps).
    d = _days_before_salary(pd.Timestamp("2026-04-29"), 2)
    assert d == 3.0, f"expected 3 days before payday, got {d}"

    # Promise on the salary day itself.
    assert _days_before_salary(pd.Timestamp("2026-04-10"), 10) == 0.0

    # Promise 2 days before payday, no wrap.
    assert _days_before_salary(pd.Timestamp("2026-04-08"), 10) == 2.0

    # Promise just after payday -> large value, not small.
    d = _days_before_salary(pd.Timestamp("2026-04-11"), 10)
    assert d == 29.0, f"expected 29 (just after payday), got {d}"

    # Missing salary day.
    assert math.isnan(_days_before_salary(pd.Timestamp("2026-04-11"), np.nan))


@check("promise-before-payday flag fires on the infeasible window only")
def test_payday_flag():
    ptps = pd.DataFrame({
        "ptp_id": ["P1", "P2", "P3"],
        "account_id": ["A1", "A2", "A3"],
        "agent_id": ["T1", "T1", "T1"],
        "captured_ts": ["2026-04-01 10:00:00"] * 3,
        "promised_date": ["2026-04-08", "2026-04-11", "2026-04-10"],
        "promised_amount": [1000.0, 1000.0, 1000.0],
        "overdue_at_capture": [2000.0, 2000.0, 2000.0],
        "answerer_recorded": ["self"] * 3,
        "payment_link_sent": [True] * 3,
    })
    accounts = pd.DataFrame({
        "account_id": ["A1", "A2", "A3"],
        "lender_id": ["L01"] * 3,
        "salary_credit_day": [10.0, 10.0, 10.0],
        "emi_amount": [1000.0] * 3,
        "ability_to_pay_estimate": [0.5] * 3,
        "preferred_language": ["english"] * 3,
        "prev_ptp_count": [2] * 3,
        "prev_ptp_broken": [1] * 3,
    })
    lenders = pd.DataFrame({"lender_id": ["L01"], "lender_type": ["Private bank"]})

    out = build_account_features(ptps, accounts, lenders).set_index("ptp_id")
    assert out.loc["P1", "acct_promise_before_payday"] == 1, "2 days before payday not flagged"
    assert out.loc["P2", "acct_promise_before_payday"] == 0, "1 day after payday wrongly flagged"
    assert out.loc["P3", "acct_promise_before_payday"] == 0, "payday itself wrongly flagged"


@check("NACH represents excluded from the kept label")
def test_nach_excluded():
    ptps = pd.DataFrame({
        "ptp_id": ["P1", "P2"],
        "account_id": ["A1", "A2"],
        "captured_ts": ["2026-04-01 10:00:00"] * 2,
        "promised_date": ["2026-04-05"] * 2,
        "promised_amount": [1000.0, 1000.0],
    })
    payments = pd.DataFrame({
        "payment_id": ["PY1", "PY2"],
        "account_id": ["A1", "A2"],
        "payment_ts": ["2026-04-05 12:00:00"] * 2,
        "amount": [1000.0, 1000.0],
        # A1 paid by hand; A2's mandate simply cleared on retry.
        "channel": ["upi_app", "nach_represent"],
    })
    labels = _recompute_labels(
        {"ptps": ptps, "payments": payments}, DEFAULT
    ).set_index("ptp_id")
    assert labels.loc["P1", "kept"] == 1, "borrower-initiated UPI payment not counted as kept"
    assert labels.loc["P2", "kept"] == 0, (
        "nach_represent counted as kept -- this trains a mandate-health model "
        "wearing a sincerity label"
    )


@check("part-payment promise measured against promised, not overdue")
def test_part_payment_ratio():
    # PTP000005 in the real data: promised 1,600 against 2,750 overdue.
    ptps = pd.DataFrame({
        "ptp_id": ["P1"], "account_id": ["A1"], "agent_id": ["T1"],
        "captured_ts": ["2026-04-01 12:16:24"], "promised_date": ["2026-04-05"],
        "promised_amount": [1600.0], "overdue_at_capture": [2750.0],
        "answerer_recorded": ["self"], "payment_link_sent": [True],
    })
    accounts = pd.DataFrame({
        "account_id": ["A1"], "lender_id": ["L01"], "salary_credit_day": [np.nan],
        "emi_amount": [1600.0], "ability_to_pay_estimate": [0.8],
        "preferred_language": ["kanglish"], "prev_ptp_count": [0], "prev_ptp_broken": [0],
    })
    lenders = pd.DataFrame({"lender_id": ["L01"], "lender_type": ["Private bank"]})
    out = build_account_features(ptps, accounts, lenders).set_index("ptp_id")
    assert abs(out.loc["P1", "acct_promise_to_overdue_ratio"] - 1600 / 2750) < 1e-9
    assert out.loc["P1", "acct_is_part_payment_promise"] == 1

    payments = pd.DataFrame({
        "payment_id": ["PY1"], "account_id": ["A1"],
        "payment_ts": ["2026-04-05 18:00:00"], "amount": [1600.0], "channel": ["upi_app"],
    })
    labels = _recompute_labels({"ptps": ptps, "payments": payments}, DEFAULT).set_index("ptp_id")
    assert labels.loc["P1", "kept"] == 1, (
        "paying the full promised 1,600 scored as broken -- label is measuring "
        "against overdue instead of promised"
    )


@check("streaming table emits one row per customer turn, in order")
def test_streaming_shape(frames, outputs, cfg):
    incall = outputs["features_incall"]
    assert len(incall) > 0
    transcripts = frames["call_transcripts"]
    n_cust_turns = int(
        (transcripts["speaker"].astype(str).str.lower() == "customer").sum()
    )
    # Every customer turn on a call that produced a PTP becomes a decision point.
    assert len(incall) <= n_cust_turns, (
        f"{len(incall)} decision points from only {n_cust_turns} customer turns"
    )
    grp = incall.groupby("ptp_id")["decision_point"]
    assert (grp.min() == 1).all(), "decision points do not start at 1"
    for ptp_id, series in incall.groupby("ptp_id")["decision_point"]:
        assert list(series) == sorted(series), f"{ptp_id} decision points out of order"
    finals = incall[incall["is_final_turn"] == 1]
    assert len(finals) <= incall["ptp_id"].nunique()


@check("safety signal fires on hardship in every language")
def test_safety_gate():
    for text in ["business is very down and no money",
                 "business bahut down hai, paise nahi hai",
                 "business tumba down ide, duddu illa"]:
        rows = [turn(0, "bot", "By which date can you make the payment?"),
                turn(1, "customer", text)]
        f = extract_prefix_features(prepare_turns(rows), 1)
        assert f["safety_signal_any"] == 1, f"hardship not flagged in {text!r}"
        assert f["lex_hardship_count"] > 0


# ==========================================================================
# 11. Fixes from the code review
# ==========================================================================


def _one_ptp(promised="2026-04-03", captured="2026-04-01 10:00:00", ptp_id="P1"):
    return {
        "ptp_id": ptp_id, "account_id": "A1", "captured_ts": captured,
        "promised_date": promised, "promised_amount": 1000.0,
    }


@check("label: link and field-cash payments count; the last grace day is inclusive")
def test_label_channels_and_window():
    ptps = pd.DataFrame([_one_ptp(ptp_id=f"P{i}") for i in range(1, 4)])
    ptps["account_id"] = ["A1", "A2", "A3"]
    payments = pd.DataFrame({
        "payment_id": ["PY1", "PY2", "PY3"],
        "account_id": ["A1", "A2", "A3"],
        # promised 3 April + 3 grace days -> window runs through 6 April 23:59.
        "payment_ts": ["2026-04-04 09:00:00", "2026-04-06 21:00:00", "2026-04-07 00:00:00"],
        "amount": [1000.0] * 3,
        "channel": ["upi_link", "cash_field", "upi_app"],
    })
    labels = _recompute_labels({"ptps": ptps, "payments": payments}, DEFAULT).set_index("ptp_id")
    assert labels.loc["P1", "kept"] == 1, "payment via the sent link (upi_link) not counted"
    assert labels.loc["P2", "kept"] == 1, "payment late on the last grace day not counted"
    assert labels.loc["P3", "kept"] == 0, "payment after the grace window counted"


@check("label: one payment is credited to only one overlapping promise")
def test_payment_credited_once():
    ptps = pd.DataFrame([
        _one_ptp(promised="2026-04-05", captured="2026-04-01 10:00:00", ptp_id="OLD"),
        _one_ptp(promised="2026-04-06", captured="2026-04-03 10:00:00", ptp_id="NEW"),
    ])
    payments = pd.DataFrame({
        "payment_id": ["PY1"], "account_id": ["A1"],
        "payment_ts": ["2026-04-05 12:00:00"], "amount": [1000.0], "channel": ["upi_app"],
    })
    labels = _recompute_labels({"ptps": ptps, "payments": payments}, DEFAULT).set_index("ptp_id")
    assert int(labels["kept"].sum()) == 1, "one payment made two promises 'kept'"
    assert labels.loc["NEW", "kept"] == 1, "payment should go to the most recent promise"


@check("account flags stay <NA> when their input is missing")
def test_flags_keep_missing():
    ptps = pd.DataFrame({
        "ptp_id": ["P1"], "account_id": ["A1"], "agent_id": ["T1"],
        "captured_ts": ["2026-04-01 10:00:00"], "promised_date": ["2026-04-08"],
        "promised_amount": [1000.0], "overdue_at_capture": [np.nan],
        "answerer_recorded": ["self"], "payment_link_sent": [True],
    })
    accounts = pd.DataFrame({
        "account_id": ["A1"], "lender_id": ["L01"], "salary_credit_day": [np.nan],
        "emi_amount": [1000.0], "ability_to_pay_estimate": [0.5],
        "preferred_language": ["english"], "prev_ptp_count": [np.nan],
        "prev_ptp_broken": [np.nan],
    })
    lenders = pd.DataFrame({"lender_id": ["L01"], "lender_type": ["NBFC"]})
    out = build_account_features(ptps, accounts, lenders).iloc[0]
    for col in ("acct_promise_before_payday", "acct_is_part_payment_promise",
                "acct_is_repeat_promiser"):
        assert pd.isna(out[col]), f"{col} = {out[col]!r} with missing input; should be <NA>"


@check("running prior-promise count uses only earlier promises on the account")
def test_prior_ptp_count():
    ptps = pd.DataFrame({
        "ptp_id": ["P2", "P1", "P3", "Q1"], "account_id": ["A1", "A1", "A1", "A2"],
        "agent_id": ["T1"] * 4,
        "captured_ts": ["2026-04-05 10:00:00", "2026-04-01 10:00:00",
                        "2026-04-09 10:00:00", "2026-04-02 10:00:00"],
        "promised_date": ["2026-04-10"] * 4, "promised_amount": [1000.0] * 4,
        "overdue_at_capture": [1000.0] * 4, "answerer_recorded": ["self"] * 4,
        "payment_link_sent": [True] * 4,
    })
    accounts = pd.DataFrame({"account_id": ["A1", "A2"], "lender_id": ["L01"] * 2,
                             "preferred_language": ["english"] * 2})
    lenders = pd.DataFrame({"lender_id": ["L01"], "lender_type": ["NBFC"]})
    out = build_account_features(ptps, accounts, lenders).set_index("ptp_id")
    got = out["acct_n_prior_ptps_in_data"].to_dict()
    assert got == {"P1": 0, "P2": 1, "P3": 2, "Q1": 0}, got


@check("feature tables carry agent_id once and no label or annotation column")
def test_no_duplicate_or_label_columns(outputs):
    for name in ("features_endcall", "features_incall"):
        cols = set(outputs[name].columns)
        assert "agent_id" in cols, f"{name} lost agent_id"
        dupes = {c for c in cols if c.endswith(("_x", "_y"))}
        assert not dupes, f"{name} has merge-suffixed columns {sorted(dupes)}"
    banned = {"kept", "paid", "amount_ratio", "label_source",
              "annotated_type", "annotator_confidence"}
    for name in ("features_endcall", "features_incall", "features_postcall"):
        hit = banned.intersection(outputs[name].columns)
        assert not hit, f"{name} contains label columns {sorted(hit)}"


@check("post-call counts ignore events on or after the promised date")
def test_postcall_cutoff(frames, outputs):
    from ptp_features.pipeline import build_postcall_features

    ptps = pd.DataFrame({
        "ptp_id": ["P1"], "captured_ts": ["2026-04-01 10:00:00"],
        "promised_date": ["2026-04-05"],
    })
    events = pd.DataFrame({
        "ptp_id": ["P1"] * 3,
        "event_type": ["payment_link_opened"] * 3,
        "event_ts": ["2026-04-02 09:00:00", "2026-04-05 08:00:00", "2026-04-01 09:00:00"],
    })
    out = build_postcall_features({"ptps": ptps, "post_call_events": events}, DEFAULT)
    n = int(out.loc[0, "post_payment_link_opened_n"])
    assert n == 1, f"expected only the 2 April open to count, got {n}"


@check("agent day-level features use earlier days even with no activity on the call day")
def test_bunching_uses_previous_days():
    from ptp_features.context import _attach_bunching_and_remarks

    attempts = pd.DataFrame({
        "attempt_id": ["A1", "A2", "A3"], "account_id": ["X"] * 3,
        "agent_id": ["T1"] * 3,
        "attempt_ts": ["2026-04-01 19:00:00", "2026-04-01 10:00:00", "2026-04-02 19:00:00"],
        "ptp_id": ["p1", "p2", "p3"], "remark": ["a", "a", "b"],
    })
    df = pd.DataFrame({
        "ptp_id": ["Q1", "Q2", "Q3"], "agent_id": ["T1"] * 3,
        # Q1: same day as the first activity -> nothing earlier.
        # Q2: two days after the last activity day -> must still see history.
        "captured_ts": pd.to_datetime(["2026-04-01 12:00:00", "2026-04-04 12:00:00",
                                       "2026-04-02 12:00:00"]),
    })
    out = _attach_bunching_and_remarks(df, attempts, {"X"}).set_index("ptp_id")
    assert pd.isna(out.loc["Q1", "agent_late_shift_bunching"]), "own day leaked in"
    assert abs(out.loc["Q3", "agent_late_shift_bunching"] - 0.5) < 1e-9
    # Days 1 (0.5) and 2 (1.0) both precede 4 April.
    assert abs(out.loc["Q2", "agent_late_shift_bunching"] - 0.75) < 1e-9, (
        out.loc["Q2", "agent_late_shift_bunching"]
    )


@check("lexicon terms are plain ASCII (no look-alike Unicode letters)")
def test_lexicon_ascii():
    bad = [(c, t) for c, terms in lx.LEXICONS.items() for t in terms if not t.isascii()]
    assert not bad, f"non-ASCII lexicon terms can never match romanised text: {bad}"


@check("annotation check runs on train + validation only")
def test_annotation_check(frames, outputs):
    from ptp_features.audit import annotation_check

    if "annotated_ptp_sample" not in frames:
        raise AssertionError("annotated_ptp_sample.csv absent; test is vacuous")
    ann = annotation_check(frames, outputs)
    assert ann["n_unmatched_ptp_ids"] == 0, "annotated ptp_ids missing from ptps.csv"
    splits = frames["splits"].set_index("account_id")["split"]
    acct = frames["ptps"].set_index("ptp_id")["account_id"]
    sample = frames["annotated_ptp_sample"]["ptp_id"]
    expected = int(sample.map(acct).map(splits).isin(["train", "validation"]).sum())
    assert ann["n_used"] == expected, f"used {ann['n_used']}, expected {expected}"


# ==========================================================================
# Runner
# ==========================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("csv_folder"))
    parser.add_argument("--out-dir", type=Path, default=Path("out_test"))
    args = parser.parse_args(argv)

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir)

    print("\nUnit tests (no data needed)")
    test_extractor_deterministic()
    test_prefix_ignores_future()
    test_monotonicity()
    test_lexicon_language_blind()
    test_no_language_gate()
    test_amount_detection()
    test_date_detection()
    test_agent_pushed_detection()
    test_first_speaker()
    test_counteroffer()
    test_circular_salary()
    test_payday_flag()
    test_nach_excluded()
    test_part_payment_ratio()
    test_safety_gate()
    test_label_channels_and_window()
    test_payment_credited_once()
    test_flags_keep_missing()
    test_prior_ptp_count()
    test_bunching_uses_previous_days()
    test_lexicon_ascii()

    print("\nIntegration tests (against the built feature store)")
    from ptp_features.pipeline import run as run_pipeline

    frames = load_raw(cfg)
    outputs = run_pipeline(cfg, frames)

    test_endcall_matches_final_prefix(frames, outputs)
    test_no_duplicate_or_label_columns(outputs)
    test_postcall_cutoff(frames, outputs)
    test_annotation_check(frames, outputs)

    test_postcall_quarantine(outputs)
    test_agent_asof(frames, outputs, cfg)
    test_agent_history_pool(frames, outputs, cfg)
    test_split_integrity(frames)
    test_grain(frames, outputs)
    test_field_ptps(frames, outputs)
    test_streaming_shape(frames, outputs, cfg)

    print(f"\n{'=' * 64}")
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    print(f"{'=' * 64}")
    if FAILED:
        for name, msg in FAILED:
            print(f"\nFAILED: {name}\n{msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
