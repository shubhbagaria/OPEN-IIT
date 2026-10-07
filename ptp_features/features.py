"""
The prefix feature extractor -- the heart of Phase 2.

ONE FUNCTION, TWO TABLES
------------------------
`extract_prefix_features(records, k)` computes every conversational feature from
turns 0..k only. The streaming table calls it at every customer turn; the
end-of-call table calls it once with k = last turn. There is deliberately no
second implementation for the "whole call" case.

This matters because Phase 4 has to serve these features live. If the training
features and the serving features came from different code paths they would
drift apart, and the model would silently degrade in production. One function
means that cannot happen, and `tests.py` asserts the equivalence explicitly.

A "turn record" is a small frozen struct holding everything derivable from a
single turn in isolation. Building them once per call and then slicing is what
keeps the streaming expansion linear rather than quadratic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import lexicons as lx
from .extractors import (
    AmountMention,
    DateMention,
    find_amounts,
    find_dates,
    is_date_question,
    is_minimum_ask,
)

CUSTOMER = "customer"


# --------------------------------------------------------------------------
# Per-turn preprocessing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRecord:
    turn_idx: int
    speaker: str
    is_customer: bool
    start_ms: float
    end_ms: float
    pause_before_ms: float
    asr_confidence: float
    raw_text: str
    norm_text: str
    n_chars: int
    n_words: int
    duration_ms: float
    lex_counts: Mapping[str, int]
    lex_terms: Mapping[str, frozenset[str]]
    lang_counts: Mapping[str, int]
    dates: tuple[DateMention, ...]
    amounts: tuple[AmountMention, ...]
    asks_date: bool
    makes_minimum_ask: bool


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def prepare_turns(
    rows: Sequence[Mapping[str, Any]], *, min_bare_amount: int = 100
) -> list[TurnRecord]:
    """
    Turn the raw transcript rows for ONE call into ordered TurnRecords.

    `rows` must all share a call_id. They are sorted by turn_idx here rather
    than trusting file order, because a single out-of-order row would corrupt
    every "who spoke first" feature downstream.
    """
    ordered = sorted(rows, key=lambda r: int(r["turn_idx"]))
    records: list[TurnRecord] = []

    for row in ordered:
        speaker = str(row.get("speaker", "") or "").strip().lower()
        is_customer = speaker == CUSTOMER

        raw_text = row.get("text") or ""
        if not isinstance(raw_text, str):
            raw_text = ""
        norm = lx.normalise(raw_text)

        start_ms = _safe_float(row.get("start_ms"), 0.0)
        end_ms = _safe_float(row.get("end_ms"), start_ms)
        duration = max(end_ms - start_ms, 0.0)

        # Lexicons and language markers are only meaningful on borrower speech;
        # the bot's lines are a fixed script and would add a constant.
        if is_customer and norm:
            lex_counts = lx.count_category_hits(norm)
            lex_terms = {
                c: frozenset(lx.distinct_category_terms(norm, c))
                for c in lx.ALL_CATEGORIES
            }
            lang_counts = lx.observed_language_counts(norm)
        else:
            lex_counts = {c: 0 for c in lx.ALL_CATEGORIES}
            lex_terms = {c: frozenset() for c in lx.ALL_CATEGORIES}
            lang_counts = {lang: 0 for lang in ("hindi", "kannada", "english")}

        records.append(
            TurnRecord(
                turn_idx=int(row["turn_idx"]),
                speaker=speaker,
                is_customer=is_customer,
                start_ms=start_ms,
                end_ms=end_ms,
                pause_before_ms=_safe_float(row.get("pause_before_ms"), math.nan),
                asr_confidence=_safe_float(row.get("asr_confidence"), math.nan),
                raw_text=raw_text,
                norm_text=norm,
                n_chars=len(norm),
                n_words=len(norm.split()),
                duration_ms=duration,
                lex_counts=lex_counts,
                lex_terms=lex_terms,
                lang_counts=lang_counts,
                # Dates and amounts ARE extracted from both sides: knowing the
                # bot proposed the date first is exactly the point.
                dates=tuple(find_dates(norm)),
                amounts=tuple(find_amounts(norm, min_bare=min_bare_amount)),
                asks_date=(not is_customer) and is_date_question(norm),
                makes_minimum_ask=(not is_customer) and is_minimum_ask(norm),
            )
        )

    return records


# --------------------------------------------------------------------------
# Small statistical helpers (NaN-safe, no numpy dependency in the hot path)
# --------------------------------------------------------------------------


def _clean(values: Sequence[float]) -> list[float]:
    return [v for v in values if v is not None and not math.isnan(v)]


def _mean(values: Sequence[float]) -> float:
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else math.nan


def _std(values: Sequence[float]) -> float:
    vals = _clean(values)
    if len(vals) < 2:
        return math.nan
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _max(values: Sequence[float]) -> float:
    vals = _clean(values)
    return max(vals) if vals else math.nan


def _min(values: Sequence[float]) -> float:
    vals = _clean(values)
    return min(vals) if vals else math.nan


def _ratio(num: float, den: float) -> float:
    if den is None or math.isnan(den) or den == 0:
        return math.nan
    return num / den


# --------------------------------------------------------------------------
# The prefix feature function
# --------------------------------------------------------------------------


def extract_prefix_features(records: Sequence[TurnRecord], k: int) -> dict[str, Any]:
    """
    Every conversational feature, computed from records[0..k] inclusive.

    Nothing after index k is read. That is the single invariant this whole
    module exists to guarantee, and it is what makes the output safe to serve
    mid-call.
    """
    if k < 0 or k >= len(records):
        raise IndexError(f"k={k} out of range for {len(records)} turns")

    prefix = records[: k + 1]
    cust = [r for r in prefix if r.is_customer]
    house = [r for r in prefix if not r.is_customer]  # bot or tele-agent

    f: dict[str, Any] = {}

    # ---------------------------------------------------------------- volume
    f["k_turn_idx"] = prefix[-1].turn_idx
    f["n_turns_so_far"] = len(prefix)
    f["n_customer_turns"] = len(cust)
    f["n_house_turns"] = len(house)
    f["call_elapsed_ms"] = prefix[-1].end_ms
    f["cust_chars_total"] = sum(r.n_chars for r in cust)
    f["cust_words_total"] = sum(r.n_words for r in cust)
    f["cust_chars_per_turn"] = _ratio(f["cust_chars_total"], len(cust))
    f["cust_words_per_turn"] = _ratio(f["cust_words_total"], len(cust))

    # --------------------------------------------------------------- lexicons
    last_cust = cust[-1] if cust else None
    for cat in lx.ALL_CATEGORIES:
        count = sum(r.lex_counts[cat] for r in cust)
        distinct = set().union(*(r.lex_terms[cat] for r in cust)) if cust else set()
        f[f"lex_{cat}_count"] = count
        f[f"lex_{cat}_any"] = int(count > 0)
        f[f"lex_{cat}_rate"] = _ratio(count, len(cust))
        f[f"lex_{cat}_distinct"] = len(distinct)
        f[f"lex_{cat}_in_last_turn"] = (
            int(last_cust.lex_counts[cat] > 0) if last_cust else 0
        )

    # Net direction: hedging against firm commitment language.
    f["lex_hedge_minus_commitment"] = (
        f["lex_hedge_count"] - f["lex_concrete_commitment_count"]
    )
    # Anything that must never trigger escalation.
    f["safety_signal_any"] = int(
        any(f[f"lex_{c}_any"] for c in lx.SAFETY_CATEGORIES)
    )

    # ------------------------------------------------------- observed language
    lang_totals = {
        lang: sum(r.lang_counts.get(lang, 0) for r in cust)
        for lang in ("hindi", "kannada", "english")
    }
    total_lang = sum(lang_totals.values())
    for lang, n in lang_totals.items():
        f[f"obs_lang_{lang}_markers"] = n
        f[f"obs_lang_{lang}_share"] = _ratio(n, total_lang)
    f["obs_lang_dominant"] = (
        max(lang_totals, key=lambda key: lang_totals[key]) if total_lang else "unknown"
    )
    # 0 = monolingual, higher = more code-mixing.
    f["obs_lang_mix"] = (
        1.0 - _ratio(max(lang_totals.values()), total_lang) if total_lang else math.nan
    )

    # ------------------------------------------------------------- structural
    cust_dates = [d for r in cust for d in r.dates]
    cust_amounts = [a for r in cust for a in r.amounts]
    house_dates = [d for r in house for d in r.dates]
    house_amounts = [a for r in house for a in r.amounts]

    f["n_cust_date_mentions"] = len(cust_dates)
    f["n_cust_concrete_date_mentions"] = sum(1 for d in cust_dates if d.concrete)
    f["n_cust_vague_date_mentions"] = sum(1 for d in cust_dates if not d.concrete)
    f["n_cust_amount_mentions"] = len(cust_amounts)
    f["n_house_date_mentions"] = len(house_dates)
    f["n_house_amount_mentions"] = len(house_amounts)

    f["cust_stated_any_date"] = int(bool(cust_dates))
    f["cust_stated_concrete_date"] = int(f["n_cust_concrete_date_mentions"] > 0)
    f["cust_stated_vague_date_only"] = int(
        f["n_cust_vague_date_mentions"] > 0 and f["n_cust_concrete_date_mentions"] == 0
    )
    f["cust_stated_any_amount"] = int(bool(cust_amounts))

    # THE agent_recorded_or_pushed detector. If the borrower never produced a
    # date or an amount, any promise on this call was constructed by the house.
    f["cust_stated_neither_date_nor_amount"] = int(
        not cust_dates and not cust_amounts
    )

    # Who got there first, by turn order. Resolved by scanning the prefix once.
    def first_speaker_of(attr: str, concrete_only: bool = False) -> str:
        for r in prefix:
            items = getattr(r, attr)
            if concrete_only:
                items = [i for i in items if getattr(i, "concrete", False)]
            if items:
                return CUSTOMER if r.is_customer else "house"
        return "none"

    f["first_date_speaker"] = first_speaker_of("dates")
    f["first_concrete_date_speaker"] = first_speaker_of("dates", concrete_only=True)
    f["first_amount_speaker"] = first_speaker_of("amounts")

    # The bot asking for a date, and how the borrower responded to it.
    date_q_turns = [r.turn_idx for r in house if r.asks_date]
    f["house_asked_date"] = int(bool(date_q_turns))
    f["n_house_date_questions"] = len(date_q_turns)
    f["house_made_minimum_ask"] = int(any(r.makes_minimum_ask for r in house))

    first_q_pos = next((i for i, r in enumerate(prefix) if r.asks_date), None)
    if first_q_pos is None:
        f["turns_since_date_question"] = math.nan
        f["cust_answered_date_q_with_concrete"] = math.nan
        f["latency_to_date_question_ms"] = math.nan
    else:
        f["turns_since_date_question"] = len(prefix) - 1 - first_q_pos
        answer = next((r for r in prefix[first_q_pos + 1 :] if r.is_customer), None)
        if answer is None:
            f["cust_answered_date_q_with_concrete"] = math.nan
            f["latency_to_date_question_ms"] = math.nan
        else:
            f["cust_answered_date_q_with_concrete"] = int(
                any(d.concrete for d in answer.dates)
            )
            # How long the borrower took to answer the single most important
            # question in the call.
            f["latency_to_date_question_ms"] = answer.pause_before_ms

    # The promise turn: first customer turn carrying a concrete date or an amount.
    promise_pos = next(
        (
            i
            for i, r in enumerate(prefix)
            if r.is_customer and (any(d.concrete for d in r.dates) or r.amounts)
        ),
        None,
    )
    f["promise_seen"] = int(promise_pos is not None)
    if promise_pos is None:
        f["promise_turn_idx"] = math.nan
        f["promise_position_ratio"] = math.nan
        f["turns_after_promise"] = math.nan
    else:
        f["promise_turn_idx"] = prefix[promise_pos].turn_idx
        f["promise_position_ratio"] = _ratio(promise_pos, len(prefix) - 1) if len(prefix) > 1 else math.nan
        f["turns_after_promise"] = len(prefix) - 1 - promise_pos

    # Did the promise arrive right after the house pushed for a minimum?
    if promise_pos is not None and promise_pos > 0:
        f["promise_follows_minimum_ask"] = int(prefix[promise_pos - 1].makes_minimum_ask)
        f["promise_follows_date_question"] = int(prefix[promise_pos - 1].asks_date)
    else:
        f["promise_follows_minimum_ask"] = math.nan
        f["promise_follows_date_question"] = math.nan

    # Negotiation: did the borrower push back on the house's number? Each
    # borrower amount is compared with the house amount standing at that moment
    # (the most recent one before it), not the house's first figure -- that is
    # usually the overdue statement, and a borrower accepting a later minimum
    # ask is agreement, not a counteroffer.
    pairs: list[tuple[float, float]] = []  # (house figure in force, borrower figure)
    house_in_force: float | None = None
    for r in prefix:
        if r.is_customer:
            if house_in_force is not None:
                pairs.extend((house_in_force, a.value) for a in r.amounts)
        elif r.amounts:
            house_in_force = r.amounts[-1].value
    f["cust_counteroffered"] = int(any(abs(c - h) > 1e-6 for h, c in pairs))
    f["cust_to_house_amount_ratio"] = (
        _ratio(pairs[0][1], pairs[0][0]) if pairs else math.nan
    )

    # Exchanges carrying substance after the first house proposal.
    first_house_prop = next(
        (
            i
            for i, r in enumerate(prefix)
            if not r.is_customer and (r.dates or r.amounts)
        ),
        None,
    )
    f["n_negotiation_exchanges"] = (
        sum(
            1
            for r in prefix[first_house_prop + 1 :]
            if r.is_customer and (r.dates or r.amounts)
        )
        if first_house_prop is not None
        else 0
    )

    f["cust_max_amount_stated"] = _max([a.value for a in cust_amounts])
    f["cust_min_amount_stated"] = _min([a.value for a in cust_amounts])
    f["cust_first_amount_stated"] = cust_amounts[0].value if cust_amounts else math.nan
    # Where the negotiation actually landed. In the real call AT0000026 the
    # borrower opens at 4,200 and settles at the bot's 2,800 minimum -- the last figure is the
    # one the follow-up should be built around, so it is kept separately from
    # the opening figure.
    f["cust_last_amount_stated"] = cust_amounts[-1].value if cust_amounts else math.nan

    # ----------------------------------------------------------------- timing
    cust_pauses = [r.pause_before_ms for r in cust]
    f["cust_pause_mean_ms"] = _mean(cust_pauses)
    f["cust_pause_max_ms"] = _max(cust_pauses)
    f["cust_pause_min_ms"] = _min(cust_pauses)
    f["cust_pause_std_ms"] = _std(cust_pauses)
    f["cust_pause_last_ms"] = cust[-1].pause_before_ms if cust else math.nan

    cust_durs = [r.duration_ms for r in cust]
    f["cust_turn_dur_mean_ms"] = _mean(cust_durs)
    f["cust_turn_dur_std_ms"] = _std(cust_durs)

    # Speech rate as a hesitation proxy -- we have no audio, so characters per
    # second of speaking time stands in for pace.
    rates = [
        _ratio(r.n_chars, r.duration_ms / 1000.0)
        for r in cust
        if r.duration_ms and r.duration_ms > 0
    ]
    f["cust_speech_rate_mean"] = _mean(rates)
    f["cust_speech_rate_std"] = _std(rates)
    f["cust_speech_rate_last"] = rates[-1] if rates else math.nan

    confs = [r.asr_confidence for r in cust]
    f["cust_asr_conf_mean"] = _mean(confs)
    f["cust_asr_conf_min"] = _min(confs)
    f["cust_asr_conf_last"] = cust[-1].asr_confidence if cust else math.nan

    cust_ms = sum(r.duration_ms for r in cust)
    house_ms = sum(r.duration_ms for r in house)
    f["cust_talk_ms"] = cust_ms
    f["house_talk_ms"] = house_ms
    f["cust_talk_share"] = _ratio(cust_ms, cust_ms + house_ms)
    f["silence_share"] = _ratio(
        sum(v for v in _clean([r.pause_before_ms for r in prefix])), prefix[-1].end_ms
    )

    # Promise-turn timing, z-scored within the call so it is comparable across
    # callers who are simply fast or slow talkers.
    if promise_pos is not None:
        p = prefix[promise_pos]
        f["pause_on_promise_ms"] = p.pause_before_ms
        mu, sd = f["cust_pause_mean_ms"], f["cust_pause_std_ms"]
        f["pause_on_promise_z"] = (
            (p.pause_before_ms - mu) / sd
            if not any(math.isnan(x) for x in (p.pause_before_ms, mu, sd)) and sd > 0
            else math.nan
        )
        f["asr_conf_on_promise"] = p.asr_confidence
        f["asr_conf_dip_on_promise"] = (
            f["cust_asr_conf_mean"] - p.asr_confidence
            if not any(math.isnan(x) for x in (f["cust_asr_conf_mean"], p.asr_confidence))
            else math.nan
        )
        f["speech_rate_on_promise"] = (
            _ratio(p.n_chars, p.duration_ms / 1000.0) if p.duration_ms else math.nan
        )
        f["promise_turn_words"] = p.n_words
    else:
        for key in (
            "pause_on_promise_ms",
            "pause_on_promise_z",
            "asr_conf_on_promise",
            "asr_conf_dip_on_promise",
            "speech_rate_on_promise",
            "promise_turn_words",
        ):
            f[key] = math.nan

    return f


# Categorical features, declared explicitly so downstream encoding is not left
# to dtype inference.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "first_date_speaker",
    "first_concrete_date_speaker",
    "first_amount_speaker",
    "obs_lang_dominant",
)


def feature_names() -> list[str]:
    """Stable ordered list of conversational feature names, derived by running
    the extractor on a minimal synthetic call. Used by the feature dictionary
    so documentation can never drift from the code."""
    dummy = prepare_turns(
        [
            {
                "turn_idx": 0,
                "speaker": "bot",
                "start_ms": 0,
                "end_ms": 1000,
                "pause_before_ms": 0,
                "text": "by which date can you pay",
                "asr_confidence": 1.0,
            },
            {
                "turn_idx": 1,
                "speaker": "customer",
                "start_ms": 1200,
                "end_ms": 2400,
                "pause_before_ms": 200,
                "text": "i will try next week rs 2,000",
                "asr_confidence": 0.8,
            },
        ]
    )
    return list(extract_prefix_features(dummy, 1).keys())
