"""
Date and amount detection inside borrower utterances.

These two detectors carry more weight than anything else in Phase 2, because
three high-value features depend on them:

  * who first proposed the date         -> escape vs negotiated promise
  * who first proposed the amount       -> agent-pushed vs borrower-offered
  * did the customer EVER state a date
    or an amount                        -> the agent_recorded_or_pushed detector

That last one is close to deterministic: if, across the whole call, the borrower
never uttered a date or an amount, then whatever got logged as a promise was
constructed by the agent or the bot, not given by the borrower.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicons import LEXICONS, normalise

# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------

# Currency markers, romanised and symbolic.
_CURRENCY = r"(?:rs|rs\.|inr|rupees|rupee|rupaye|rupaya|rupay|raksha|₹)"

# A number with thousands separators: 4,200 / 12,200 / 1,00,000 (Indian grouping)
_GROUPED_NUMBER = r"\d{1,3}(?:,\d{2,3})+"

# A bare run of digits, optionally with a decimal part ("4200.50").
_BARE_NUMBER = r"\d+(?:\.\d{1,2})?"

# Scale words that turn a small number into a large amount: "5 hazaar", "2 lakh".
_SCALE_WORDS: dict[str, int] = {
    "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000, "thousand": 1_000,
    "saavira": 1_000, "savira": 1_000, "sahasra": 1_000,
    "lakh": 100_000, "lac": 100_000, "laksha": 100_000,
    "crore": 10_000_000, "koti": 10_000_000,
}
_SCALE_RE = "|".join(sorted(_SCALE_WORDS, key=len, reverse=True))

# Three ways an amount can appear, tried in this order:
#   1. currency marker adjacent to a number        -> "Rs 12,200", "4200 rupees"
#   2. a number followed by a scale word           -> "5 hazaar"
#   3. a grouped or large bare number              -> "4,200", "2800"
_AMOUNT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?<!\w){_CURRENCY}\s*((?:{_GROUPED_NUMBER})|(?:{_BARE_NUMBER}))(?!\w)"
    ),
    re.compile(
        rf"(?<!\w)((?:{_GROUPED_NUMBER})|(?:{_BARE_NUMBER}))\s*{_CURRENCY}(?!\w)"
    ),
    re.compile(rf"(?<!\w)({_BARE_NUMBER})\s*(?:{_SCALE_RE})(?!\w)"),
    re.compile(rf"(?<!\w)({_GROUPED_NUMBER})(?!\w)"),
    re.compile(rf"(?<!\w)({_BARE_NUMBER})(?!\w)"),
)

# Contexts where a number is definitely NOT money. Checked against the window of
# text immediately following the number.
_NON_MONEY_FOLLOWERS = re.compile(
    r"^\s*(?:"
    r"din|dina|dinadinda|dinagala|days?|day|roz|hafte?|weeks?|week|"
    r"mahine?|months?|month|tingal|saal|years?|year|varsha|"
    r"baje|oclock|o clock|am|pm|"
    r"tarikh|tareekh|"               # "3 tarikh" is a date, handled below
    r"loans?|emis?|times?|baar|sala|"
    r"percent|%"
    r")(?!\w)"
)


@dataclass(frozen=True)
class AmountMention:
    value: float
    raw: str
    start: int


def _to_number(raw: str, scale: int = 1) -> float | None:
    try:
        return float(raw.replace(",", "")) * scale
    except ValueError:
        return None


def find_amounts(normalised_text: str, *, min_bare: int = 100) -> list[AmountMention]:
    """
    All money amounts in an already-normalised utterance.

    `min_bare` guards against reading small bare numbers as rupees. "7 dinadinda"
    (7 days) and "2 loans" must not register as amounts; "2,800" and "4200" must.
    A number carrying a currency marker or a scale word bypasses the threshold,
    because "rs 50" is unambiguously money.
    """
    if not normalised_text:
        return []

    # A number that is part of a date is not money. Blank those spans out
    # first, otherwise "2026-04-29" yields an amount of Rs 2026 and "8 april"
    # competes with the real figure in the same sentence. Masking preserves
    # string length, so reported offsets still line up with the input.
    text = _mask_date_spans(normalised_text)

    found: list[AmountMention] = []
    claimed: list[tuple[int, int]] = []  # spans already consumed

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    for idx, pattern in enumerate(_AMOUNT_PATTERNS):
        explicit = idx <= 2  # currency-marked or scale-worded
        for m in pattern.finditer(text):
            s, e = m.span()
            if overlaps(s, e):
                continue

            raw = m.group(1)
            scale = 1
            if idx == 2:  # number + scale word
                tail = text[m.end(1) : e].strip()
                scale = next(
                    (v for k, v in _SCALE_WORDS.items() if tail.startswith(k)), 1
                )

            value = _to_number(raw, scale)
            if value is None:
                continue

            if not explicit:
                # Bare number: reject if a unit word follows, or if too small.
                if _NON_MONEY_FOLLOWERS.match(text[e:]):
                    continue
                if value < min_bare:
                    continue

            claimed.append((s, e))
            found.append(AmountMention(value=value, raw=raw, start=s))

    found.sort(key=lambda a: a.start)
    return found


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

# Short forms "mon", "wed", "sat" and "sun" are deliberately absent: in
# code-mixed speech they are far more often words ("sun" = listen, "sat" as a
# spelling of saat = seven) than weekdays, and each false hit fabricates a
# concrete date.
_WEEKDAYS = (
    "monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    "tue|tues|thu|thur|thurs|fri|"
    "somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar|"
    "somavara|mangalavara|budhavara|guruvara|shukravara|shanivara|bhanuvara"
)

# "today / tomorrow / day after", across the three languages. Longer forms come
# first so "kal hi" is claimed as one span rather than as "kal".
# Note "kal" means both yesterday and tomorrow in Hindi; tense is not resolved.
_RELATIVE_CONCRETE = (
    "day after tomorrow|day after|today|tomorrow|tonight|"
    "aaj hi|kal hi|aaj|aj|kal|parso|parson|"
    "matte naale|ivattu|ivatte|naale|nale|naadiddu|naadidu|nadidu"
)

# A date that lands in a calendar.
_CONCRETE_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 3 april / 3rd april / april 3
    re.compile(rf"(?<!\w)\d{{1,2}}\s*(?:st|nd|rd|th)?\s+(?:{_MONTHS})(?!\w)"),
    re.compile(rf"(?<!\w)(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?!\w)"),
    # ISO first, so it claims the whole span: without this, 2026-04-29 matches
    # only as "04-29" and the leading 2026 is left exposed to the amount
    # detector as a four-digit rupee figure.
    re.compile(r"(?<!\w)\d{4}-\d{1,2}-\d{1,2}(?!\w)"),
    # 03/04, 03-04-2026
    re.compile(r"(?<!\w)\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?(?!\w)"),
    # "5 tarikh" / "5 tarikhko" -- a day-of-month reference
    re.compile(r"(?<!\w)\d{1,2}\s*(?:tarikh|tareekh|tarikhko|tareekhko)(?!\w)"),
    # weekday names
    re.compile(rf"(?<!\w)(?:{_WEEKDAYS})(?!\w)"),
    # today / tomorrow / day after
    re.compile(rf"(?<!\w)(?:{_RELATIVE_CONCRETE})(?!\w)"),
    # "salary day" style references tied to a known event
    re.compile(r"(?<!\w)(?:salary day|pay day|payday|salary date)(?!\w)"),
)

# Vague dates reuse the lexicon so there is exactly one definition in the codebase.
_VAGUE_DATE_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(normalise(t)).replace(r"\ ", r"\s+")
        for t in sorted(LEXICONS["vague_date"], key=len, reverse=True)
        if normalise(t)
    )
    + r")(?!\w)"
)


def _mask_date_spans(text: str) -> str:
    """
    Replace every concrete-date span with spaces, preserving string length.

    Used by `find_amounts` so that the digits inside a date are never counted
    as money. Length preservation keeps reported character offsets valid.
    """
    if not text:
        return text
    chars = list(text)
    for pattern in _CONCRETE_DATE_PATTERNS:
        for m in pattern.finditer(text):
            for i in range(m.start(), m.end()):
                chars[i] = " "
    return "".join(chars)


@dataclass(frozen=True)
class DateMention:
    raw: str
    start: int
    concrete: bool


def find_dates(normalised_text: str) -> list[DateMention]:
    """
    All date references in an utterance, each flagged concrete or vague.

    Concrete means it can be written on a calendar ("3 april", "kal", "friday").
    Vague means it cannot ("next week", "month end", "amele").

    The distinction matters: a concrete date is a checkable commitment, a vague
    one is usually a soft deflection.
    """
    if not normalised_text:
        return []

    found: list[DateMention] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    for pattern in _CONCRETE_DATE_PATTERNS:
        for m in pattern.finditer(normalised_text):
            if overlaps(*m.span()):
                continue
            claimed.append(m.span())
            found.append(DateMention(raw=m.group(0), start=m.start(), concrete=True))

    for m in _VAGUE_DATE_PATTERN.finditer(normalised_text):
        if overlaps(*m.span()):
            continue
        claimed.append(m.span())
        found.append(DateMention(raw=m.group(0), start=m.start(), concrete=False))

    found.sort(key=lambda d: d.start)
    return found


# --------------------------------------------------------------------------
# Bot/agent question detection
# --------------------------------------------------------------------------
#
# Needed for two features: latency to the date question (how long the borrower
# took to answer the single most important question in the call), and whether a
# promise arrived immediately after a minimum-amount demand.

_DATE_QUESTION = re.compile(
    r"(?:"
    r"by which date|which date|what date|when (?:can|will) you|when will you pay|"
    r"by when|date (?:kab|yaava|yava)|"
    r"kab tak|kab karoge|kab doge|kab bharoge|kis din|kaunse din|"
    r"yaava dina|yava dina|yaava dinaankadolage|yava dinaankadolage|"
    r"yaavaga|yavaga|dinaanka"
    r")"
)

_MINIMUM_ASK = re.compile(
    r"(?:"
    r"at least|atleast|minimum|part payment|partial payment|some amount|"
    r"kam se kam|kuch to|thoda to|kanishta|kanista|swalpa adaru|"
    r"today itself|aaj hi kar|ivattu kattibidi|ivatte katti"
    r")"
)


def is_date_question(normalised_text: str) -> bool:
    """Did this bot/agent turn ask the borrower for a date?"""
    return bool(normalised_text) and bool(_DATE_QUESTION.search(normalised_text))


def is_minimum_ask(normalised_text: str) -> bool:
    """Did this bot/agent turn push for a minimum or part payment?"""
    return bool(normalised_text) and bool(_MINIMUM_ASK.search(normalised_text))
