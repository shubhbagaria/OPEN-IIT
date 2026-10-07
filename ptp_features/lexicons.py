"""
Multilingual cue lexicons for borrower speech.

DESIGN RULE THAT MATTERS MOST
-----------------------------
Every lexicon is applied to every utterance, regardless of the account's
`preferred_language`. That field records what language the borrower *registered*
with, not what they actually spoke, and the transcripts are code-mixed anyway --
a borrower tagged `english` routinely says "dekhta hoon". Gating the lexicon on
`preferred_language` silently drops those hits and quietly destroys recall.

`preferred_language` is still kept as a model feature, and we additionally emit
observed-script indicators so a model can learn language effects itself. It is
simply never used as a filter.

SEED VS MINED
-------------
The dictionaries below are a *seed*, hand-built from the transcript samples and
from ordinary Hindi/Kannada collections vocabulary. They are not the finished
article. `mine_candidate_terms()` at the bottom ranks n-grams from the TRAIN
split by association with the label so the seed can be extended from evidence
rather than imagination. Run it, eyeball the output, and fold confirmed terms
back into these dictionaries.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Cue categories
# --------------------------------------------------------------------------
#
# Each category maps to a flat list of surface forms. Forms are matched
# case-insensitively on word boundaries over normalised (romanised, lowercased)
# text. Multi-word forms are matched as phrases.
#
# Categories are deliberately behavioural rather than linguistic: each one is
# meant to correspond to something a collections SME would recognise.

LEXICONS: Mapping[str, tuple[str, ...]] = {
    # ---------------------------------------------------------------- hedging
    # Softening / non-committal language. The classic weak-promise marker.
    "hedge": (
        # english
        "try", "will try", "maybe", "may be", "might", "probably", "not sure",
        "we will see", "let me see", "i think", "hopefully", "should be able",
        "i guess", "more or less", "somehow", "if possible",
        # hinglish
        "dekhta", "dekhta hoon", "dekhte hain", "dekhunga", "koshish",
        "koshish karunga", "try karunga", "try karta", "sochta", "sochkar",
        "shayad", "pata nahi", "malum nahi", "ho sakta", "dekhna padega",
        "kuch kar", "jitna ho sake",
        # kanglish
        "nodtini", "nodona", "nodi", "nodoke", "gottilla", "swalpa",
        "try madtini", "try maadtini", "prayatna", "aagabahudu", "nodbeku",
        "heltini", "nodtiddini",
    ),
    # ------------------------------------------------------------ vague dates
    # A date that cannot be put in a calendar.
    "vague_date": (
        # english
        "next week", "this week", "month end", "end of month", "month ending",
        "coming week", "few days", "couple of days", "some days", "soon",
        "shortly", "later", "sometime", "next month", "in a week",
        "by month end", "within a week", "after some time",
        # hinglish
        "agle hafte", "is hafte", "hafte", "mahine ke end", "mahine ka end",
        "month end tak", "kuch din", "kuch dino", "thode din", "jaldi",
        "baad mein", "baad me", "aage", "do char din", "char paanch din",
        "baad ko",
        # kanglish
        "munde vaara", "vaara", "tingala kone", "tingla kone", "kelavu dina",
        "swalpa dina", "amele", "ammele", "aamele", "munde", "beega",
        "innu kelavu",
    ),
    # ------------------------------------------------------------ conditional
    # The promise is contingent on something outside the borrower's control.
    # Strongly associated with genuine_infeasible rather than escape.
    "conditional": (
        # english
        "if", "once", "after i get", "when i get", "provided", "depends",
        "subject to", "as soon as i get", "in case",
        # hinglish
        "agar", "agar mil", "aayi toh", "aaye toh", "mil jaye",
        "mil jaata", "salary aayi", "salary aane", "paisa aaye",
        "payment aaye", "party se", "jab mile", "jab aaye", "milte hi",
        # kanglish
        "kottre", "bandre", "bandhre", "sikkidre", "sikre", "banda mele",
        "kotta mele", "aadre", "party payment kottre", "salary bandre",
    ),
    # -------------------------------------------------------------- hardship
    # Signals of genuine inability. THIS CATEGORY DRIVES THE SAFETY GATE --
    # a hit here must never be allowed to push an account toward harsher
    # treatment. Recall matters far more than precision here.
    "hardship": (
        # Intensifier-split variants. Borrowers say "business TUMBA down ide"
        # and "business BAHUT down hai", so matching only "business down ide"
        # misses the real utterance. These head-only bigrams catch the phrase
        # whatever adverb is wedged into the middle. Recall matters far more
        # than precision in this category, because a miss here means a
        # hardship case can be escalated.
        "tumba down", "down ide", "bahut down", "down hai", "kaafi down",
        "very down", "really down", "quite down", "bohot down",
        # english
        "business down", "no business", "lost my job", "job gone", "no job",
        "laid off", "hospital", "medical", "treatment", "surgery", "illness",
        "sick", "accident", "death", "passed away", "funeral", "no money",
        "nothing left", "very difficult", "struggling", "shop closed",
        "salary not credited", "salary delayed", "not paid salary",
        # hinglish
        "business down hai", "dhanda nahi", "kaam nahi", "naukri chali gayi",
        "naukri nahi", "job chali gayi", "paisa nahi", "paise nahi",
        "paise nahi hai", "bahut mushkil", "tangi", "pareshan", "bimar",
        "bimari", "aspatal", "ilaaj", "dawai", "mrityu", "guzar gaye",
        "dukaan band", "salary nahi aayi", "tanakhwah nahi",
        # kanglish
        "business down ide", "vyapara illa", "kelsa illa", "kelsa hogide",
        "duddu illa", "hana illa", "kashta", "tumba kashta", "aspatre",
        "aarogya", "haushe illa", "angadi muchide", "sambala baridilla",
    ),
    # ------------------------------------------------------------ third party
    # The person on the line is not the borrower, or is promising for someone.
    "third_party": (
        # english
        "he is not", "she is not", "not here", "not available", "wrong number",
        "my husband", "my wife", "my son", "my father", "my brother",
        "his phone", "her phone", "i will tell him", "i will tell her",
        "he will pay", "she will pay", "owner is not", "he has gone",
        # hinglish
        "bhai ne", "pati", "patni", "beta", "pitaji", "papa", "bhaiya",
        "woh nahi hai", "ghar pe nahi", "bahar gaye", "unka phone",
        "main bol dunga", "unko bol", "woh karenge", "galat number",
        # kanglish
        "avaru illa", "avanu illa", "manege illa", "nanna ganda",
        "nanna hendathi", "maga", "appa", "anna", "avrige helthini",
        "avru kodtare", "tappu number",
    ),
    # --------------------------------------------------- call-ending pressure
    # Borrower is trying to get off the phone. Hallmark of the escape promise.
    "termination": (
        # english
        "i have to go", "have to go", "got to go", "gotta go", "busy",
        "in a meeting", "driving", "call later", "call me later",
        "not now", "later please", "bye", "ok ok", "okay okay",
        "fine fine", "yes yes yes",
        # hinglish
        "mujhe jaana", "jaana hai", "baad mein baat", "abhi busy",
        "abhi nahi", "rakhta hoon", "phone rakh", "kaam me hoon",
        "meeting me", "gaadi chala", "theek hai theek hai", "haan haan",
        "accha accha", "ho gaya na",
        # kanglish
        "hogbeku", "naanu hogbeku", "busy idini", "amele maathadona",
        "eega beda", "phone ittbidi", "saaku", "aytu aytu", "haudu haudu",
        "sari sari",
    ),
    # ------------------------------------------------------- payment-link refusal
    # Refusing the instrument that would make the promise verifiable. Short,
    # rare, and highly diagnostic -- worth its own category.
    "link_refusal": (
        # english
        "don't send", "dont send", "no need to send", "no link", "not needed",
        "i will do it myself", "i will do myself", "myself", "no need",
        "i know how", "already have",
        # hinglish
        "mat bhejo", "mat bhejna", "zaroorat nahi", "khud kar", "khud karunga",
        "khud kar lunga", "link mat", "nahi chahiye",
        # kanglish
        "kalsbeda", "kalsbedi", "naane maadtini", "naane madtini", "agatya illa",
        "link beda", "nanege gottide",
    ),
    # ------------------------------------------------------- over-agreement
    # Repeated affirmation without substance. Separate from `termination`
    # because it can occur mid-call.
    "over_agreement": (
        "haan haan", "han han", "haan ji haan", "yes yes", "ok ok", "okay okay",
        "sure sure", "theek hai theek hai", "thik hai thik hai",
        "haudu haudu", "sari sari", "aytu aytu", "done done",
    ),
    # ------------------------------------------------------------- refusal
    # Explicit inability or unwillingness, as distinct from hedging.
    "refusal": (
        # english
        "cannot", "can not", "can't", "not possible", "no way", "impossible",
        "will not", "won't", "refuse", "not paying", "i won't pay",
        # hinglish
        "nahi hoga", "nahi ho payega", "nahi kar", "mumkin nahi",
        "possible nahi", "nahi dunga", "nahi bharunga",
        # kanglish
        "aagalla", "aagolla", "saadhya illa", "madoke aagalla", "kodolla",
    ),
    # ------------------------------------------------- concrete commitment
    # Firm, specific language. The positive counterpart to `hedge`.
    "concrete_commitment": (
        # english
        "i will pay", "i am paying", "paying today", "definitely", "for sure",
        "confirmed", "guarantee", "surely", "without fail", "i promise",
        "right now", "immediately", "just now", "already paid", "doing it now",
        # hinglish
        "pakka", "zaroor", "bilkul", "abhi karta", "abhi kar deta",
        "aaj hi", "kar deta hoon", "de deta hoon", "bhar deta hoon",
        "jarur karunga", "100 percent",
        # kanglish
        "khanditha", "khandita", "khachita", "ivattu", "eega",
        "eegale", "maadtini", "kodtini", "kattibidtini", "poora kodtini",
        "ivatte", "tappade",
    ),
}

# Categories whose presence should never be allowed to escalate treatment.
SAFETY_CATEGORIES: tuple[str, ...] = ("hardship", "refusal")

ALL_CATEGORIES: tuple[str, ...] = tuple(LEXICONS.keys())


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

# Commas INSIDE a number are removed rather than turned into spaces, so that
# "4,200" becomes "4200" and not "4 200". This matters more than it looks:
# splitting the group turns "Rs 12,200" into "Rs 12", and the amount features
# are then wrong on every grouped figure in the transcripts without ever
# raising an error.
_DIGIT_GROUP_RE = re.compile(r"(?<=\d),(?=\d)")

# Strip punctuation, EXCEPT a separator sitting between two digits. That keeps
# "4200.50" (decimal), "03/04" and "2026-04-29" (dates) and "5:30" (time)
# intact, while still removing ordinary sentence punctuation.
_PUNCT_RE = re.compile(r"(?!(?<=\d)[./:-](?=\d))[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

# ASR markers that appear in the transcripts and should not be treated as words.
_ASR_MARKERS = ("[inaudible]", "[noise]", "[silence]", "[unclear]")

# The rupee sign is not a word character, so the punctuation pass would delete
# it and "₹50" would lose its currency marker. Spell it out first.
_RUPEE_RE = re.compile(r"₹\s*")

# Placeholders the vendor substitutes for PII.
_PII_PLACEHOLDERS = ("[name]", "[lender]", "[phone]", "[amount]", "[address]")


def normalise(text: str) -> str:
    """
    Lowercase, strip accents, remove ASR/PII markers and punctuation, collapse
    whitespace. Deliberately *not* stemmed: transliterated Indic text has no
    reliable stemmer, and stemming "maadtini" to "maad" loses the tense that
    distinguishes a commitment from a hedge.
    """
    if not isinstance(text, str) or not text:
        return ""
    t = text.lower()
    for marker in _ASR_MARKERS + _PII_PLACEHOLDERS:
        t = t.replace(marker, " ")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _RUPEE_RE.sub(" rs ", t)
    t = _DIGIT_GROUP_RE.sub("", t)  # join digit groups BEFORE stripping punctuation
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _compile_category(terms: Sequence[str]) -> re.Pattern[str]:
    """
    One alternation regex per category. Longest terms first so that
    "try karunga" wins over "try" and we don't double-count the same span.
    Word boundaries prevent "try" matching inside "country".
    """
    ordered = sorted({normalise(t) for t in terms if normalise(t)}, key=len, reverse=True)
    escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in ordered]
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)")


CATEGORY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    name: _compile_category(terms) for name, terms in LEXICONS.items()
}


def count_category_hits(normalised_text: str) -> dict[str, int]:
    """
    Count non-overlapping hits for every category on already-normalised text.

    All categories are always evaluated. There is no language argument, by
    design -- see the module docstring.
    """
    if not normalised_text:
        return {c: 0 for c in ALL_CATEGORIES}
    return {
        c: len(CATEGORY_PATTERNS[c].findall(normalised_text)) for c in ALL_CATEGORIES
    }


def distinct_category_terms(normalised_text: str, category: str) -> set[str]:
    """Which distinct surface forms of a category fired. Used for diversity
    features: five different hedges is a stronger signal than one hedge
    repeated five times."""
    if not normalised_text:
        return set()
    return set(CATEGORY_PATTERNS[category].findall(normalised_text))


# --------------------------------------------------------------------------
# Observed-script indicators
# --------------------------------------------------------------------------
#
# Cheap, high-precision markers of which language was actually spoken, as
# opposed to what the account says. Function words only -- they are frequent,
# unambiguous, and rarely borrowed across these three languages.

_SCRIPT_MARKERS: Mapping[str, tuple[str, ...]] = {
    "hindi": ("hai", "nahi", "kar", "mein", "mera", "aap", "hoon", "karunga",
              "paisa", "abhi", "kal", "theek", "accha", "bhai", "haan"),
    "kannada": ("ide", "illa", "maadtini", "madtini", "kodtini", "naanu",
                "nimma", "nanna", "haudu", "eega", "amele", "aytu", "swalpa",
                "dina", "beku"),
    "english": ("the", "will", "have", "please", "payment", "account",
                "tomorrow", "today", "money", "pay", "date", "sure"),
}

_SCRIPT_PATTERNS = {
    lang: _compile_category(terms) for lang, terms in _SCRIPT_MARKERS.items()
}


def observed_language_counts(normalised_text: str) -> dict[str, int]:
    """Marker counts per language, for the observed-language features."""
    if not normalised_text:
        return {lang: 0 for lang in _SCRIPT_PATTERNS}
    return {
        lang: len(pat.findall(normalised_text))
        for lang, pat in _SCRIPT_PATTERNS.items()
    }


# --------------------------------------------------------------------------
# Data-driven lexicon extension
# --------------------------------------------------------------------------


def mine_candidate_terms(
    texts: Iterable[str],
    labels: Iterable[int],
    *,
    ngram_range: tuple[int, int] = (1, 3),
    min_count: int = 15,
    top_k: int = 300,
) -> list[tuple[str, int, float, float]]:
    """
    Rank n-grams by how strongly they separate kept from broken promises.

    Returns (term, count, kept_rate, lift) sorted by |lift|, where lift is
    kept_rate(term) - kept_rate(overall).

    MUST be run on the TRAIN split only. The output is a *candidate list for a
    human to review*, not something to wire in automatically: a term can
    correlate with the label for reasons that have nothing to do with sincerity
    (a particular lender's script, a particular agent's habits).
    """
    texts = list(texts)
    labels = list(labels)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")

    lo, hi = ngram_range
    term_total: Counter[str] = Counter()
    term_kept: Counter[str] = Counter()

    for text, label in zip(texts, labels):
        tokens = normalise(text).split()
        seen: set[str] = set()
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                seen.add(" ".join(tokens[i : i + n]))
        for term in seen:
            term_total[term] += 1
            if label == 1:
                term_kept[term] += 1

    overall = (sum(labels) / len(labels)) if labels else 0.0
    rows: list[tuple[str, int, float, float]] = []
    for term, count in term_total.items():
        if count < min_count:
            continue
        rate = term_kept[term] / count
        rows.append((term, count, rate, rate - overall))

    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    return rows[:top_k]


def coverage_report(normalised_texts: Iterable[str]) -> dict[str, float]:
    """
    Share of utterances in which each category fires, plus the share firing no
    category at all.

    The stated Phase 2 objective is 100% language coverage. `uncovered_rate` is
    the number that tracks it: if it is high for one language group, the lexicon
    has a hole and `mine_candidate_terms` should be re-run on that subset.
    """
    texts = list(normalised_texts)
    if not texts:
        return {}
    hits = {c: 0 for c in ALL_CATEGORIES}
    uncovered = 0
    for t in texts:
        counts = count_category_hits(t)
        if not any(counts.values()):
            uncovered += 1
        for c, v in counts.items():
            if v:
                hits[c] += 1
    out = {f"pct_utterances_{c}": hits[c] / len(texts) for c in ALL_CATEGORIES}
    out["uncovered_rate"] = uncovered / len(texts)
    out["n_utterances"] = float(len(texts))
    return out
