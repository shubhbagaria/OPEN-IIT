"""Decision layer: guardrails -> in-call guidance -> compliance filter -> parking schedule.

Order matters and is fixed by design:
  1. Regulatory guardrails run FIRST and can only *soften* treatment
     (hardship -> restructuring / human; third party -> zero disclosure).
  2. The guidance script is chosen from the (guard-railed) category.
  3. Every utterance passes a deterministic compliance filter before it is shown/spoken.
  4. The disposition engine sets parking/reminders from category x calibrated keep probability.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .config import CATEGORIES, CONTACT_HOUR_END, CONTACT_HOUR_START

AGGRESSIVE = {"escape", "repeat_promiser"}


def _inr(x):
    return f"{int(round(x)):,}"


def token_amount(promised_amount, overdue=None):
    """Rs 1,000-2,000 token ask, never more than 40% of the promise, rounded to 100."""
    base = min(2000.0, max(1000.0, 0.2 * promised_amount))
    base = min(base, 0.4 * promised_amount) if promised_amount > 0 else base
    return max(100.0, round(base / 100) * 100)


def post_salary_day(salary_day, offset=2):
    if salary_day is None or (isinstance(salary_day, float) and np.isnan(salary_day)):
        return None
    d = int(salary_day) + offset
    return d - 30 if d > 30 else d


def _ordinal(d):
    return f"{d}{'th' if 11 <= d % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


# ------------------------------------------------------------------ guidance scripts
SCRIPTS = {
    "genuine_feasible": {
        "hinglish": "Bahut badhiya, {date} ko ₹{amount} confirm hai. Main abhi WhatsApp pe payment link bhej raha hoon, ek din pehle reminder bhi aayega.",
        "kanglish": "Thumba olledu, {date} ge ₹{amount} confirm. Eega WhatsApp nalli payment link kalistini, ondu dina modalu reminder barutte.",
        "english": "Great, ₹{amount} on {date} is confirmed. I am sending the payment link on WhatsApp now and you will get a reminder a day before.",
    },
    "genuine_infeasible": {
        "hinglish": "Main samajh sakta hoon, aapki situation mushkil hai. Kya hum date salary ke baad {alt_date} rakh dein, ya aaj sirf ₹{token} jitna ho sake? Zaroorat ho toh main restructuring option bhi check karwa deta hoon.",
        "kanglish": "Nimma paristhiti arthavaagutte. Salary aada mele {alt_date} ge date maadona, illa andre ivattu ₹{token} maatra saaku. Beku andre restructuring option kooda nodona.",
        "english": "I understand this is a difficult time. Shall we move the date to {alt_date}, after your salary, or would a small ₹{token} today be manageable? I can also check a restructuring option for you.",
    },
    "escape": {
        "hinglish": "Bas ek minute sir. Main abhi ₹{token} ka link bhej raha hoon, abhi kar denge toh aapka account update ho jayega aur reminder calls kam ho jayengi. Baaki {date} tak. Theek hai?",
        "kanglish": "Ondu nimisha sir. ₹{token} link eega kalistini, eega maadidre account update aagutte, reminder call kammi aagutte. Baaki {date} olage. Sari na?",
        "english": "Just one minute please. I am sending a ₹{token} link right now; paying it now updates your account and reduces follow-up calls. The rest by {date}. Is that okay?",
    },
    "third_party": {
        "hinglish": "Koi baat nahi. [NAME] ji se baat karne ka sahi samay kya rahega? Main tab dobara call kar loonga. Dhanyavaad.",
        "kanglish": "Paravagilla. [NAME] avrige call maadalu sari samaya yaavudu? Aaga matte call maadtini. Dhanyavaadagalu.",
        "english": "No problem. What would be a good time to reach [NAME] directly? I will call back then. Thank you.",
    },
    "repeat_promiser": {
        "hinglish": "Sir, pichle commitments poore nahi ho paaye, isliye is baar ek chhota part payment ₹{token} aaj zaroori hai taaki promise record ho sake. Nahi hua toh account senior recovery team ke review mein jaa sakta hai. Link bhej doon?",
        "kanglish": "Sir, hinde maadida commitment aagilla, adakke ee sala ivattu ₹{token} part payment beku, aaga promise record aagutte. Illa andre account senior recovery team review ge hogabahudu. Link kalisla?",
        "english": "Sir, the earlier commitments could not be met, so this time a part payment of ₹{token} today is needed to record the promise. Otherwise the account may go to the senior recovery team for review. Shall I send the link?",
    },
    "agent_recorded_or_pushed": {
        "hinglish": "Sir, main confirm karna chahta hoon - kya aap khud {date} ko ₹{amount} pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye.",
        "kanglish": "Sir, confirm maadkollona - neeve {date} ge ₹{amount} kattalu aagutta? Illa andre nimage aaguva date heli.",
        "english": "Just to confirm in your own words - will you be able to pay ₹{amount} on {date}? If not, please tell me a date that works for you.",
    },
}
HARDSHIP_SAFE = SCRIPTS["genuine_infeasible"]
WAIT_SCRIPT = {
    "hinglish": "(Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?')",
    "kanglish": "(Listen - customer avara maathinalli date keli: 'Yaava dinaankadolage maadtira?')",
    "english": "(Listen - ask the customer to state their own date: 'Which date works for you?')",
}

# ------------------------------------------------------------------ compliance filter
BANNED = re.compile(
    r"police|jail|arrest|fir\b|court case|ghar aa(?:ke|yenge)|beizzat|izzat|naukri chali|office aa|tell your family|"
    r"relatives ko bata|sharam|\bdhoka\b|fraud|maar|threat|neighbou?rs",
    re.IGNORECASE,
)
DEBT_WORDS = re.compile(r"loan|emi|overdue|baaki|due|₹|rs\b|rupay|rupees|payment|bhugtan|outstanding|cibil|bounce|kattu|kodi",
                        re.IGNORECASE)


def compliance_check(utterance, category, capture_hour=None):
    issues = []
    if BANNED.search(utterance):
        issues.append("FPC_abusive_or_threatening_language")
    if category == "third_party" and DEBT_WORDS.search(utterance):
        issues.append("DPDP_third_party_debt_disclosure")
    if capture_hour is not None and not (CONTACT_HOUR_START <= capture_hour < CONTACT_HOUR_END):
        issues.append("FPC_contact_outside_0800_1900")
    return issues


# ------------------------------------------------------------------ guardrails
@dataclass
class Decision:
    category_model: str
    category_final: str
    category_conf: float
    p_keep: float
    guardrails: list = field(default_factory=list)
    utterance: str = ""
    compliance_issues: list = field(default_factory=list)
    parking_days: int = 0
    reminders: list = field(default_factory=list)
    queue: str = ""
    follow_up: str = ""


def apply_guardrails(cat, feats):
    """Can only move a case to a *less* aggressive treatment, never the opposite."""
    rails = []
    hardship = (feats.get("n_c_hardship", 0) or 0) > 0 or (feats.get("hardship_30d", 0) or 0) > 0
    severe = (feats.get("n_c_severe_hardship", 0) or 0) > 0
    third = (feats.get("n_c_third_party", 0) or 0) > 0 or (feats.get("n_c_wrong_number", 0) or 0) > 0
    if third and cat != "third_party":
        rails.append("DPDP_third_party_detected->no_disclosure")
        cat = "third_party"
    elif cat in AGGRESSIVE and (severe or hardship):
        rails.append(f"RBI_FPC_hardship_signal->{cat}_softened_to_genuine_infeasible")
        cat = "genuine_infeasible"
    if severe:
        rails.append("vulnerable_customer->human_review")
    return cat, rails


def guidance(cat, lang, feats, p_keep, promised_amount, promised_date_str, salary_day=None):
    lang = lang if lang in ("hinglish", "kanglish", "english") else "hinglish"
    if cat == "genuine_feasible" and p_keep >= 0.5:
        tmpl = SCRIPTS["genuine_feasible"][lang]
    else:
        tmpl = SCRIPTS[cat][lang]
    sd = feats.get("salary_day_said")
    sd = sd if sd is not None and not (isinstance(sd, float) and np.isnan(sd)) else salary_day
    alt = post_salary_day(sd)
    alt_date = _ordinal(alt) if alt else ("salary ke 2 din baad" if lang == "hinglish" else
                                          "salary aada 2 dina amele" if lang == "kanglish" else
                                          "two days after your salary")
    return tmpl.format(amount=_inr(promised_amount), token=_inr(token_amount(promised_amount)),
                       date=promised_date_str, alt_date=alt_date)


# ------------------------------------------------------------------ dynamic parking engine
def parking_plan(cat, p_keep, promise_lag_days, rails):
    """Replace the fixed 'park until PTP date' rule with a credibility-weighted schedule.
    Returns (parking_days, reminders, queue, follow_up)."""
    lag = max(int(promise_lag_days), 0)
    if any("human_review" in r for r in rails):
        return min(lag, 3), ["D-1 soft reminder"], "hardship_desk_human", "restructuring offer by trained agent"
    if cat == "third_party":
        return 0, [], "rpc_confirmation", "call borrower directly within 24h to confirm; no debt details to 3P"
    if cat == "agent_recorded_or_pushed":
        return 1, ["D+0 WhatsApp self-confirmation"], "integrity_review", "independent confirm call (different agent/bot) within 24h"
    if cat == "genuine_infeasible":
        return lag, ["D-1 reminder", "salary-day nudge"], "restructuring_review", "reschedule to salary+2 / smaller amount; no escalation"
    if cat == "repeat_promiser":
        return min(lag, 2), ["D+0 link", "D+1 reminder"], "senior_caller", "stronger treatment; part payment ask, shorter window"
    if cat == "escape":
        return min(lag, 2 if p_keep >= 0.3 else 1), ["D+0 token-payment link", "D+1 reminder"], "early_followup", \
            "follow up in 24-48h, not on PTP date"
    # genuine_feasible
    if p_keep >= 0.6:
        return lag, ["D-1 reminder"], "park", "do not disturb until promised date"
    if p_keep >= 0.35:
        return max(lag - 1, 0), ["D-2 WhatsApp", "D-1 reminder"], "park_with_check", "light-touch pre-due check"
    return min(lag, 3), ["D+0 link", "D-1 reminder"], "early_followup", "credibility low despite category; check in 72h"
