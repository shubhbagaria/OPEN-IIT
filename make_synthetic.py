#!/usr/bin/env python3
"""
Schema-exact synthetic data for testing the Phase 2 pipeline.

Every column name, dtype and value vocabulary here was taken from the real
files. Transcripts are built from templates in all three language styles,
including the specific constructions seen in the real samples (Kannada
"party payment kottre", Hindi "dekhta hoon", the English "don't send a link
I will do it myself").

PURPOSE: to prove the pipeline is correct. The numbers it produces are
meaningless -- run the pipeline on the real CSVs for real numbers.

Deliberate properties, so the tests have something to detect:
  * some calls where the borrower never states a date or amount  (agent-pushed)
  * some promises dated just before salary credit                (infeasible)
  * nach_represent payments that would wrongly count as "kept"
  * a borrower registered as `english` who speaks Hinglish       (lexicon gating)
  * accounts appearing with multiple promises                    (as-of tests)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261006
RNG = random.Random(SEED)
NP_RNG = np.random.default_rng(SEED)

LANGUAGES = ["hinglish", "kanglish", "english"]
PORTFOLIOS = ["pl_salaried", "msme", "credit_card", "microfinance", "two_wheeler"]
INCOME_TYPES = ["salaried", "self_employed"]
BUCKETS = ["X", "1-30", "31-60", "61-90", "90+"]
BUREAU = ["300-549", "550-649", "650-699", "700-749", "750+"]
BOUNCE = ["insufficient_funds", "mandate_not_registered", "account_closed", "signature_mismatch"]
PAY_CHANNELS = ["upi_app", "bbps", "neft", "branch_cash", "nach_represent"]
TOWNS = ["T1", "T2", "T3"]


# --------------------------------------------------------------------------
# Transcript templates
# --------------------------------------------------------------------------
#
# Each template is (speaker, text). {amt} and {date} are filled in.
# Every template is tagged with the behaviour it is meant to represent so the
# tests can assert the pipeline detects it.

BOT_OPEN = {
    "hinglish": "Namaste, yeh [LENDER] ki taraf se automated call hai. Kya main [NAME] se baat kar raha hoon?",
    "kanglish": "Namaskara, idu [LENDER] inda ondu automated call. Neevu [NAME] avra?",
    "english": "Hello, this is an automated call from [LENDER]. Am I speaking with [NAME]?",
}
BOT_DUE = {
    "hinglish": "Aapke account par {amt} rupees ka payment {dpd} dino se baaki hai.",
    "kanglish": "Nimma account nalli {amt} rupees payment {dpd} dinadinda baaki ide.",
    "english": "An amount of Rs {amt} has been due on your account for {dpd} days.",
}
BOT_ASK_DATE = {
    "hinglish": "Aap kis tarikh tak payment karenge?",
    "kanglish": "Yaava dinaankadolage payment maadtira?",
    "english": "By which date can you make the payment?",
}
BOT_MIN_ASK = {
    "hinglish": "Poora nahi ho to kam se kam {part} rupees aaj hi bhar dijiye.",
    "kanglish": "Poora aagilla andre kanishta {part} rupees ivattu kattibidi.",
    "english": "If not the full amount, please pay at least Rs {part} today.",
}
BOT_CONFIRM = {
    "hinglish": "Confirm kar raha hoon, {date} tak {amt} rupees.",
    "kanglish": "Confirm maadtiddini, {date} olage {amt} rupees.",
    "english": "Confirming, Rs {amt} by {date}.",
}
BOT_LINK = {
    "hinglish": "Aapke number par payment link bhej raha hoon.",
    "kanglish": "Nimma number ge payment link barutte.",
    "english": "You will get a payment link on this number.",
}
BOT_CLOSE = {
    "hinglish": "Aapka samay dene ke liye dhanyavaad.",
    "kanglish": "Thank you sir.",
    "english": "Thanks for your time.",
}

# Borrower response sets, by behaviour.
CUST_RESPONSES = {
    # Borrower states a concrete date AND the full amount -- exercises the
    # amount-extraction path end to end.
    "genuine_feasible": {
        "hinglish": ["Haan ji boliye", "{date} ko pakka {amt} rupees kar dunga", "Haan link bhej dijiye", "Theek hai"],
        "kanglish": ["Haudu enu vishaya", "{date} olage khanditha {amt} rupees kodtini", "Haan, link kalsi", "Aytu"],
        "english": ["Yes, speaking", "I will definitely pay {amt} by {date}", "Yes send the link", "Okay"],
    },
    # Borrower counteroffers a smaller figure than the house asked for --
    # exercises cust_counteroffered and cust_to_house_amount_ratio.
    "genuine_infeasible": {
        "hinglish": ["Haan bataiye", "Business bahut down hai, salary aayi toh {date} ko karunga", "Paise nahi hai abhi", "Abhi sirf {part} rupees de sakta hoon"],
        "kanglish": ["Haudu heli", "Business tumba down ide. Party payment kottre {date} olage maadtini", "Duddu illa eega", "Eega {part} rupees matra kodbahudu"],
        "english": ["Yes tell me", "Business is very down, if I get paid I will do it by {date}", "No money right now", "I can only pay {part} now"],
    },
    "escape": {
        "hinglish": ["Haan haan", "Dekhta hoon, next week try karunga", "Accha accha, mujhe jaana hai", "Link mat bhejo main khud kar lunga"],
        "kanglish": ["Haudu haudu", "Nodtini, munde vaara try madtini", "Sari sari, naanu hogbeku", "Link beda naane maadtini"],
        "english": ["Yes yes", "I will see, maybe next week", "Okay okay, I have to go", "Don't send a link I will do it myself"],
    },
    "third_party": {
        "hinglish": ["Woh ghar pe nahi hai, main unka bhai hoon", "Main unko bol dunga", "Woh {date} ko kar denge", "Theek hai"],
        "kanglish": ["Avaru illa, naanu avara anna", "Avrige helthini", "Avru {date} olage kodtare", "Aytu"],
        "english": ["He is not here, I am his brother", "I will tell him", "He will pay by {date}", "Okay"],
    },
    # The borrower never states a date or an amount. The bot records one anyway.
    "agent_pushed": {
        "hinglish": ["Hmm", "Haan", "Accha", "Theek hai"],
        "kanglish": ["Hmm", "Haudu", "Sari", "Aytu"],
        "english": ["Hmm", "Yes", "Okay", "Fine"],
    },
}

BEHAVIOURS = list(CUST_RESPONSES.keys())


def _speech_style(behaviour: str) -> str:
    """Which language style the borrower actually speaks. Deliberately decoupled
    from the account's registered preferred_language for ~20% of accounts, so
    the lexicon-gating test has something to catch."""
    return RNG.choice(LANGUAGES)


def build_call(call_id: str, behaviour: str, style: str, amt: int, dpd: int,
               date_str: str) -> list[dict]:
    """One call's transcript turns, with plausible timings."""
    part = max(int(amt * 0.4 // 50 * 50), 100)
    cust_lines = CUST_RESPONSES[behaviour][style]

    script: list[tuple[str, str]] = [
        ("bot", BOT_OPEN[style]),
        ("customer", cust_lines[0]),
        ("bot", BOT_DUE[style].format(amt=f"{amt:,}", dpd=dpd)),
        ("customer", cust_lines[1]),
        ("bot", BOT_ASK_DATE[style]),
        ("customer", cust_lines[2]),
    ]
    if behaviour in ("escape", "genuine_infeasible"):
        script.append(("bot", BOT_MIN_ASK[style].format(part=f"{part:,}")))
        script.append(("customer", cust_lines[3]))
    script += [
        ("bot", BOT_CONFIRM[style].format(amt=f"{amt:,}", date=date_str)),
        ("bot", BOT_LINK[style]),
        ("bot", BOT_CLOSE[style]),
    ]

    turns: list[dict] = []
    clock = RNG.randint(1000, 1600)
    for idx, (speaker, text) in enumerate(script):
        text = text.format(date=date_str, amt=f"{amt:,}", part=f"{part:,}")
        pause = RNG.randint(150, 700) if speaker == "bot" else RNG.randint(250, 2600)
        # Hesitant behaviours pause longer before committing.
        if speaker == "customer" and behaviour in ("escape", "genuine_infeasible"):
            pause += RNG.randint(300, 1500)
        start = clock + pause
        duration = int(len(text) * RNG.uniform(45, 75))
        turns.append(
            {
                "call_id": call_id,
                "turn_idx": idx,
                "speaker": speaker,
                "start_ms": start,
                "end_ms": start + duration,
                "pause_before_ms": pause,
                "text": text,
                "asr_confidence": 1.0 if speaker == "bot" else round(RNG.uniform(0.70, 0.96), 3),
            }
        )
        clock = start + duration
    return turns


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate(out_dir: Path, n_accounts: int = 400, n_ptps: int = 700) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2026-04-01")

    # ---- lenders ---------------------------------------------------------
    lenders = pd.DataFrame(
        {
            "lender_id": [f"L0{i}" for i in range(1, 7)],
            "lender_name": [f"Lender {c}" for c in "ABCDEF"],
            "lender_type": ["Private bank", "NBFC", "Fintech NBFC", "NBFC-MFI",
                            "NBFC (MSME)", "Small finance bank"],
            "kyc_address_format": ["kyc_mixed", "kyc_upper", "app_free_text",
                                   "kyc_mixed", "kyc_upper", "app_free_text"],
        }
    )

    # ---- agents ----------------------------------------------------------
    agent_rows = []
    for i in range(1, 21):
        agent_rows.append({
            "agent_id": f"TA{i:03d}", "channel": "tele",
            "language_team": RNG.choice(LANGUAGES), "town_id": "",
            "tenure_months": float(RNG.randint(1, 60)), "shift": "day",
        })
    for i in range(1, 10):
        agent_rows.append({
            "agent_id": f"FA{i:03d}", "channel": "field",
            "language_team": RNG.choice(LANGUAGES), "town_id": RNG.choice(TOWNS),
            "tenure_months": float(RNG.randint(1, 65)), "shift": "day",
        })
    agent_rows.append({
        "agent_id": "BOT01", "channel": "voice_bot", "language_team": "all",
        "town_id": "", "tenure_months": np.nan, "shift": "all",
    })
    agents = pd.DataFrame(agent_rows)
    callable_agents = [a for a in agents["agent_id"] if not a.startswith("FA")]

    # ---- accounts --------------------------------------------------------
    acc_rows = []
    for i in range(1, n_accounts + 1):
        income = RNG.choice(INCOME_TYPES)
        emi = float(RNG.randrange(1500, 25000, 50))
        bucket = RNG.choice(BUCKETS)
        n_prev = RNG.randint(0, 4)
        acc_rows.append({
            "account_id": f"AC{i:06d}",
            "lender_id": RNG.choice(list(lenders["lender_id"])),
            "portfolio": RNG.choice(PORTFOLIOS),
            "income_type": income,
            "town_id": RNG.choice(TOWNS),
            # 20% mismatch between registered and spoken language -- the
            # lexicon-gating trap.
            "preferred_language": RNG.choice(LANGUAGES),
            "bucket_start": bucket,
            "dpd_start": {"X": 0, "1-30": 18, "31-60": 45, "61-90": 75, "90+": 120}[bucket],
            "emi_amount": emi,
            "overdue_start": emi * RNG.randint(1, 3),
            "outstanding": emi * RNG.randint(8, 50),
            "salary_credit_day": float(RNG.randint(1, 28)) if income == "salaried" else np.nan,
            "bureau_score_band": RNG.choice(BUREAU),
            "other_active_loans": RNG.randint(0, 4),
            "paid_other_lenders_30d": RNG.random() < 0.2,
            "last_bounce_reason": RNG.choice(BOUNCE),
            "ability_to_pay_estimate": round(RNG.uniform(0.3, 1.0), 2),
            "prev_ptp_count": n_prev,
            "prev_ptp_broken": RNG.randint(0, n_prev) if n_prev else 0,
            "dialling_arm": RNG.choices(["rule_based", "random_contact_point"],
                                        weights=[0.9, 0.1])[0],
        })
    accounts = pd.DataFrame(acc_rows)

    # ---- splits ----------------------------------------------------------
    split_choice = NP_RNG.choice(
        ["train", "validation", "test"], size=n_accounts, p=[0.7, 0.15, 0.15]
    )
    splits = pd.DataFrame({"account_id": accounts["account_id"], "split": split_choice})

    # ---- attempts, ptps, transcripts -------------------------------------
    attempt_rows, ptp_rows, transcript_rows, post_rows = [], [], [], []
    attempt_n = 0
    ptp_n = 0

    # Deliberately give some accounts several promises, for the as-of tests.
    account_pool = list(accounts["account_id"])
    ptp_accounts = [RNG.choice(account_pool) for _ in range(n_ptps)]

    for account_id in ptp_accounts:
        acc = accounts.loc[accounts["account_id"] == account_id].iloc[0]
        behaviour = RNG.choices(BEHAVIOURS, weights=[0.30, 0.25, 0.25, 0.12, 0.08])[0]
        style = _speech_style(behaviour)
        agent_id = RNG.choice(callable_agents)

        captured = start + pd.Timedelta(
            days=RNG.randint(0, 120), hours=RNG.randint(8, 20), minutes=RNG.randint(0, 59)
        )
        horizon = RNG.randint(1, 12)
        promised_date = (captured + pd.Timedelta(days=horizon)).normalize()
        amt = int(round(acc["emi_amount"] * RNG.uniform(0.3, 1.2) / 50) * 50)
        overdue = float(acc["overdue_start"])

        attempt_n += 1
        ptp_n += 1
        attempt_id = f"AT{attempt_n:07d}"
        ptp_id = f"PTP{ptp_n:06d}"

        turns = build_call(
            attempt_id, behaviour, style, amt,
            int(acc["dpd_start"]) or RNG.randint(5, 30),
            f"{promised_date.day} {promised_date:%B}" if hasattr(promised_date, "strftime") else "5 April",
        )
        transcript_rows.extend(turns)

        talk_s = int(turns[-1]["end_ms"] / 1000)
        attempt_rows.append({
            "attempt_id": attempt_id, "account_id": account_id,
            "phone_id": f"PH{RNG.randint(1, 2000):06d}",
            "attempt_ts": captured, "channel": "voice_bot" if agent_id == "BOT01" else "tele_agent",
            "agent_id": agent_id, "dialling_arm": acc["dialling_arm"],
            "selection_propensity": round(RNG.uniform(0.3, 1.0), 3),
            "network_response": "answered", "ring_duration_s": RNG.randint(3, 20),
            "talk_duration_s": talk_s, "hangup_by": RNG.choice(["customer", "agent"]),
            "disposition": "ptp", "remark": RNG.choice(
                ["ptp taken", "customer promised", "will pay", "ptp taken"]
            ),
            "ptp_id": ptp_id, "has_transcript": True,
        })

        ptp_rows.append({
            "ptp_id": ptp_id, "account_id": account_id, "source": "call",
            "attempt_id": attempt_id, "visit_id": "",
            "channel": "voice_bot" if agent_id == "BOT01" else "tele_agent",
            "agent_id": agent_id, "captured_ts": captured,
            "promised_date": promised_date.date(), "promised_amount": float(amt),
            "overdue_at_capture": overdue,
            "answerer_recorded": "third_party" if behaviour == "third_party" else "self",
            "payment_link_sent": behaviour != "escape",
        })

        # post-call events
        post_rows.append({"ptp_id": ptp_id, "event_type": "payment_link_sent",
                          "event_ts": captured + pd.Timedelta(minutes=2)})
        if RNG.random() < (0.7 if behaviour == "genuine_feasible" else 0.25):
            post_rows.append({"ptp_id": ptp_id, "event_type": "payment_link_opened",
                              "event_ts": captured + pd.Timedelta(hours=RNG.randint(1, 48))})
        if RNG.random() < 0.5:
            post_rows.append({"ptp_id": ptp_id, "event_type": "whatsapp_confirmation_sent",
                              "event_ts": captured + pd.Timedelta(days=1)})
        if RNG.random() < 0.2:
            post_rows.append({"ptp_id": ptp_id, "event_type": "confirmation_reply_yes",
                              "event_ts": captured + pd.Timedelta(days=1, hours=5)})

    # Some field-sourced promises with no transcript at all.
    for _ in range(int(n_ptps * 0.05)):
        ptp_n += 1
        account_id = RNG.choice(account_pool)
        captured = start + pd.Timedelta(days=RNG.randint(0, 120))
        ptp_rows.append({
            "ptp_id": f"PTP{ptp_n:06d}", "account_id": account_id, "source": "field_visit",
            "attempt_id": "", "visit_id": f"FV{RNG.randint(1, 999):05d}",
            "channel": "field", "agent_id": f"FA{RNG.randint(1, 9):03d}",
            "captured_ts": captured,
            "promised_date": (captured + pd.Timedelta(days=5)).date(),
            "promised_amount": 5000.0, "overdue_at_capture": 9000.0,
            "answerer_recorded": "self", "payment_link_sent": False,
        })

    # Unanswered attempts, so agent connect-rate denominators are realistic.
    for _ in range(n_ptps * 3):
        attempt_n += 1
        attempt_rows.append({
            "attempt_id": f"AT{attempt_n:07d}",
            "account_id": RNG.choice(account_pool),
            "phone_id": f"PH{RNG.randint(1, 2000):06d}",
            "attempt_ts": start + pd.Timedelta(days=RNG.randint(0, 120),
                                               hours=RNG.randint(8, 20)),
            "channel": "voice_bot", "agent_id": RNG.choice(callable_agents),
            "dialling_arm": "rule_based",
            "selection_propensity": round(RNG.uniform(0.3, 1.0), 3),
            "network_response": "ring_no_answer", "ring_duration_s": RNG.randint(20, 50),
            "talk_duration_s": 0, "hangup_by": "network", "disposition": "no_answer",
            "remark": "ring hua, uthaya nahi", "ptp_id": np.nan, "has_transcript": False,
        })

    ptps = pd.DataFrame(ptp_rows)
    attempts = pd.DataFrame(attempt_rows)
    transcripts = pd.DataFrame(transcript_rows)
    post_events = pd.DataFrame(post_rows)

    # ---- payments --------------------------------------------------------
    pay_rows = []
    pay_n = 0
    for row in ptps.itertuples(index=False):
        behaviour_keep = RNG.random()
        if behaviour_keep < 0.45:
            pay_n += 1
            delay = RNG.choice([-1, 0, 0, 1, 2, 5, 9])
            ratio = RNG.choice([1.0, 1.0, 0.5, 0.3])
            # ~25% of payments arrive as NACH represents, which must NOT count.
            channel = RNG.choices(PAY_CHANNELS, weights=[0.3, 0.15, 0.15, 0.15, 0.25])[0]
            pay_rows.append({
                "payment_id": f"PY{pay_n:07d}", "account_id": row.account_id,
                "payment_ts": pd.Timestamp(row.promised_date) + pd.Timedelta(
                    days=int(delay), hours=RNG.randint(9, 22)
                ),
                "amount": round(float(row.promised_amount) * ratio, 2),
                "channel": channel,
            })
    payments = pd.DataFrame(pay_rows)

    # ---- annotated sample -------------------------------------------------
    sample_ids = list(ptps["ptp_id"].sample(min(200, len(ptps)), random_state=SEED))
    annotated = pd.DataFrame({
        "ptp_id": sample_ids,
        "annotated_type": [
            RNG.choice(["genuine_feasible", "genuine_infeasible", "escape",
                        "third_party", "repeat_promiser", "agent_recorded_or_pushed"])
            for _ in sample_ids
        ],
        "annotator_confidence": [RNG.choice(["high", "medium", "low"]) for _ in sample_ids],
    })

    # ---- write ------------------------------------------------------------
    files = {
        "ptps.csv": ptps, "call_transcripts.csv": transcripts,
        "post_call_events.csv": post_events, "annotated_ptp_sample.csv": annotated,
        "accounts.csv": accounts, "agents.csv": agents, "payments.csv": payments,
        "dial_attempts.csv": attempts, "splits.csv": splits, "lenders.csv": lenders,
    }
    for name, frame in files.items():
        frame.to_csv(out_dir / name, index=False)
        print(f"  {name:<28} {len(frame):>7,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data_synthetic"))
    parser.add_argument("--accounts", type=int, default=400)
    parser.add_argument("--ptps", type=int, default=700)
    args = parser.parse_args()
    print(f"Generating synthetic data in {args.out_dir} ...")
    generate(args.out_dir, args.accounts, args.ptps)
    print("Done. NOTE: for testing the pipeline only -- these numbers mean nothing.")
