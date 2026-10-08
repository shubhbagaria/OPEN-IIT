"""FastAPI application layer for the CreditNirvana PTP Intelligence web app.

Run from the repository root:
    python -m uvicorn webapp.backend.app:app --port 8000
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ptp_copilot.config import CATEGORIES, COST, MODEL_DIR  # noqa: E402
from webapp.backend.services import (AuditLog, DataStore, QwenService, TabularModel, _clean, assess,  # noqa: E402
                                     records)

app = FastAPI(title="CreditNirvana PTP Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

STORE = DataStore()
TAB = None
QWEN = QwenService()
AUDIT = AuditLog()
BOOT = time.strftime("%Y-%m-%d %H:%M:%S")


@app.on_event("startup")
def _startup():
    global TAB
    QWEN.start()                      # loads in the background; health reports progress
    try:
        STORE.load()
    except Exception as e:
        STORE.error = f"{type(e).__name__}: {e}"
        print("[data] failed:", STORE.error)
    try:
        TAB = TabularModel()
    except Exception as e:
        STORE.error = (STORE.error or "") + f" | tabular model: {e}"


def need_data():
    if not STORE.ready:
        raise HTTPException(503, detail=STORE.error or "Data is still loading")


def tab_version():
    mt = (MODEL_DIR / "models.pkl").stat().st_mtime
    return f"lgbm-incall-{time.strftime('%Y-%m-%d', time.localtime(mt))}"


def model_versions():
    return dict(tabular=tab_version(), qwen=QWEN.version()["version_tag"])


def pct(a):
    a = list(a)
    return dict(p50=round(float(np.percentile(a, 50)), 1), p95=round(float(np.percentile(a, 95)), 1), n=len(a)) if a else None


# =============================================================================== health
@app.get("/api/health")
def health():
    v = QWEN.version()
    return dict(
        backend="ok", started_at=BOOT,
        data=dict(ready=STORE.ready, error=STORE.error, loaded_at=STORE.loaded_at,
                  ptps=len(STORE.base) if STORE.ready else None,
                  replayable_calls=len(STORE.calls) if STORE.ready else None),
        tabular=dict(ready=TAB is not None, version=tab_version(),
                     measured_latency_ms=pct(TAB.latencies) if TAB else None),
        qwen=dict(status=QWEN.status, error=QWEN.error, model=v["name"], version=v["version_tag"],
                  precision=QWEN.precision, device=QWEN.device, vram_gb=QWEN.vram_gb, loaded_at=QWEN.loaded_at,
                  tokenizer_loaded=QWEN.status == "ready", adapter="LoRA r=16 merged into weights (outputs/slm_merged)",
                  temperature=QWEN.temperature, measured_latency_ms=pct(QWEN.latencies),
                  measured_generation_ms=pct(QWEN.gen_latencies),
                  benchmark=next((r for r in STORE.lat_slm["rows"] if r["precision"] == QWEN.precision), None)
                  if STORE.ready else None),
    )


# =============================================================================== overview
@app.get("/api/overview")
def overview():
    need_data()
    en, b = STORE.en, STORE.base
    R = STORE.results
    biz = R["business"]
    base_sc = next(s for s in biz["scenarios"] if abs(s["conversion_of_flagged_broken"] - 0.15) < 1e-9)
    wk = b[["ptp_id", "captured_ts"]].merge(en[["ptp_id", "p_keep_in_call", "kept"]], on="ptp_id")
    wk["week"] = wk["captured_ts"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    trend = wk.groupby("week").agg(ptps=("ptp_id", "size"), mean_confidence=("p_keep_in_call", "mean"),
                                   actual_keep_rate=("kept", "mean")).reset_index()
    status = en["status"].value_counts()
    pending = int(b["grace_censored"].sum())
    hist = np.histogram(en["p_keep_in_call"], bins=np.linspace(0, 1, 11))[0]
    hk = np.histogram(en.loc[en.kept == 1, "p_keep_in_call"], bins=np.linspace(0, 1, 11))[0]
    return dict(
        kpis=[
            dict(key="ptps", label="PTPs captured (Apr–Jun 2026)", value=len(en), fmt="int", status="ACTUAL",
                 note=f"{pending} still inside the grace window at data end (pending)"),
            dict(key="avg_conf", label="Average PTP credibility", value=float(en["p_keep_in_call"].mean()),
                 fmt="pct", status="MODEL OUTPUT", note="Calibrated in-call keep probability, out-of-fold"),
            dict(key="keep_rate", label="PTP keep rate", value=float(en["kept"].mean()), fmt="pct", status="ACTUAL",
                 note="≥90% paid by promised date + 2 days"),
            dict(key="flagged", label="Broken PTPs flagged before parking", value=biz["broken_flagged_before_parking"],
                 fmt="pct", status="MEASURED", note=f"Held-out test split; flag precision {biz['precision_of_flag']:.0%}"),
            dict(key="recovered", label="Recovered within cure window", value=float(b["paid_in_cure"].sum()),
                 fmt="inr", status="ACTUAL", note="All-channel payments credited to PTPs (+14 days)"),
            dict(key="lift", label="Recovery lift per 1,000 PTPs", value=base_sc["recovery_lift_per_1000_rs_low"],
                 value_high=base_sc["recovery_lift_per_1000_rs_high"], fmt="inr_range", status="PROJECTED",
                 note="Base scenario: 15% of flagged broken PTPs converted; requires RCT"),
            dict(key="cost", label="Escalation cost avoided per 1,000 PTPs",
                 value=base_sc["escalation_cost_avoided_per_1000"], fmt="inr", status="PROJECTED",
                 note=f"3 tele connects @₹{COST['tele_caller']} + 35% field @₹{COST['field_visit']}"),
        ],
        trend=records(trend),
        outcomes=[dict(name=k, value=int(status.get(k, 0))) for k in ["kept_on_time", "kept_late", "partial", "broken"]]
                 + [dict(name="pending", value=pending)],
        categories=[dict(name=c, predicted=int((en["predicted_category"] == c).sum()),
                         ground_truth=int((en["ground_truth_category"] == c).sum())) for c in CATEGORIES],
        confidence_hist=[dict(bin=f"{i / 10:.1f}–{(i + 1) / 10:.1f}", total=int(hist[i]), kept=int(hk[i]),
                              not_kept=int(hist[i] - hk[i])) for i in range(10)],
        model_quality=dict(keep_auc_test=R["keep_in_call"]["auc"], keep_ece_test=R["keep_in_call"]["ece_calibrated"],
                           keep_auc_oof=R["crossfit_all_ptps"]["keep_auc_in_call"],
                           category_macro_f1_test=R["category_vs_ground_truth"]["report"]["macro avg"]["f1-score"],
                           qwen_macro_f1_test=STORE.slm_eval["final"]["macro_f1"],
                           qwen_nf4_macro_f1_test=STORE.lat_slm.get("nf4_quality_test_final", {}).get("macro_f1"),
                           hardship_misclass_costsens=R["hardship_misclassified_as_aggressive_costsens"]),
    )


# =============================================================================== live calls
def _call_filter(df, q, category, risk, status, split, lang):
    if q:
        ql = q.lower()
        df = df[df["call_id"].str.lower().str.contains(ql) | df["account_id"].str.lower().str.contains(ql)
                | df["ptp_id"].str.lower().str.contains(ql) | df["agent_id"].str.lower().str.contains(ql)]
    if category:
        df = df[df["predicted_category"] == category]
    if risk:
        r = np.where(df["confidence_score"] < 0.25, "high", np.where(df["confidence_score"] < 0.5, "medium", "low"))
        df = df[r == risk]
    if status:
        df = df[df["status"] == status]
    if split:
        df = df[df["split"] == split]
    if lang:
        df = df[df["preferred_language"] == lang]
    return df


@app.get("/api/live-calls")
def live_calls(q: str = "", category: str = "", risk: str = "", status: str = "", split: str = "", lang: str = "",
               sort: str = "captured_ts", order: str = "desc", page: int = 1, size: int = 25):
    need_data()
    df = _call_filter(STORE.calls, q, category, risk, status, split, lang)
    if sort in df:
        df = df.sort_values(sort, ascending=order == "asc")
    total = len(df)
    df = df.iloc[(page - 1) * size: page * size].copy()
    df["risk"] = np.where(df["confidence_score"] < 0.25, "high", np.where(df["confidence_score"] < 0.5, "medium", "low"))
    return dict(total=total, page=page, size=size, rows=records(df),
                provenance="Recorded calls from call_transcripts.csv. Confidence/category/action = stored out-of-fold "
                           "model output (enriched_ptp_predictions.csv); open a call to run the models live.")


@app.get("/api/live-calls/scenarios")
def scenarios():
    need_data()
    c = STORE.calls[STORE.calls["split"] == "test"]
    out = []
    for cat in CATEGORIES:
        s = c[c["ground_truth_category"] == cat]
        s = s.assign(h=(s["category_source"] == "human_annotation").astype(int)).sort_values(["h", "n_turns"],
                                                                                           ascending=[False, False])
        s = s[(s["n_turns"] >= 5) & (s["n_turns"] <= 20)]   # third-party calls are short
        if len(s):
            r = s.iloc[0]
            out.append(dict(category=cat, call_id=r["call_id"], ptp_id=r["ptp_id"], label_source=r["category_source"],
                            n_turns=int(r["n_turns"]), language=r["preferred_language"]))
    return out


@app.get("/api/live-calls/{call_id}")
def live_call(call_id: str):
    need_data()
    m = STORE.calls[STORE.calls["call_id"] == call_id]
    if m.empty:
        raise HTTPException(404, "Call not found")
    r = m.iloc[0]
    b = STORE.base_by_ptp.loc[r["ptp_id"]]
    acc = STORE.accounts.set_index("account_id").loc[r["account_id"]]
    tr = STORE.calls_by_id[call_id]
    return dict(
        call=records(m)[0],
        account=dict(account_id=r["account_id"], lender=_lender(acc["lender_id"]), portfolio=acc["portfolio"],
                     bucket=acc["bucket_start"], dpd=_clean(acc["dpd_start"]), outstanding=_clean(acc["outstanding"]),
                     overdue_at_capture=_clean(b["overdue_at_capture"]), emi=_clean(acc["emi_amount"]),
                     salary_credit_day=_clean(acc["salary_credit_day"]), language=acc["preferred_language"],
                     prior_ptps=_clean(b["total_prior_ptps"]), prior_broken=_clean(b["total_prior_broken"])),
        ptp=dict(ptp_id=r["ptp_id"], promised_amount=_clean(b["promised_amount"]),
                 promised_date=pd.Timestamp(b["promised_date"]).strftime("%Y-%m-%d"),
                 captured_ts=pd.Timestamp(b["captured_ts"]).isoformat(), agent_id=b["agent_id"], channel=b["channel"]),
        transcript=records(tr[["turn_idx", "speaker", "start_ms", "end_ms", "pause_before_ms", "text",
                               "asr_confidence"]]),
        mode="LIVE DEMO SIMULATION: recorded call replayed utterance by utterance; each step re-scores the "
             "transcript prefix with the real models (no future turns are visible to the models).",
    )


def _lender(lid):
    m = STORE.lenders[STORE.lenders["lender_id"] == lid]
    return dict(id=lid, name=m["lender_name"].iloc[0] if len(m) else None, type=m["lender_type"].iloc[0] if len(m) else None)


def _ptp_for_call(call_id):
    m = STORE.calls[STORE.calls["call_id"] == call_id]
    if m.empty:
        raise HTTPException(404, "Call not found")
    return m["ptp_id"].iloc[0]


@app.post("/api/live-calls/{call_id}/step")
def step(call_id: str, body: dict = Body(default={})):
    need_data()
    ptp = _ptp_for_call(call_id)
    a = assess(STORE, TAB, QWEN, ptp, upto_turn=body.get("turn"), use_qwen=body.get("qwen", True))
    if a["is_final"] and body.get("audit", True):
        a["audit_id"] = AUDIT.write("live_simulation_final", a, model_versions())["audit_id"]
    return a


@app.post("/api/live-calls/{call_id}/predict")
def predict(call_id: str):
    need_data()
    a = assess(STORE, TAB, QWEN, _ptp_for_call(call_id))
    a["audit_id"] = AUDIT.write("full_call_prediction", a, model_versions())["audit_id"]
    return a


@app.post("/api/live-calls/{call_id}/generate")
def generate(call_id: str, body: dict = Body(default={})):
    need_data()
    from ptp_copilot import policy as P
    ptp = _ptp_for_call(call_id)
    b = STORE.base_by_ptp.loc[ptp]
    tr = STORE.calls_by_id[call_id]
    k = len(tr) - 1 if body.get("turn") is None else int(body["turn"])
    g = QWEN.generate(b.to_dict(), list(zip(tr["speaker"].iloc[:k + 1], tr["text"].iloc[:k + 1])))
    if g.get("available"):
        a = assess(STORE, TAB, QWEN, ptp, upto_turn=k, use_qwen=False)
        g["compliance_issues_raw"] = P.compliance_check(g["utterance"] or g["raw"], a["final_category"])
        g["shown_to_agent"] = False
        g["note"] = ("Raw SLM draft for transparency. The agent is shown the approved script for the guard-railed "
                     "category, never free-generated text.")
        AUDIT.write("slm_generation", a, model_versions(), extra=dict(raw=g["raw"], latency_ms=g["latency_ms"],
                                                                      compliance_issues_raw=g["compliance_issues_raw"]))
    return g


# =============================================================================== customers
@app.get("/api/customers")
def customers(q: str = "", portfolio: str = "", lender: str = "", bucket: str = "", sort: str = "n_ptps",
              order: str = "desc", page: int = 1, size: int = 25):
    need_data()
    a = STORE.accounts.copy()
    g = STORE.en.groupby("account_id").agg(n_ptps=("ptp_id", "size"), kept=("kept", "sum"),
                                           mean_confidence=("p_keep_in_call", "mean"),
                                           latest_category=("predicted_category", "last"))
    a = a.join(g, on="account_id")
    a["n_ptps"] = a["n_ptps"].fillna(0).astype(int)
    a["keep_rate"] = a["kept"] / a["n_ptps"].replace(0, np.nan)
    if q:
        a = a[a["account_id"].str.lower().str.contains(q.lower())]
    if portfolio:
        a = a[a["portfolio"] == portfolio]
    if lender:
        a = a[a["lender_id"] == lender]
    if bucket:
        a = a[a["bucket_start"] == bucket]
    if sort in a:
        a = a.sort_values(sort, ascending=order == "asc")
    total = len(a)
    a = a.iloc[(page - 1) * size: page * size]
    cols = ["account_id", "lender_id", "portfolio", "income_type", "preferred_language", "bucket_start", "dpd_start",
            "outstanding", "overdue_start", "bureau_score_band", "n_ptps", "kept", "keep_rate", "mean_confidence",
            "latest_category"]
    return dict(total=total, page=page, size=size, rows=records(a[cols]),
                filters=dict(portfolio=sorted(STORE.accounts["portfolio"].unique()),
                             lender=sorted(STORE.accounts["lender_id"].unique()),
                             bucket=["X", "1-30", "31-60", "61-90", "90+"]))


@app.get("/api/customers/{account_id}")
def customer(account_id: str):
    need_data()
    acc = STORE.accounts[STORE.accounts["account_id"] == account_id]
    if acc.empty:
        raise HTTPException(404, "Customer not found")
    a = acc.iloc[0]
    ptps = STORE.base[STORE.base["account_id"] == account_id][
        ["ptp_id", "attempt_id", "captured_ts", "promised_date", "promised_amount", "channel", "agent_id",
         "paid_in_grace", "paid_in_cure", "settle_days"]].merge(
        STORE.en[["ptp_id", "status", "kept", "predicted_category", "confidence_score", "category_confidence",
                  "queue", "parking_days", "follow_up", "guardrails", "explanation", "ground_truth_category"]],
        on="ptp_id").sort_values("captured_ts")
    ptps["has_transcript"] = ptps["attempt_id"].isin(STORE.calls_by_id)
    pays = STORE.payments[STORE.payments["account_id"] == account_id].sort_values("payment_ts")
    dials = STORE.dials[STORE.dials["account_id"] == account_id].sort_values("attempt_ts", ascending=False)
    disp = dials["disposition"].value_counts()
    return dict(
        account={k: _clean(v) for k, v in a.to_dict().items()}, lender=_lender(a["lender_id"]),
        ptps=records(ptps), payments=records(pays), dial_summary=dict(attempts=len(dials),
                                                                      answered=int((dials["network_response"] == "answered").sum()),
                                                                      dispositions={k: int(v) for k, v in disp.items()}),
        dials=records(dials.head(40)),
        unavailable=["customer name", "contact details", "address"],
    )


# =============================================================================== agents
@app.get("/api/agents")
def agents():
    need_data()
    d = STORE.dash.merge(STORE.agents, on=["agent_id"], how="left", suffixes=("", "_meta"))
    return dict(rows=records(d), note="Integrity risk z-score combines pushed-PTP share, copy-pasted remark share, "
                                      "shift-end share and quality ratio (tele agents only). It is a review signal, "
                                      "not a fraud determination.")


@app.get("/api/agents/{agent_id}")
def agent(agent_id: str):
    need_data()
    d = STORE.dash[STORE.dash["agent_id"] == agent_id]
    if d.empty:
        raise HTTPException(404, "Agent not found")
    b = STORE.base[STORE.base["agent_id"] == agent_id].merge(
        STORE.en[["ptp_id", "predicted_category", "confidence_score", "queue"]], on="ptp_id")
    hours = b["captured_ts"].dt.hour.value_counts().sort_index()
    rem = b["remark"].fillna("").value_counts().head(8)
    peers = STORE.dash[STORE.dash["channel"] == d["channel"].iloc[0]]
    cat = b["predicted_category"].value_counts()
    return dict(
        agent={**records(d)[0], **records(STORE.agents[STORE.agents["agent_id"] == agent_id])[0]},
        peer_avg=dict(quality_ratio=float(peers["quality_ratio"].mean()), pushed_share=float(peers["pushed_share"].mean()),
                      shift_end_share=float(peers["shift_end_share"].mean()),
                      generic_remark_share=float(peers["generic_remark_share"].mean()),
                      keep_rate=float((peers["actual_kept"] / peers["raw_ptps"]).mean())),
        stats=dict(ptps=len(b), accounts=int(b["account_id"].nunique()), kept=int(b["kept"].sum()),
                   keep_rate=float(b["kept"].mean()) if len(b) else None,
                   quality_weighted=float(b["confidence_score"].sum()),
                   recorded_without_cust_date=None),
        capture_hours=[dict(hour=int(h), ptps=int(v)) for h, v in hours.items()],
        categories=[dict(name=c, value=int(cat.get(c, 0))) for c in CATEGORIES],
        top_remarks=[dict(remark=k or "(empty)", count=int(v),
                          accounts=int(STORE.base[STORE.base["remark"] == k]["account_id"].nunique()))
                     for k, v in rem.items()],
        ptps=records(b.sort_values("captured_ts", ascending=False)[
            ["ptp_id", "attempt_id", "account_id", "captured_ts", "promised_amount", "status", "kept",
             "predicted_category", "confidence_score", "queue"]].head(60)),
    )


# =============================================================================== treatment
@app.get("/api/treatment/options")
def treatment_options(q: str = "", category: str = "", page: int = 1, size: int = 20):
    return live_calls(q=q, category=category, page=page, size=size)


@app.get("/api/treatment/{ptp_id}")
def treatment(ptp_id: str):
    need_data()
    if ptp_id not in STORE.base_by_ptp.index:
        raise HTTPException(404, "PTP not found")
    if STORE.base_by_ptp.loc[ptp_id, "attempt_id"] not in STORE.calls_by_id:
        s = STORE.en_by_ptp.loc[ptp_id]
        return dict(ptp_id=ptp_id, live=False,
                    reason="No transcript for this PTP (field visit); showing the stored decision only.",
                    stored={k: _clean(v) for k, v in s.to_dict().items()})
    a = assess(STORE, TAB, QWEN, ptp_id)
    a["live"] = True
    a["context"] = {k: _clean(STORE.base_by_ptp.loc[ptp_id, k]) for k in
                    ["portfolio", "bucket_ord", "dpd_start", "ability_to_pay_estimate", "salary_credit_day",
                     "total_prior_ptps", "total_prior_broken", "promise_lag_days", "is_salary_window",
                     "is_pre_salary_hazard", "hardship_30d", "promised_amount", "overdue_at_capture"]}
    a["audit_id"] = AUDIT.write("treatment_decision", a, model_versions())["audit_id"]
    return a


# =============================================================================== analytics
@app.get("/api/analytics")
def analytics(lender: str = "", portfolio: str = "", channel: str = "", agent: str = "", category: str = "",
              date_from: str = "", date_to: str = ""):
    need_data()
    x = STORE.base[["ptp_id", "account_id", "agent_id", "channel", "lender_id", "portfolio", "captured_ts",
                    "promised_amount", "paid_in_grace", "paid_in_cure"]].merge(
        STORE.en[["ptp_id", "kept", "cured", "status", "predicted_category", "p_keep_in_call", "queue"]], on="ptp_id")
    for col, v in [("lender_id", lender), ("portfolio", portfolio), ("channel", channel), ("agent_id", agent),
                   ("predicted_category", category)]:
        if v:
            x = x[x[col] == v]
    if date_from:
        x = x[x["captured_ts"] >= pd.Timestamp(date_from)]
    if date_to:
        x = x[x["captured_ts"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1)]
    d = STORE.dials.merge(STORE.accounts[["account_id", "lender_id", "portfolio"]], on="account_id")
    for col, v in [("lender_id", lender), ("portfolio", portfolio), ("agent_id", agent)]:
        if v:
            d = d[d[col] == v]
    if date_from:
        d = d[d["attempt_ts"] >= pd.Timestamp(date_from)]
    if date_to:
        d = d[d["attempt_ts"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1)]
    contacts = int((d["network_response"] == "answered").sum())
    if x.empty:
        return dict(empty=True)
    x["week"] = x["captured_ts"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    trend = x.groupby("week").agg(ptps=("ptp_id", "size"), keep_rate=("kept", "mean"),
                                  mean_confidence=("p_keep_in_call", "mean"), recovered=("paid_in_cure", "sum")).reset_index()
    catperf = x.groupby("predicted_category").agg(ptps=("ptp_id", "size"), actual_keep_rate=("kept", "mean"),
                                                  mean_confidence=("p_keep_in_call", "mean"),
                                                  recovered=("paid_in_cure", "sum")).reset_index()
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(x["p_keep_in_call"], edges) - 1, 0, 9)
    calib = x.groupby(b).agg(predicted=("p_keep_in_call", "mean"), observed=("kept", "mean"), n=("kept", "size")).reset_index()
    human_q = {"hardship_desk_human", "integrity_review", "senior_caller", "restructuring_review"}
    return dict(
        empty=False,
        kpis=dict(ptps=len(x), keep_rate=float(x["kept"].mean()), cured_rate=float(x["cured"].mean()),
                  broken=int((x["status"] == "broken").sum()), recovered=float(x["paid_in_cure"].sum()),
                  promised=float(x["promised_amount"].sum()), contacts=contacts,
                  recovery_per_contact=float(x["paid_in_cure"].sum() / contacts) if contacts else None,
                  routed_to_human=int(x["queue"].isin(human_q).sum())),
        trend=records(trend), category_performance=records(catperf),
        calibration=records(calib.drop(columns=calib.columns[0])),
        outcomes=[dict(name=k, value=int((x["status"] == k).sum())) for k in ["kept_on_time", "kept_late", "partial", "broken"]],
        queues=[dict(name=k, value=int(v)) for k, v in x["queue"].value_counts().items()],
        filters=dict(lender=sorted(STORE.base["lender_id"].unique()), portfolio=sorted(STORE.base["portfolio"].unique()),
                     channel=sorted(STORE.base["channel"].unique()), agent=sorted(STORE.base["agent_id"].unique()),
                     category=CATEGORIES, date_min=str(STORE.base["captured_ts"].min().date()),
                     date_max=str(STORE.base["captured_ts"].max().date())),
        unavailable=dict(ai_vs_control="No randomised AI-assisted vs control experiment exists in the supplied data. "
                                       "See the RCT design (README §7); projected effects are on the Overview as PROJECTED."),
        measured=dict(streaming_curve=STORE.results["streaming_curve"], business=STORE.results["business"]),
    )


# =============================================================================== audit
@app.get("/api/audit")
def audit(q: str = "", kind: str = "", status: str = "", category: str = ""):
    rows = AUDIT.read()[::-1]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r["ptp_id"] + r["call_id"] + r["account_id"] + r["audit_id"]).lower()]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    if status:
        rows = [r for r in rows if r["audit_status"] == status]
    if category:
        rows = [r for r in rows if r["final_category"] == category]
    return dict(rows=rows[:500], total=len(rows),
                batch_audit=STORE.results.get("audit") if STORE.ready else None)


@app.get("/api/audit/{audit_id}")
def audit_one(audit_id: str):
    for r in AUDIT.read():
        if r["audit_id"] == audit_id:
            return r
    raise HTTPException(404, "Audit record not found")


# =============================================================================== static frontend (built)
DIST = ROOT / "webapp" / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
