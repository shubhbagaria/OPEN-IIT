import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { treatmentApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CATS, CAT_LABEL, Card, CategoryPill, DataTable, Empty, PageHead, Pager, QUEUE_LABEL, StatusBadge, STATUS_LABEL, fmtInr, fmtPct } from "../components/ui";

function Decision({ ptp }) {
  const q = useApi((o) => treatmentApi.get(ptp, o), [ptp]);
  return (
    <Async q={q} rows={10}>
      {(a) => {
        if (!a.live) return (
          <Card title={`PTP ${a.ptp_id}`} right={<StatusBadge s="STORED MODEL OUTPUT" />}>
            <div className="banner mb">{a.reason}</div>
            <dl className="kv">
              <dt>Category</dt><dd><CategoryPill c={a.stored.predicted_category} conf={a.stored.category_confidence} /></dd>
              <dt>Credibility</dt><dd>{fmtPct(a.stored.confidence_score, 0)}</dd>
              <dt>Action</dt><dd>{QUEUE_LABEL[a.stored.queue]}</dd>
              <dt>Follow-up</dt><dd>{a.stored.follow_up}</dd>
              <dt>Explanation</dt><dd className="small">{a.stored.explanation}</dd>
            </dl>
          </Card>
        );
        const c = a.context;
        const sal = c.is_salary_window ? "in salary window" : c.is_pre_salary_hazard ? "just before salary" : c.salary_credit_day ? `salary day ${c.salary_credit_day}` : "salary day unknown";
        return (
          <>
            <div className="between mb">
              <div className="row">
                <h2 style={{ margin: 0, fontSize: 18 }}>PTP {a.ptp_id}</h2>
                <Link to={`/customers/${a.account_id}`}>{a.account_id}</Link>
                <Link to={`/live-calls/${a.call_id}`}>Replay call {a.call_id}</Link>
              </div>
              <div className="row" style={{ gap: 4 }}><StatusBadge s="MODEL OUTPUT" />{a.audit_id && <Link className="small" to={`/audit/${a.audit_id}`}>audit {a.audit_id}</Link>}</div>
            </div>
            <div className="flow">
              <div className="node"><h5>Conversation credibility</h5><div className="big num">{fmtPct(a.credibility.p_keep, 0)}</div><div className="small muted">LightGBM in-call, calibrated · risk {a.credibility.risk}</div></div>
              <div className="node"><h5>Behaviour category</h5><CategoryPill c={a.tabular_category.category} conf={a.tabular_category.confidence} />
                <div className="small muted" style={{ marginTop: 6 }}>{a.llm.label}: {a.llm.available ? <><CategoryPill c={a.llm.category} conf={a.llm.confidence} /></> : "unavailable"}</div></div>
              <div className="node"><h5>Customer context</h5><div className="small">{c.portfolio} · DPD {c.dpd_start}<br />{fmtInr(c.promised_amount)} of {fmtInr(c.overdue_at_capture)} overdue<br />ability to pay {c.ability_to_pay_estimate ?? "N/A"} · {sal}</div></div>
              <div className="node"><h5>Hardship</h5>{a.hardship_safeguard ? <span className="badge projected">Safeguard active</span> : <span className="small muted">No hardship signal</span>}
                <div className="small muted" style={{ marginTop: 6 }}>{a.evidence.filter((e) => e.polarity === "!").map((e) => e.text).join("; ") || "N/A"}</div></div>
              <div className="node"><h5>Account history</h5><div className="small">{c.total_prior_broken ?? 0} broken of {c.total_prior_ptps ?? 0} prior PTPs<br />promise in {c.promise_lag_days} days{c.hardship_30d ? <><br />hardship disposition in last 30d</> : null}</div></div>
            </div>
            <div className="flow-arrow" aria-hidden="true">↓</div>
            <div className="decision">
              <div className="between">
                <div>
                  <div className="small muted">Treatment decision · category after guardrails</div>
                  <div className="row" style={{ marginTop: 4 }}><CategoryPill c={a.final_category} /><span style={{ fontWeight: 700, fontSize: 18 }}>{QUEUE_LABEL[a.treatment.queue]}</span></div>
                </div>
                <div style={{ textAlign: "right" }} className="small">
                  <div>Parking: <b>{a.treatment.parking_days} day(s)</b></div>
                  <div>Reminders: {a.treatment.reminders.join(", ") || "None"}</div>
                </div>
              </div>
              <div className="mt small"><b>Follow-up:</b> {a.treatment.follow_up}</div>
              {a.guardrails.length > 0 && <div className="small" style={{ marginTop: 6 }}><b>Safeguards applied:</b> {a.guardrails.join("; ")}</div>}
              <div className="script mt">{a.treatment.next_utterance}</div>
              <div className="small muted" style={{ marginTop: 6 }}>Policy check: {a.treatment.policy_check}{a.treatment.compliance_issues.length ? ` (${a.treatment.compliance_issues.join(", ")})` : ""}</div>
            </div>
            <div className="grid g2 mt">
              <Card title="Evidence used">
                {a.evidence.map((e) => <div key={e.signal} className="ev"><span className={`ic ${{ "+": "pos", "-": "neg", "!": "alert", "~": "neu" }[e.polarity]}`}>{e.polarity === "~" ? "•" : e.polarity}</span><div>{e.text}{e.quote && <div className="q">"{e.quote}"</div>}</div></div>)}
              </Card>
              <Card title="Reasoning trace" sub="Structured, deterministic steps, not model chain-of-thought">
                <ol className="small" style={{ paddingLeft: 18, margin: 0, display: "grid", gap: 6 }}>
                  <li>Tabular model scored the full transcript: P(kept) = {fmtPct(a.credibility.p_keep)}; cost-sensitive category = {CAT_LABEL[a.tabular_category.category]}.</li>
                  <li>{a.llm.label} second opinion: {a.llm.available ? `${CAT_LABEL[a.llm.category]} (${fmtPct(a.llm.confidence, 0)})` : "unavailable"}{a.llm.reasons?.length ? `: ${a.llm.reasons.join("; ")}` : ""}.</li>
                  <li>Guardrails: {a.guardrails.length ? a.guardrails.join("; ") : "none triggered"}. Final category = {CAT_LABEL[a.final_category]}.</li>
                  <li>Parking engine: {CAT_LABEL[a.final_category]} × credibility {fmtPct(a.credibility.p_keep, 0)} × promise lag {c.promise_lag_days}d → {QUEUE_LABEL[a.treatment.queue]}, {a.treatment.parking_days} day(s).</li>
                  <li>Approved {a.treatment.utterance_kind} rendered in the borrower's language; compliance filter: {a.treatment.policy_check}.</li>
                </ol>
                <div className="small muted mt">Ground truth (evaluation only): <CategoryPill c={a.ground_truth.category} /> · outcome {STATUS_LABEL[a.ground_truth.outcome]}</div>
              </Card>
            </div>
          </>
        );
      }}
    </Async>
  );
}

export default function Treatment() {
  const [sp, setSp] = useSearchParams();
  const ptp = sp.get("ptp");
  const [text, setText] = useState("");
  const [qText, setQText] = useState("");
  const [cat, setCat] = useState("");
  const [page, setPage] = useState(1);
  useEffect(() => { const t = setTimeout(() => { setQText(text); setPage(1); }, 350); return () => clearTimeout(t); }, [text]);
  const opts = useApi((o) => treatmentApi.options({ q: qText, category: cat, page, size: 12 }, o), [qText, cat, page]);
  return (
    <>
      <PageHead title="Treatment Engine"
        desc="Credibility + behaviour category + customer context + hardship + account history → treatment. Every decision is recomputed live by the backend policy engine and written to the audit log." />
      <div className="grid tgrid">
        <Card bodyClass="" title="Select a PTP" sub="PTPs with a call transcript">
          <div className="toolbar" style={{ padding: "12px 16px 0" }}>
            <input className="input" style={{ minWidth: 0, flex: 1 }} placeholder="Search PTP, account or agent" value={text} onChange={(e) => setText(e.target.value)} aria-label="Search PTPs" />
            <select className="select" value={cat} onChange={(e) => { setCat(e.target.value); setPage(1); }} aria-label="Stored category">
              <option value="">All categories</option>{CATS.map((c) => <option key={c} value={c}>{CAT_LABEL[c]}</option>)}
            </select>
          </div>
          <div style={{ marginTop: 10 }}>
            <Async q={opts} rows={6}>
              {(d) => (
                <>
                  <DataTable rowKey="ptp_id" rows={d.rows} selectedKey={ptp} onRowClick={(r) => setSp({ ptp: r.ptp_id })} columns={[
                    { key: "ptp_id", label: "PTP", render: (r) => <div><b>{r.ptp_id}</b><div className="small muted">{r.account_id}</div></div> },
                    { key: "predicted_category", label: "Stored category", render: (r) => <CategoryPill c={r.predicted_category} /> },
                  ]} />
                  <Pager page={d.page} size={d.size} total={d.total} onPage={setPage} />
                </>
              )}
            </Async>
          </div>
        </Card>
        <div>{ptp ? <Decision ptp={ptp} /> : <Card><Empty title="No PTP selected">Pick a PTP on the left, or open one from a customer or agent page.</Empty></Card>}</div>
      </div>
    </>
  );
}
