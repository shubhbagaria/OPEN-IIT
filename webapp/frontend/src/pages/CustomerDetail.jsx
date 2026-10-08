import { Link, useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { customersApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, Card, CategoryPill, DataTable, Kpi, PageHead, QUEUE_LABEL, StatusBadge, STATUS_LABEL, Unavailable, fmtDate, fmtDateTime, fmtInr, fmtPct } from "../components/ui";

export default function CustomerDetail() {
  const { customerId } = useParams();
  const nav = useNavigate();
  const q = useApi((o) => customersApi.get(customerId, o), [customerId]);
  return (
    <Async q={q} rows={10}>
      {(d) => {
        const a = d.account;
        const kept = d.ptps.filter((p) => p.kept === 1).length;
        const paid = d.payments.reduce((s, p) => s + p.amount, 0);
        const confHist = d.ptps.map((p) => ({ date: p.captured_ts?.slice(0, 10), credibility: p.confidence_score, kept: p.kept }));
        const payTl = d.payments.map((p) => ({ t: new Date(p.payment_ts).getTime(), amount: p.amount, channel: p.channel }));
        return (
          <>
            <PageHead title={`Customer ${a.account_id}`}
              desc={`${d.lender.name ?? a.lender_id} (${d.lender.type ?? "N/A"}) · ${a.portfolio} · ${a.income_type} · language ${a.preferred_language}`}
              right={<Link className="btn" to="/customers">Back to customers</Link>} />
            <div className="kpis">
              <Kpi label="DPD at start" value={a.dpd_start ?? "N/A"} status="ACTUAL" note={`Bucket ${a.bucket_start}`} />
              <Kpi label="Outstanding" value={fmtInr(a.outstanding)} status="ACTUAL" note={`Overdue ${fmtInr(a.overdue_start)} · EMI ${fmtInr(a.emi_amount)}`} />
              <Kpi label="PTPs kept" value={`${kept} / ${d.ptps.length}`} status="ACTUAL" note={`Before data window: ${a.prev_ptp_broken} broken of ${a.prev_ptp_count}`} />
              <Kpi label="Payments received" value={fmtInr(paid)} status="ACTUAL" note={`${d.payments.length} payment(s)`} />
              <Kpi label="Bureau / ability to pay" value={a.bureau_score_band} status="ACTUAL" note={`Ability-to-pay estimate ${a.ability_to_pay_estimate ?? "N/A"}`} />
            </div>
            <div className="grid g2">
              <Card title="Account health" right={<StatusBadge s="ACTUAL" />}>
                <dl className="kv">
                  <dt>Salary credit day</dt><dd>{a.salary_credit_day ?? "Not on file"}</dd>
                  <dt>Other active loans</dt><dd>{a.other_active_loans}</dd>
                  <dt>Paid other lenders (30d)</dt><dd>{a.paid_other_lenders_30d ? "Yes" : "No"}</dd>
                  <dt>Last bounce reason</dt><dd>{a.last_bounce_reason}</dd>
                  <dt>Dial attempts</dt><dd>{d.dial_summary.attempts} ({d.dial_summary.answered} answered)</dd>
                  <dt>Dispositions</dt><dd className="small">{Object.entries(d.dial_summary.dispositions).map(([k, v]) => `${k} ${v}`).join(" · ") || "N/A"}</dd>
                  <dt>Not in supplied data</dt><dd className="muted small">{d.unavailable.join(", ")}</dd>
                </dl>
              </Card>
              <Card title="Payment timeline" sub="All-channel payments from payments.csv" right={<StatusBadge s="ACTUAL" />}>
                {payTl.length === 0 ? <Unavailable title="No payments recorded for this account" /> : (
                  <ResponsiveContainer width="100%" height={200}>
                    <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="t" domain={["auto", "auto"]} tickFormatter={(t) => new Date(t).toLocaleDateString("en-IN", { day: "2-digit", month: "short" })} />
                      <YAxis type="number" dataKey="amount" tickFormatter={(v) => fmtInr(v)} width={70} />
                      <ZAxis range={[60, 60]} />
                      <Tooltip formatter={(v, n) => (n === "t" ? fmtDate(v) : fmtInr(v))} labelFormatter={() => ""} />
                      <Scatter data={payTl} fill="#3b82f6" />
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </Card>
            </div>
            <Card className="mt" bodyClass="" title="PTP history, assessments and treatment" sub="Outcome is actual; category, credibility and action are stored out-of-fold model output. Click a PTP for its treatment decision."
              right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="STORED MODEL OUTPUT" /></div>}>
              <DataTable rowKey="ptp_id" rows={d.ptps} onRowClick={(r) => nav(`/treatment?ptp=${r.ptp_id}`)} columns={[
                { key: "ptp_id", label: "PTP", render: (r) => <span style={{ fontWeight: 600 }}>{r.ptp_id}</span> },
                { key: "captured_ts", label: "Captured", render: (r) => fmtDateTime(r.captured_ts) },
                { key: "promised", label: "Promise", num: true, render: (r) => `${fmtInr(r.promised_amount)} by ${r.promised_date?.slice(0, 10)}` },
                { key: "channel", label: "Channel", render: (r) => `${r.channel} · ${r.agent_id}` },
                { key: "status", label: "Outcome", render: (r) => STATUS_LABEL[r.status] },
                { key: "paid_in_cure", label: "Paid (cure)", num: true, render: (r) => fmtInr(r.paid_in_cure) },
                { key: "confidence_score", label: "Credibility", num: true, render: (r) => fmtPct(r.confidence_score, 0) },
                { key: "predicted_category", label: "Behaviour", render: (r) => <CategoryPill c={r.predicted_category} /> },
                { key: "queue", label: "Treatment", render: (r) => QUEUE_LABEL[r.queue] || r.queue },
                { key: "call", label: "Conversation", render: (r) => r.has_transcript ? <Link to={`/live-calls/${r.attempt_id}`} onClick={(e) => e.stopPropagation()}>Replay call</Link> : <span className="muted small">No transcript (field)</span> },
              ]} />
            </Card>
            <div className="grid g2 mt">
              <Card title="Credibility history" sub="Model credibility per PTP over time" right={<StatusBadge s="STORED MODEL OUTPUT" />}>
                {confHist.length < 2 ? <div className="small muted">Only one PTP; no trend to show.</div> : (
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={confHist} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip formatter={(v) => fmtPct(v, 0)} />
                      <Line dataKey="credibility" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>
              <Card bodyClass="" title="Recent dial attempts and agent remarks" right={<StatusBadge s="ACTUAL" />}>
                <div style={{ maxHeight: 220, overflowY: "auto" }}>
                  <DataTable rowKey="attempt_id" rows={d.dials} columns={[
                    { key: "attempt_ts", label: "When", render: (r) => fmtDateTime(r.attempt_ts) },
                    { key: "agent_id", label: "Agent" },
                    { key: "disposition", label: "Disposition" },
                    { key: "remark", label: "Remark", render: (r) => <span className="small">{r.remark}</span> },
                  ]} />
                </div>
              </Card>
            </div>
          </>
        );
      }}
    </Async>
  );
}
