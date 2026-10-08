import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { analyticsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CAT_LABEL, Card, DataTable, Empty, Kpi, PageHead, QUEUE_LABEL, StatusBadge, STATUS_LABEL, Unavailable, fmtInr, fmtInt, fmtPct } from "../components/ui";

const EMPTY = { lender: "", portfolio: "", channel: "", agent: "", category: "", date_from: "", date_to: "" };

export default function Analytics() {
  const [f, setF] = useState(EMPTY);
  const q = useApi((o) => analyticsApi.get(f, o), [JSON.stringify(f)]);
  const opts = q.data?.filters;
  const sel = (k, label, values, fmt = (x) => x) => (
    <select className="select" value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })} aria-label={label}>
      <option value="">{`All ${label.toLowerCase()}`}</option>
      {(values || []).map((v) => <option key={v} value={v}>{fmt(v)}</option>)}
    </select>
  );
  return (
    <>
      <PageHead title="Analytics" desc="Outcome and model analytics over the supplied data. Filters only cover fields that exist in the data." />
      <Card className="mb" bodyClass="card-b">
        <div className="toolbar" style={{ marginBottom: 0 }}>
          {sel("lender", "Lenders", opts?.lender)}
          {sel("portfolio", "Portfolios", opts?.portfolio)}
          {sel("channel", "Channels", opts?.channel)}
          {sel("agent", "Agents", opts?.agent)}
          {sel("category", "Categories", opts?.category, (c) => CAT_LABEL[c])}
          <label className="small muted row" style={{ gap: 4 }}>From <input type="date" className="select" value={f.date_from} min={opts?.date_min} max={opts?.date_max} onChange={(e) => setF({ ...f, date_from: e.target.value })} /></label>
          <label className="small muted row" style={{ gap: 4 }}>To <input type="date" className="select" value={f.date_to} min={opts?.date_min} max={opts?.date_max} onChange={(e) => setF({ ...f, date_to: e.target.value })} /></label>
          <button className="btn" onClick={() => setF(EMPTY)}>Clear filters</button>
        </div>
      </Card>
      <Async q={q} rows={10}>
        {(d) => d.empty ? <Card><Empty title="No PTPs match these filters" /></Card> : (
          <>
            <div className="kpis">
              <Kpi label="PTPs" value={fmtInt(d.kpis.ptps)} status="ACTUAL" />
              <Kpi label="Keep rate" value={fmtPct(d.kpis.keep_rate)} status="ACTUAL" note={`Cured within +14d: ${fmtPct(d.kpis.cured_rate)}`} />
              <Kpi label="Broken PTPs" value={fmtInt(d.kpis.broken)} status="ACTUAL" />
              <Kpi label="Recovered (cure window)" value={fmtInr(d.kpis.recovered)} status="ACTUAL" note={`of ${fmtInr(d.kpis.promised)} promised`} />
              <Kpi label="Recovery per contact" value={fmtInr(d.kpis.recovery_per_contact)} status="ACTUAL" note={`${fmtInt(d.kpis.contacts)} answered dial attempts`} />
              <Kpi label="Routed to human review" value={fmtInt(d.kpis.routed_to_human)} status="MODEL OUTPUT" note="Hardship desk, integrity, senior, restructuring queues" />
            </div>
            <div className="grid g2">
              <Card title="Weekly trend" sub="PTP volume (bars), keep rate and mean credibility (lines)" right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="MODEL OUTPUT" /></div>}>
                <ResponsiveContainer width="100%" height={250}>
                  <ComposedChart data={d.trend} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="week" tickFormatter={(w) => w.slice(5)} />
                    <YAxis yAxisId="r" tickFormatter={(v) => `${Math.round(v * 100)}%`} domain={[0, 0.7]} />
                    <Tooltip formatter={(v, n) => [n === "PTPs" ? fmtInt(v) : fmtPct(v), n]} />
                    <Legend />
                    <Line yAxisId="r" name="Keep rate" dataKey="keep_rate" stroke="#16a34a" strokeWidth={2} dot={{ r: 2 }} />
                    <Line yAxisId="r" name="Mean credibility" dataKey="mean_confidence" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Calibration" sub="Observed keep rate vs predicted credibility (10 bands)" right={<StatusBadge s="MODEL OUTPUT" />}>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={d.calibration} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="predicted" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <Tooltip formatter={(v, n) => [n === "n" ? v : fmtPct(v), n]} labelFormatter={(v) => `Predicted ${fmtPct(v)}`} />
                    <Line name="Observed" dataKey="observed" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                    <Line name="Perfect" data={[{ predicted: 0, observed: 0 }, { predicted: 1, observed: 1 }]} dataKey="observed" stroke="#94a3b8" strokeDasharray="4 4" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
              <Card bodyClass="" title="Category performance" sub="Actual keep rate of the PTPs the model put in each category" right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="MODEL OUTPUT" /></div>}>
                <DataTable rowKey="predicted_category" rows={d.category_performance} columns={[
                  { key: "predicted_category", label: "Predicted category", render: (r) => CAT_LABEL[r.predicted_category] },
                  { key: "ptps", label: "PTPs", num: true },
                  { key: "actual_keep_rate", label: "Actual keep rate", num: true, render: (r) => fmtPct(r.actual_keep_rate) },
                  { key: "mean_confidence", label: "Mean credibility", num: true, render: (r) => fmtPct(r.mean_confidence) },
                  { key: "recovered", label: "Recovered", num: true, render: (r) => fmtInr(r.recovered) },
                ]} />
              </Card>
              <Card title="Outcomes and treatment queues" right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="MODEL OUTPUT" /></div>}>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart data={d.outcomes} layout="vertical" margin={{ top: 0, right: 30, left: 30, bottom: 0 }}>
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="name" tickFormatter={(n) => STATUS_LABEL[n]} width={100} />
                    <Tooltip labelFormatter={(n) => STATUS_LABEL[n]} formatter={(v) => [fmtInt(v), "PTPs"]} />
                    <Bar dataKey="value" fill="#3b82f6" barSize={12} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <ResponsiveContainer width="100%" height={170}>
                  <BarChart data={d.queues} layout="vertical" margin={{ top: 8, right: 30, left: 60, bottom: 0 }}>
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="name" tickFormatter={(n) => QUEUE_LABEL[n] || n} width={150} />
                    <Tooltip labelFormatter={(n) => QUEUE_LABEL[n] || n} formatter={(v) => [fmtInt(v), "PTPs"]} />
                    <Bar dataKey="value" fill="#93c5fd" barSize={12} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Live prediction quality vs share of call heard" sub="Held-out test calls; offline evaluation" right={<StatusBadge s="MEASURED" />}>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={d.measured.streaming_curve} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="call_fraction" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip formatter={(v) => v.toFixed(3)} />
                    <Legend />
                    <Line name="Keep AUC" dataKey="keep_auc" stroke="#3b82f6" strokeWidth={2} />
                    <Line name="Category macro-F1" dataKey="category_macro_f1" stroke="#93c5fd" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
              <Card title="AI-assisted vs control comparison">
                <Unavailable title="No randomised experiment in the supplied data">{d.unavailable.ai_vs_control}</Unavailable>
                <div className="small mt">
                  <div className="between"><span>Broken PTPs flagged before parking</span><span className="row" style={{ gap: 6 }}><b>{fmtPct(d.measured.business.broken_flagged_before_parking)}</b><StatusBadge s="MEASURED" /></span></div>
                  <div className="between mt"><span>Contact-days unlocked per 1,000 PTPs</span><span className="row" style={{ gap: 6 }}><b>{fmtInt(d.measured.business.contact_days_unlocked_per_1000_ptps)}</b><StatusBadge s="MEASURED" /></span></div>
                  <div className="between mt"><span>PTPs per arm to detect +3pp keep-rate lift</span><span className="row" style={{ gap: 6 }}><b>{fmtInt(d.measured.business.rct_ptps_per_arm_for_3pp)}</b><StatusBadge s="TARGET" /></span></div>
                </div>
              </Card>
            </div>
          </>
        )}
      </Async>
    </>
  );
}
