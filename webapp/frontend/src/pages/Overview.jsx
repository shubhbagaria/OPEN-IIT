import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { overviewApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CAT_LABEL, CAT_SHORT, Card, Kpi, PageHead, StatusBadge, STATUS_LABEL, fmtInr, fmtInt, fmtPct } from "../components/ui";

const BLUE = "#3b82f6", LIGHT = "#93c5fd", GREEN = "#16a34a", AMBER = "#d97706", MUTED = "#94a3b8";

function kpiValue(k) {
  if (k.fmt === "pct") return fmtPct(k.value);
  if (k.fmt === "inr") return fmtInr(k.value);
  if (k.fmt === "inr_range") return `${fmtInr(k.value)} – ${fmtInr(k.value_high)}`;
  return fmtInt(k.value);
}

export default function Overview() {
  const q = useApi((o) => overviewApi.get(o));
  return (
    <>
      <PageHead
        title="PTP Intelligence Overview"
        desc="How credible are captured promises-to-pay, how did they turn out, and what does the copilot change? Every number carries its provenance."
        right={<Link className="btn primary" to="/live-calls">Open live call console</Link>}
      />
      <Async q={q} rows={8}>
        {(d) => (
          <>
            <div className="kpis">
              {d.kpis.map((k) => <Kpi key={k.key} label={k.label} value={kpiValue(k)} status={k.status} note={k.note} />)}
            </div>
            <div className="grid g2">
              <Card title="Credibility vs actual keep rate, by capture week" sub="Weekly mean of the in-call keep probability against the realised keep rate"
                right={<div className="row" style={{ gap: 4 }}><StatusBadge s="MODEL OUTPUT" /><StatusBadge s="ACTUAL" /></div>}>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={d.trend} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="week" tickFormatter={(w) => w.slice(5)} />
                    <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} domain={[0, 0.6]} />
                    <Tooltip formatter={(v, n) => [fmtPct(v), n]} labelFormatter={(w) => `Week of ${w}`} />
                    <Legend />
                    <Line name="Mean credibility (model)" dataKey="mean_confidence" stroke={BLUE} strokeWidth={2} dot={{ r: 3 }} />
                    <Line name="Actual keep rate" dataKey="actual_keep_rate" stroke={LIGHT} strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
              <Card title="PTP outcomes" sub="Kept = ≥90% paid by promised date + 2 days; cure window +14 days" right={<StatusBadge s="ACTUAL" />}>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={d.outcomes} layout="vertical" margin={{ top: 4, right: 40, left: 30, bottom: 0 }}>
                    <CartesianGrid horizontal={false} />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" tickFormatter={(n) => STATUS_LABEL[n] || n} width={110} />
                    <Tooltip formatter={(v) => [fmtInt(v), "PTPs"]} labelFormatter={(n) => STATUS_LABEL[n] || n} />
                    <Bar dataKey="value" fill={BLUE} radius={[0, 4, 4, 0]} barSize={20} label={{ position: "right", fill: "#5b6b82", fontSize: 11 }} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Behaviour categories" sub="Model prediction (out-of-fold) vs ground-truth label (human + pseudo-label)"
                right={<div className="row" style={{ gap: 4 }}><StatusBadge s="MODEL OUTPUT" /><StatusBadge s="ACTUAL" /></div>}>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={d.categories} margin={{ top: 8, right: 8, left: -10, bottom: 30 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="name" tickFormatter={(c) => CAT_SHORT[c]} interval={0} angle={-20} textAnchor="end" />
                    <YAxis />
                    <Tooltip labelFormatter={(c) => CAT_LABEL[c]} formatter={(v, n) => [fmtInt(v), n]} />
                    <Legend verticalAlign="top" height={28} />
                    <Bar name="Predicted" dataKey="predicted" fill={BLUE} radius={[4, 4, 0, 0]} />
                    <Bar name="Ground truth" dataKey="ground_truth" fill={LIGHT} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Credibility score distribution" sub="How many PTPs fall in each keep-probability band, split by what actually happened"
                right={<div className="row" style={{ gap: 4 }}><StatusBadge s="MODEL OUTPUT" /><StatusBadge s="ACTUAL" /></div>}>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={d.confidence_hist} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="bin" interval={1} />
                    <YAxis />
                    <Tooltip formatter={(v, n) => [fmtInt(v), n]} labelFormatter={(b) => `Credibility ${b}`} />
                    <Legend verticalAlign="top" height={28} />
                    <Bar name="Kept" dataKey="kept" stackId="a" fill={GREEN} />
                    <Bar name="Not kept" dataKey="not_kept" stackId="a" fill={MUTED} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
            <Card className="mt" title="Model quality on held-out data" right={<StatusBadge s="MEASURED" />}>
              <div className="grid g4">
                <Kpi label="Keep AUC, in-call (test)" value={d.model_quality.keep_auc_test.toFixed(3)} note={`Out-of-fold, all PTPs: ${d.model_quality.keep_auc_oof.toFixed(3)}`} />
                <Kpi label="Calibration error (ECE, test)" value={d.model_quality.keep_ece_test.toFixed(3)} note="Lower is better" />
                <Kpi label="Category macro-F1" value={d.model_quality.category_macro_f1_test.toFixed(3)} note={`Qwen SLM: ${d.model_quality.qwen_macro_f1_test.toFixed(3)} fp16 / ${d.model_quality.qwen_nf4_macro_f1_test?.toFixed(3) ?? "N/A"} 4-bit`} />
                <Kpi label="Hardship → aggressive errors" value={fmtPct(d.model_quality.hardship_misclass_costsens)} note="Cost-sensitive decision; then 0 after guardrails" />
              </div>
            </Card>
          </>
        )}
      </Async>
    </>
  );
}
