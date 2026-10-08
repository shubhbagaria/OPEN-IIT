import { Link, useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { agentsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CAT_LABEL, Card, CategoryPill, DataTable, Kpi, PageHead, QUEUE_LABEL, StatusBadge, STATUS_LABEL, fmtDateTime, fmtInr, fmtPct } from "../components/ui";

function Compare({ label, v, peer, higherIsRisk = true }) {
  const max = Math.max(v || 0, peer || 0, 0.01) * 1.25;
  const worse = higherIsRisk ? v > peer * 1.15 : v < peer * 0.85;
  return (
    <div className="bar-cmp">
      <span>{label}</span>
      <div className="track" title={`Agent ${fmtPct(v)} vs peer average ${fmtPct(peer)}`}>
        <div className="fill" style={{ width: `${(v / max) * 100}%`, background: worse ? "#d97706" : "#3b82f6" }} />
        <div className="peer" style={{ left: `${(peer / max) * 100}%` }} />
      </div>
      <span className="num">{fmtPct(v, 0)}</span>
    </div>
  );
}

export default function AgentDetail() {
  const { agentId } = useParams();
  const nav = useNavigate();
  const q = useApi((o) => agentsApi.get(agentId, o), [agentId]);
  return (
    <Async q={q} rows={10}>
      {(d) => {
        const a = d.agent, p = d.peer_avg;
        return (
          <>
            <PageHead title={`Agent ${a.agent_id}`} desc={`${a.channel} · team ${a.language_team ?? "N/A"} · tenure ${a.tenure_months ?? "N/A"} months · ${a.shift ?? ""} shift`}
              right={<Link className="btn" to="/agents">Back to agents</Link>} />
            {a.integrity_flag && <div className="banner warn mb"><strong>POTENTIAL PTP INFLATION DETECTED</strong><span>This is a review signal for the integrity team, not a fraud determination. The bars below show which indicators drive it.</span></div>}
            <div className="kpis">
              <Kpi label="Accounts handled (PTPs)" value={d.stats.accounts} status="ACTUAL" note={`${d.stats.ptps} PTPs captured`} />
              <Kpi label="Keep rate" value={fmtPct(d.stats.keep_rate)} status="ACTUAL" note={`Peer average ${fmtPct(p.keep_rate)}`} />
              <Kpi label="Quality-weighted PTPs" value={d.stats.quality_weighted.toFixed(1)} status="MODEL OUTPUT" note={`Σ credibility vs ${d.stats.ptps} raw`} />
              <Kpi label="Integrity z-score" value={a.integrity_risk_z === null ? "N/A" : a.integrity_risk_z.toFixed(2)} status="MODEL OUTPUT" note="> 0.75 is flagged for review" />
            </div>
            <div className="grid g2">
              <Card title="Why this signal?" sub="Agent value (bar) vs peer average in the same channel (black tick)" right={<StatusBadge s="MODEL OUTPUT" />}>
                <Compare label="Agent-pushed PTP share" v={a.pushed_share} peer={p.pushed_share} />
                <Compare label="Copy-pasted remarks" v={a.generic_remark_share} peer={p.generic_remark_share} />
                <Compare label="End-of-shift PTPs" v={a.shift_end_share} peer={p.shift_end_share} />
                <Compare label="Mean credibility" v={a.quality_ratio} peer={p.quality_ratio} higherIsRisk={false} />
                <div className="small muted mt">Agent-pushed share uses the model's category; remarks and timing come straight from dial_attempts.csv.</div>
              </Card>
              <Card title="PTPs by capture hour" sub="Clustering near the end of the shift can indicate quota-driven logging" right={<StatusBadge s="ACTUAL" />}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={d.capture_hours} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="hour" tickFormatter={(h) => `${h}:00`} />
                    <YAxis />
                    <Tooltip labelFormatter={(h) => `${h}:00–${h}:59`} />
                    <Bar dataKey="ptps" name="PTPs" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Most frequent remarks" sub="Identical text reused across many accounts suggests copy-paste logging" right={<StatusBadge s="ACTUAL" />}>
                <DataTable rowKey="remark" rows={d.top_remarks} columns={[
                  { key: "remark", label: "Remark", render: (r) => <span className="small">{r.remark}</span> },
                  { key: "count", label: "Uses", num: true },
                  { key: "accounts", label: "Accounts (all agents)", num: true },
                ]} />
              </Card>
              <Card title="Behaviour mix of this agent's PTPs" right={<StatusBadge s="STORED MODEL OUTPUT" />}>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={d.categories} layout="vertical" margin={{ top: 4, right: 30, left: 60, bottom: 0 }}>
                    <CartesianGrid horizontal={false} />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" tickFormatter={(c) => CAT_LABEL[c]} width={140} />
                    <Tooltip labelFormatter={(c) => CAT_LABEL[c]} />
                    <Bar dataKey="value" name="PTPs" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
            <Card className="mt" bodyClass="" title="Recent PTPs" sub="Click to open the treatment decision" right={<StatusBadge s="ACTUAL" />}>
              <DataTable rowKey="ptp_id" rows={d.ptps} onRowClick={(r) => nav(`/treatment?ptp=${r.ptp_id}`)} columns={[
                { key: "ptp_id", label: "PTP", render: (r) => <b>{r.ptp_id}</b> },
                { key: "account_id", label: "Account", render: (r) => <Link to={`/customers/${r.account_id}`} onClick={(e) => e.stopPropagation()}>{r.account_id}</Link> },
                { key: "captured_ts", label: "Captured", render: (r) => fmtDateTime(r.captured_ts) },
                { key: "promised_amount", label: "Amount", num: true, render: (r) => fmtInr(r.promised_amount) },
                { key: "status", label: "Outcome", render: (r) => STATUS_LABEL[r.status] },
                { key: "confidence_score", label: "Credibility", num: true, render: (r) => fmtPct(r.confidence_score, 0) },
                { key: "predicted_category", label: "Behaviour", render: (r) => <CategoryPill c={r.predicted_category} /> },
                { key: "queue", label: "Action", render: (r) => QUEUE_LABEL[r.queue] || r.queue },
              ]} />
            </Card>
          </>
        );
      }}
    </Async>
  );
}
