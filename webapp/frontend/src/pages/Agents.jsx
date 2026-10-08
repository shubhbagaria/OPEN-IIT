import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { agentsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, Card, DataTable, PageHead, StatusBadge, fmtPct, useClientSort } from "../components/ui";

export default function Agents() {
  const nav = useNavigate();
  const q = useApi((o) => agentsApi.list(o));
  const [text, setText] = useState("");
  const [channel, setChannel] = useState("");
  const [flagOnly, setFlagOnly] = useState(false);
  const rows = (q.data?.rows || []).filter((r) =>
    (!text || r.agent_id.toLowerCase().includes(text.toLowerCase())) && (!channel || r.channel === channel) && (!flagOnly || r.integrity_flag));
  const s = useClientSort(rows, { key: "integrity_risk_z", dir: "desc" });
  const cols = [
    { key: "agent_id", label: "Agent", sortable: true, render: (r) => <div><div style={{ fontWeight: 600 }}>{r.agent_id}</div><div className="small muted">{r.channel} · {r.language_team ?? ""}{r.tenure_months ? ` · ${r.tenure_months} mo` : ""}</div></div> },
    { key: "raw_ptps", label: "Raw PTPs", sortable: true, num: true },
    { key: "quality_weighted_ptps", label: "Quality-weighted PTPs", sortable: true, num: true, render: (r) => r.quality_weighted_ptps.toFixed(1) },
    { key: "actual_kept", label: "Kept", sortable: true, num: true },
    { key: "quality_ratio", label: "Mean credibility", sortable: true, num: true, render: (r) => fmtPct(r.quality_ratio, 0) },
    { key: "keep_lift_vs_peers", label: "Keep rate vs peers", sortable: true, num: true, render: (r) => `${r.keep_lift_vs_peers > 0 ? "+" : ""}${(r.keep_lift_vs_peers * 100).toFixed(1)} pp` },
    { key: "pushed_share", label: "Agent-pushed share", sortable: true, num: true, render: (r) => fmtPct(r.pushed_share, 0) },
    { key: "generic_remark_share", label: "Copy-paste remarks", sortable: true, num: true, render: (r) => fmtPct(r.generic_remark_share, 0) },
    { key: "shift_end_share", label: "Shift-end PTPs", sortable: true, num: true, render: (r) => fmtPct(r.shift_end_share, 0) },
    { key: "integrity_risk_z", label: "Integrity signal", sortable: true, render: (r) => r.integrity_risk_z === null ? <span className="muted small">N/A (non-tele)</span> :
      r.integrity_flag ? <span className="badge projected">Potential PTP inflation, review</span> : <span className="small muted num">z {r.integrity_risk_z.toFixed(2)}</span> },
  ];
  return (
    <>
      <PageHead title="Agent Quality" desc="Quality-weighted PTPs next to raw counts, plus integrity indicators for the supervisory review team." />
      <Card bodyClass="" title="Agents and bot" sub={q.data?.note} right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="STORED MODEL OUTPUT" /></div>}>
        <div className="toolbar" style={{ padding: "12px 16px 0" }}>
          <input className="input" placeholder="Search agent ID" value={text} onChange={(e) => setText(e.target.value)} aria-label="Search agents" />
          <select className="select" value={channel} onChange={(e) => setChannel(e.target.value)} aria-label="Channel">
            <option value="">All channels</option><option value="tele_agent">Tele agent</option><option value="field">Field</option><option value="voice_bot">Voice bot</option>
          </select>
          <label className="row small" style={{ gap: 6 }}><input type="checkbox" checked={flagOnly} onChange={(e) => setFlagOnly(e.target.checked)} /> Flagged only</label>
        </div>
        <div style={{ marginTop: 12 }}>
          <Async q={q} rows={8}>{() => <DataTable columns={cols} rows={s.rows} rowKey="agent_id" sort={s.sort} onSort={s.onSort} onRowClick={(r) => nav(`/agents/${r.agent_id}`)} />}</Async>
        </div>
      </Card>
    </>
  );
}
