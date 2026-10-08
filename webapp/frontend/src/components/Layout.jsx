import { useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { QwenStatusLine, useHealth } from "./Health";

const I = {
  overview: <path d="M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z" />,
  live: <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zm7 11a7 7 0 0 1-14 0H3a9 9 0 0 0 8 8.94V23h2v-2.06A9 9 0 0 0 21 12z" />,
  customers: <path d="M16 11a4 4 0 1 0-4-4 4 4 0 0 0 4 4zm-8 0a3 3 0 1 0-3-3 3 3 0 0 0 3 3zm0 2c-2.7 0-8 1.3-8 4v3h8v-3c0-1 .4-2 1.1-2.8A12 12 0 0 0 8 13zm8 0c-2.7 0-8 1.3-8 4v3h16v-3c0-2.7-5.3-4-8-4z" />,
  agents: <path d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5zm0 2c-3.3 0-10 1.7-10 5v3h20v-3c0-3.3-6.7-5-10-5z" />,
  treatment: <path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-8 14H7v-2h4zm6-4H7v-2h10zm0-4H7V7h10z" />,
  analytics: <path d="M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z" />,
  audit: <path d="M12 1 3 5v6c0 5.6 3.8 10.7 9 12 5.2-1.3 9-6.4 9-12V5zm-2 16-4-4 1.4-1.4L10 14.2l6.6-6.6L18 9z" />,
};
const NAV = [
  ["/overview", "Overview", "overview"],
  ["/live-calls", "Live Calls", "live"],
  ["/customers", "Customers", "customers"],
  ["/agents", "Agent Quality", "agents"],
  ["/treatment", "Treatment Engine", "treatment"],
  ["/analytics", "Analytics", "analytics"],
  ["/audit", "Audit & Compliance", "audit"],
];
const NAMES = { overview: "Overview", "live-calls": "Live Calls", customers: "Customers", agents: "Agent Quality", treatment: "Treatment Engine", analytics: "Analytics", audit: "Audit & Compliance" };

export default function Layout({ children }) {
  const [open, setOpen] = useState(false);
  const { health, at } = useHealth();
  const loc = useLocation();
  const parts = loc.pathname.split("/").filter(Boolean);
  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Main navigation">
        <div className="brand">
          <div className="brand-name">CreditNirvana</div>
          <div className="brand-sub">PTP Intelligence</div>
        </div>
        <nav className="nav">
          {NAV.map(([to, label, icon]) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)}>
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">{I[icon]}</svg>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="side-status" aria-live="polite">
          <div className="title">Qwen SLM</div>
          <div className="row" style={{ marginTop: 4 }}><QwenStatusLine /></div>
          <div className="row"><span>Model</span><span className="mono" style={{ textAlign: "right" }}>{health?.qwen?.version ?? "N/A"}</span></div>
          <div className="row"><span>Device</span><span style={{ textAlign: "right" }}>{health?.qwen?.device ? health.qwen.device.replace("NVIDIA GeForce ", "") : "N/A"}</span></div>
          <div className="row"><span>Tabular</span><span>{health?.tabular?.ready ? "Ready" : "N/A"}</span></div>
          <div className="row"><span>Data</span><span>{health?.data?.ready ? `${health.data.ptps.toLocaleString()} PTPs` : "Loading"}</span></div>
          <div className="row"><span>Checked</span><span>{at ? at.toLocaleTimeString() : "N/A"}</span></div>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <div className="row">
            <button className="btn sm menu-btn" onClick={() => setOpen((o) => !o)} aria-label="Toggle navigation">Menu</button>
            <nav className="crumbs" aria-label="Breadcrumb">
              <Link to="/overview">PTP Intelligence</Link>
              {parts.map((p, i) => (
                <span key={i} className="row" style={{ gap: 6 }}>
                  <span>/</span>
                  {i === parts.length - 1 ? <span className="cur">{NAMES[p] || p}</span> : <Link to={"/" + parts.slice(0, i + 1).join("/")}>{NAMES[p] || p}</Link>}
                </span>
              ))}
            </nav>
          </div>
          <div className="small muted row" style={{ gap: 6 }}>
            <QwenStatusLine />
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
