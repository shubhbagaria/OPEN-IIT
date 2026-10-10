import { createContext, useContext, useEffect, useState } from "react";
import { healthApi } from "../services/api";

const Ctx = createContext({ health: null, error: null });

// Polls the real backend health endpoint. Nothing is assumed "online" without it.
export function HealthProvider({ children }) {
  const [state, setState] = useState({ health: null, error: null, at: null });
  useEffect(() => {
    let alive = true;
    let timer;
    const poll = async () => {
      try {
        const h = await healthApi.get();
        if (alive) setState({ health: h, error: null, at: new Date() });
        timer = setTimeout(poll, h.llm.status === "loading" || !h.data.ready ? 3000 : 15000);
      } catch (e) {
        if (alive) setState({ health: null, error: e, at: new Date() });
        timer = setTimeout(poll, 5000);
      }
    };
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export const useHealth = () => useContext(Ctx);

export function llmReady(h) {
  return h?.llm?.status === "ready";
}

export const llmLabel = (h) => h?.llm?.label || "Language model";

export function LlmStatusLine() {
  const { health, error } = useHealth();
  if (error) return <span><span className="dot bad" /> Backend unreachable</span>;
  if (!health) return <span><span className="dot warn" /> Checking…</span>;
  const s = health.llm.status;
  const cls = s === "ready" ? "ok" : s === "loading" || s === "not_loaded" ? "warn" : "bad";
  const txt = { ready: "Ready", loading: "Loading…", not_loaded: "Not loaded", unavailable: `${health.llm.label} unavailable` }[s] || s;
  return <span className="row" style={{ gap: 6 }}><span className={`dot ${cls}`} /> {txt}</span>;
}
