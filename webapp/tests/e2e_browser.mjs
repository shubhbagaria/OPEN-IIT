import puppeteer from "puppeteer-core";

// APP_URL = where the website is served; API_URL = backend (same as APP_URL when the backend serves the build)
const BASE = process.env.APP_URL || "http://127.0.0.1:8000";
const API = process.env.API_URL || BASE;
const SIM_TIMEOUT_MS = Number(process.env.SIM_TIMEOUT_MS || 90000);
const OUT = process.argv[2] || ".";
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  headless: "new",
  args: ["--no-first-run", "--disable-extensions"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("PAGEERROR " + e.message));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const text = () => page.evaluate(() => document.body.innerText);
const report = [];
const check = (name, ok, info = "") => { report.push(`${ok ? "PASS" : "FAIL"}  ${name}${info ? "  :: " + info : ""}`); };

async function visit(path, mustContain, shot) {
  await page.goto(BASE + path, { waitUntil: "networkidle0", timeout: 60000 });
  await sleep(800);
  const t = await text();
  const ok = mustContain.every((s) => t.toLowerCase().includes(s.toLowerCase()));
  check(`page ${path}`, ok && !t.includes("Unable to load data"), ok ? "" : `missing: ${mustContain.filter((s) => !t.toLowerCase().includes(s.toLowerCase())).join(", ")}`);
  if (shot) await page.screenshot({ path: `${OUT}/${shot}.png`, fullPage: true });
  return t;
}

// 1 health in sidebar
let t = await visit("/overview", ["PTP Intelligence Overview", "PTPs captured", "ACTUAL", "PROJECTED", "MEASURED", "Ready"], "01_overview");
check("sidebar shows real LLM version", /gemini-[\w.-]+-api|ptp-qwen0\.5b-lora-r16/.test(t));
// 2 live calls list + filter
t = await visit("/live-calls?category=third_party", ["Live Calls", "Demo scenarios", "Third-Party Promise"], "02_live_calls");
// 3 live call detail + simulation (third-party scenario)
const sc = await (await fetch(API + "/api/live-calls/scenarios")).json();
const tp = sc.find((s) => s.category === "third_party");
const hard = sc.find((s) => s.category === "genuine_infeasible");
for (const s of [tp, hard]) {
  await visit(`/live-calls/${s.call_id}`, ["Call replay", "Call not started"]);
  await page.evaluate(() => [...document.querySelectorAll(".seg button")].find((b) => b.textContent === "4x").click());
  await page.evaluate(() => [...document.querySelectorAll("button")].find((b) => b.textContent === "Start").click());
  const t0 = Date.now();
  let midTurns = null, midScore = null;
  while (Date.now() - t0 < SIM_TIMEOUT_MS) {
    await sleep(700);
    const st = await page.evaluate(() => ({
      turns: document.querySelectorAll(".msg").length,
      pos: document.querySelector(".controller .num")?.textContent,
      cred: document.querySelector("svg[role=img] text")?.textContent,
    }));
    if (st.turns >= 3 && midTurns === null) { midTurns = st.turns; midScore = st.cred; await page.screenshot({ path: `${OUT}/03_live_mid_${s.category}.png`, fullPage: false }); }
    const [a, b] = (st.pos || "Turn 0/1").replace("Turn ", "").split("/").map(Number);
    if (a === b && (await text()).includes("Ground-truth category")) break;   // final card only
  }
  t = await text();
  check(`simulation progressive (${s.category})`, midTurns !== null && midTurns < s.n_turns, `mid-run turns shown=${midTurns}, score=${midScore}`);
  check(`simulation completed (${s.category})`, t.includes("Ground-truth category") && t.includes("Copilot now") && t.toLowerCase().includes("say next"));
  check(`LLM output visible (${s.category})`, /(Gemini|Qwen SLM)[\s\S]{0,120}(agrees|disagrees)/.test(t));
  if (s.category === "third_party") check("privacy safeguard banner shown", t.includes("PRIVACY SAFEGUARD ACTIVE"));
  if (s.category === "genuine_infeasible") check("hardship safeguard banner shown", t.includes("HARDSHIP SAFEGUARD ACTIVE"));
  await page.screenshot({ path: `${OUT}/04_live_done_${s.category}.png`, fullPage: true });
}
// raw qwen draft
await page.evaluate(() => [...document.querySelectorAll("button")].find((b) => /Generate raw .* draft/.test(b.textContent)).click());
await sleep(45000);   // API models can queue behind rate limits
t = await text();
check("raw LLM draft generated", /Raw .* output \(not shown to agent\)/.test(t));
// 4 customer page via link
const acc = (await (await fetch(API + `/api/live-calls/${hard.call_id}`)).json()).account.account_id;
await visit(`/customers/${acc}`, ["PTP history", "Payment timeline", "Account health"], "05_customer");
await visit("/customers", ["Customer Intelligence", "Accounts"]);
// 5 treatment
await visit(`/treatment?ptp=${hard.ptp_id}`, ["Treatment decision", "Conversation credibility", "Reasoning trace"], "06_treatment");
// 6 agents
await visit("/agents", ["Agent Quality", "Potential PTP inflation"], "07_agents");
await visit("/agents/TA013", ["Why this signal?", "POTENTIAL PTP INFLATION DETECTED"], "08_agent_detail");
// 7 analytics + filter
await visit("/analytics", ["Analytics", "Weekly trend", "No randomised experiment"], "09_analytics");
// 8 audit + detail
t = await visit("/audit", ["Decision audit trail", "Batch compliance audit"], "10_audit");
const aud = await (await fetch(API + "/api/audit")).json();
check("audit records created by app", aud.total >= 3, `total=${aud.total}`);
await visit(`/audit/${aud.rows[0].audit_id}`, ["Input / context", "Model output", "Treatment, safeguards and policy"], "11_audit_detail");
// 9 404 + mobile
await visit("/does-not-exist", ["Page not found"]);
await page.setViewport({ width: 390, height: 844, isMobile: true });
await visit(`/live-calls/${tp.call_id}`, ["Call replay"], "12_mobile_live");
const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
check("no horizontal page overflow at 390px", !overflow);
check("no console errors", errors.length === 0, errors.slice(0, 5).join(" | "));
console.log(report.join("\n"));
await browser.close();
