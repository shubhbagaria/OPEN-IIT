"""Central configuration: paths, business constants, label definitions.

Every number that encodes a business decision lives here so that it can be
reviewed and signed off by Collections Ops / Risk / Compliance in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "csv_folder"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- "Kept" definition
GRACE_DAYS = 2            # NEFT/NACH clearing latency + weekend processing
CURE_DAYS = 14            # extended cure window, caps observation inside the 30-day billing cycle
KEPT_THRESHOLD = 0.90     # >=90% of promised amount absorbs bounce charges (Rs 250-500) and rounding
PARTIAL_THRESHOLD = 0.25  # >=25% but <90% -> "partial" (operationally a different follow-up)
DATA_END = "2026-07-24 23:59:59"   # last observed payment; windows ending after this are censored

# ---------------------------------------------------------------- categories
CATEGORIES = [
    "genuine_feasible",
    "genuine_infeasible",
    "escape",
    "third_party",
    "repeat_promiser",
    "agent_recorded_or_pushed",
]
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}

# Misclassification cost matrix (rows = true, cols = predicted). Used for
# cost-sensitive decoding. Hardship predicted as escape/repeat is the costliest
# error (RBI FPC: aggressive treatment of a vulnerable borrower).
COST_MATRIX = [
    #  gf   gi   esc  tp   rep  agt
    [0.0, 1.0, 2.0, 1.0, 2.0, 1.0],   # genuine_feasible
    [2.0, 0.0, 8.0, 1.0, 8.0, 2.0],   # genuine_infeasible  (hardship -> aggressive = 8x)
    [3.0, 1.0, 0.0, 1.0, 0.5, 1.0],   # escape
    [3.0, 2.0, 5.0, 0.0, 5.0, 1.0],   # third_party (any debt talk to 3P = DPDP risk)
    [3.0, 1.0, 0.5, 1.0, 0.0, 1.0],   # repeat_promiser
    [3.0, 1.0, 1.0, 1.0, 1.0, 0.0],   # agent_recorded_or_pushed
]

ANNOTATOR_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}

# ---------------------------------------------------------------- channel economics (Rs)
COST = {
    "sms_whatsapp": 0.25,
    "voice_bot": 2.0,
    "tele_caller": 22.5,        # mid-point of Rs 15-30 per connect
    "senior_caller": 30.0,
    "field_visit": 275.0,       # mid-point of Rs 150-400 per productive visit
}

# ---------------------------------------------------------------- parking engine (days)
BASE_PARKING_RULE = "until_promised_date"   # incumbent: park until PTP date, then +1 day check
MAX_PARKING_DAYS = 30

# ---------------------------------------------------------------- RBI FPC contact window
CONTACT_HOUR_START = 8
CONTACT_HOUR_END = 19

# ---------------------------------------------------------------- latency SLA
TTFT_SLA_MS = 100
VRAM_SLA_GB = 1.0
HUMAN_PAUSE_MS = (600, 1000)
SLM_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
