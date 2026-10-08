"""Static evaluation figures for the report (PNG). Run: python -m ptp_copilot.figures"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CATEGORIES, OUT_DIR

S1, S2, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
FIG = OUT_DIR / "figures"


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    FIG.mkdir(exist_ok=True)
    R = json.load(open(OUT_DIR / "results.json"))
    en = pd.read_csv(OUT_DIR / "enriched_ptp_predictions.csv")

    # 1. reliability diagram (cross-fitted in-call keep probability, all PTPs)
    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=SURF)
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(en["p_keep_in_call"], bins) - 1, 0, 9)
    g = en.groupby(b).agg(p=("p_keep_in_call", "mean"), y=("kept", "mean"), n=("kept", "size"))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
    ax.plot(g["p"], g["y"], color=S1, lw=2, marker="o", ms=6, label="in-call model")
    for _, r in g.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.p, r.y), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK2)
    ax.set_xlabel("Predicted keep probability", color=INK2, fontsize=9)
    ax.set_ylabel("Observed keep rate", color=INK2, fontsize=9)
    _style(ax, f"Calibration, out-of-fold, all PTPs (ECE {R['crossfit_all_ptps']['keep_ece_in_call']:.3f})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_in_call.png", dpi=160)

    # 2. quality vs share of call heard
    c = pd.DataFrame(R["streaming_curve"])
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURF)
    x = np.arange(len(c))
    ax.plot(x, c["keep_auc"], color=S1, lw=2, marker="o", ms=6, label="Keep AUC")
    ax.plot(x, c["category_macro_f1"], color=S2, lw=2, marker="o", ms=6, label="Category macro-F1")
    for xi, a, f in zip(x, c["keep_auc"], c["category_macro_f1"]):
        ax.annotate(f"{a:.2f}", (xi, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=INK)
        ax.annotate(f"{f:.2f}", (xi, f), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, c["call_fraction"])
    ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("Share of the call heard so far (test calls)", color=INK2, fontsize=9)
    _style(ax, "Live prediction quality as the call unfolds")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "streaming_quality.png", dpi=160)

    # 3. confusion matrix (cost-sensitive decision, test split)
    cm = np.array(R["category_vs_ground_truth"]["confusion"])
    short = ["Feasible", "Infeasible", "Escape", "Third party", "Repeat", "Agent-pushed"]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURF)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", S1]), vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(6), short, rotation=30, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(6), short, fontsize=8, color=INK2)
    ax.set_xlabel("Predicted (cost-sensitive)", color=INK2, fontsize=9)
    ax.set_ylabel("Ground truth", color=INK2, fontsize=9)
    ax.set_title("Category confusion, test split (row-normalised shading)", loc="left", fontsize=11, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_category.png", dpi=160)

    # 4. SHAP drivers
    imp = pd.Series(R["in_call_top_features"]).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6), facecolor=SURF)
    ax.barh(imp.index, imp.values, color=S1, height=0.6)
    for y, v in enumerate(imp.values):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("mean |SHAP| (log-odds), test split", color=INK2, fontsize=9)
    _style(ax, "In-call keep model: top drivers")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "shap_in_call.png", dpi=160)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
