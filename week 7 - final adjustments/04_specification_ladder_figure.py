"""
04_specification_ladder_figure.py
SOSC 314 final phase, task g: the specification ladder (the measurement story in one picture).

PURPOSE
    Show how the recession effect changes as the comparison becomes fairer:
        quarterly raw  ->  + linear trend  ->  + decade effects
        ad level + decade effects  ->  + advertiser effects
    for all four outcomes, on the unfiltered corpus and on the cleanest filter version.

QUESTION
    What was the initial pattern, why did it weaken, and does a cleaner corpus change it?

WHY IT MATTERS
    The Week 5 feedback asked us to tell the measurement story clearly. This is
    Figure 1 of the final report and the central slide of the presentation.

FILES LOADED
    week 7 - final adjustments/model runner/csv outputs/model_results_by_filter_version.csv
    (produced by 02_filter_versions_and_model_runner.py; Drive copy used if not in the repo)

OUTPUTS (week 7 - final adjustments/specification ladder/)
    diagnostic images/fig_ladder_all_outcomes.png   four panels, unfiltered vs cleanest filter
    diagnostic images/fig_ladder_lm_positive.png    one panel, for slides
    csv outputs/specification_ladder_table.csv      the plotted numbers (table view)
    robustness tables/specification_ladder_summary.txt

NOTES
    Effects are % of each series' own mean (the project's scale rule), with 95% CIs.
    Quarterly: macro (average of ads) series, majority recession definition, HAC (4 lags).
    Ad level: per-ad shares, advertisers with 2+ ads, errors clustered by year (script 06 rule).

RUN
    Paste into one Colab cell, or run as a file. Runs in seconds; reads one small CSV.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WEEK7 = "week 7 - final adjustments"
REPO_URL = "https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git"
DRIVE_DATA = Path("/content/drive/MyDrive/SOSC314_data")

# Reference palette (dataviz skill, light mode), validated: slots 1-2 pass all checks
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_FILTERED, C_RAW = "#2a78d6", "#eb6834"          # blue = cleanest filter, orange = unfiltered

STEPS = [("quarterly", "raw", "Raw comparison"),
         ("quarterly", "trend", "+ linear trend"),
         ("quarterly", "decade", "+ decade effects"),
         ("ad", "decade", "Ad level, decade effects"),
         ("ad", "decade+advertiser", "+ advertiser effects")]
OUTCOMES = [("LM", "Pos", "Loughran-McDonald positive"), ("LM", "Neg", "Loughran-McDonald negative"),
            ("Harvard", "Pos", "Harvard positive"), ("Harvard", "Neg", "Harvard negative")]

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def find_repo_root():
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    if os.environ.get("SOSC314_REPO"):
        starts.append(Path(os.environ["SOSC314_REPO"]))
    starts += [Path.cwd(), Path("/content/repo"), Path("/content/Jake_Daniel_Kiran_SOSC314")]
    for s in starts:
        for p in [s, *s.parents]:
            if (p / "week three data cleaning").is_dir():
                return p
    if IN_COLAB:
        t = Path("/content/Jake_Daniel_Kiran_SOSC314")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(t)], check=True,
                       env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        return t
    sys.exit("Could not find the repo. Set SOSC314_REPO to the path of your clone.")


ROOT = find_repo_root()
if IN_COLAB and not Path("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"Note: could not mount Drive ({e}).")

cands = [ROOT / WEEK7 / "model runner" / "csv outputs" / "model_results_by_filter_version.csv",
         DRIVE_DATA / "model runner" / "csv outputs" / "model_results_by_filter_version.csv"]
existing = [p for p in cands if p.exists()]
if not existing:
    sys.exit("model_results_by_filter_version.csv not found. Run 02_filter_versions_and_model_runner.py first.")
SRC = max(existing, key=lambda p: p.stat().st_mtime)      # newest copy (repo or Drive)

OUT = ROOT / WEEK7 / "specification ladder"
OUT_IMG, OUT_CSV, OUT_TAB = OUT / "diagnostic images", OUT / "csv outputs", OUT / "robustness tables"
for d in (OUT_IMG, OUT_CSV, OUT_TAB):
    d.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(SRC)
versions = sorted(v for v in res["version"].unique() if v != "none")
CLEAN = versions[-1]                       # latest filter version (e.g. v3)
lines = [f"Source: {SRC}", f"Compared: unfiltered corpus ('none') vs cleanest filter ('{CLEAN}')"]

sel = res[((res.level == "quarterly") & (res.aggregation == "macro") & (res.recession_def == "majority")) |
          (res.level == "ad")]
rows = []
for d, s, oname in OUTCOMES:
    for lvl, spec, sname in STEPS:
        for v in ("none", CLEAN):
            r = sel[(sel.dictionary == d) & (sel.outcome == s) & (sel.level == lvl) &
                    (sel.spec == spec) & (sel.version == v)]
            if len(r) != 1:
                sys.exit(f"Expected one row for {d} {s} {lvl} {spec} {v}, found {len(r)}.")
            r = r.iloc[0]
            rows.append(dict(outcome=oname, step=sname, level=lvl, spec=spec, version=v,
                             pct_of_mean=r.pct_of_mean, ci_low_pct=r.ci_low_pct, ci_high_pct=r.ci_high_pct,
                             p=r.p, n=int(r.n)))
tab = pd.DataFrame(rows)
tab.to_csv(OUT_CSV / "specification_ladder_table.csv", index=False)

# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
plt.rcParams.update({"axes.unicode_minus": True, "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
xmin = np.floor(min(tab.ci_low_pct.min(), -1) / 5) * 5
xmax = np.ceil(max(tab.ci_high_pct.max(), 1) / 5) * 5


def ladder(ax, sub, title, show_y=True, label_points=False):
    y = np.arange(len(STEPS))
    ax.axhspan(2.5, 4.5, color=GRID, alpha=0.35, lw=0, zorder=0)       # ad-level block
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for v, color, off, marker, face in (("none", C_RAW, -0.14, "o", SURFACE),
                                        (CLEAN, C_FILTERED, 0.14, "s", C_FILTERED)):
        g = sub[sub.version == v].set_index("step").reindex([s[2] for s in STEPS])
        ax.hlines(y + off, g.ci_low_pct, g.ci_high_pct, color=color, lw=2, zorder=2)
        ax.scatter(g.pct_of_mean, y + off, s=42, marker=marker, facecolor=face, edgecolor=color,
                   linewidth=1.8, zorder=3)
        if label_points and v == CLEAN:
            for yi, (_, r) in zip(y, g.iterrows()):
                star = ("  p = " + f"{r.p:.3f}".lstrip("0")) if r.p >= 0.001 else "  p < .001"
                val = f"{r.pct_of_mean:+.1f}%".replace("-", "\u2212")
                ax.text(r.ci_high_pct + (xmax - xmin) * 0.015, yi + off, val + star,
                        va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [s[2] for s in STEPS] if show_y else [""] * len(STEPS))
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def legend(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=C_RAW, markerfacecolor=SURFACE, markeredgewidth=1.8, lw=2,
                label="Unfiltered corpus"),
         Line2D([], [], marker="s", color=C_FILTERED, markerfacecolor=C_FILTERED, lw=2,
                label=f"Cleanest ad filter ({CLEAN})")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, y), ncol=2, frameon=False, fontsize=9)


# Figure 1: four outcomes
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
for ax, (d, s, oname), show_y in zip(axes.flat, OUTCOMES, (True, False, True, False)):
    ladder(ax, tab[tab.outcome == oname], oname, show_y=show_y)
for ax in axes[1]:
    ax.set_xlabel("Recession effect, % of series mean (95% CI)")
fig.suptitle("How the recession effect changes as the comparison gets fairer",
             x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold", y=0.995)
legend(fig, 0.958)
fig.text(0.01, 0.005, "Quarterly: 268 quarters, HAC errors (4 lags). Shaded rows: ad-level models, errors clustered "
         "by year. Negative = lower in recession quarters.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.935), h_pad=1.2)
fig.savefig(OUT_IMG / "fig_ladder_all_outcomes.png", dpi=220)
plt.close(fig)

# Figure 2: LM positive only, labelled, for slides
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ladder(ax, tab[tab.outcome == "Loughran-McDonald positive"], "Loughran-McDonald positive words",
       label_points=True)
ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.25)
ax.set_xlabel("Recession effect, % of series mean (95% CI)")
legend(fig, 0.99)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT_IMG / "fig_ladder_lm_positive.png", dpi=220)
plt.close(fig)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
lines.append("\nRecession effect, % of mean (p) - unfiltered | cleanest filter")
for oname in [o[2] for o in OUTCOMES]:
    lines.append(f"\n{oname}")
    for _, _, sname in STEPS:
        a = tab[(tab.outcome == oname) & (tab.step == sname)].set_index("version")
        lines.append(f"  {sname:28} {a.loc['none', 'pct_of_mean']:+6.2f} (p={a.loc['none', 'p']:.3f})  |  "
                     f"{a.loc[CLEAN, 'pct_of_mean']:+6.2f} (p={a.loc[CLEAN, 'p']:.3f})")
shift = (tab.pivot_table(index=["outcome", "step"], columns="version", values="pct_of_mean")
         .eval(f"`{CLEAN}` - none").abs().max())
lines.append(f"\nLargest change between unfiltered and {CLEAN}: {shift:.2f} percentage points of the mean.")
(OUT_TAB / "specification_ladder_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved figures to {OUT_IMG}")

if IN_COLAB and Path("/content/drive/MyDrive").exists():
    import shutil
    shutil.copytree(OUT, DRIVE_DATA / "specification ladder", dirs_exist_ok=True)
    print(f"Copy saved to Drive: {DRIVE_DATA / 'specification ladder'}")
