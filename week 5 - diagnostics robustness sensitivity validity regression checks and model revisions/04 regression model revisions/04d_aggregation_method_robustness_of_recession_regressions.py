# =========================================================
# WEEK 5 MODEL REVISION 04D:
# DOES THE AGGREGATION METHOD CHANGE THE REGRESSION RESULTS?
# (AVERAGE OF RATIOS vs POOLED WORDS)
# =========================================================
#
# PURPOSE:
# Re-estimate the Week 5 recession regressions on both ways of
# turning ad-level sentiment into a quarterly series, and compare
# the results side by side.
#
# MAIN DIAGNOSTIC QUESTION:
# Do the regression conclusions depend on how ads are combined
# into quarters?
#
#   Average of ratios ("macro"):
#       compute each ad's positive/total ratio, then average the
#       ads in the quarter. Every ad counts equally, so a 10-word
#       ad weighs as much as an 8,000-word one. Asks: how positive
#       is the TYPICAL advertisement?
#
#   Pooled words:
#       add up all positive words and all words in the quarter,
#       then divide once. Long ads carry more weight. Asks: how
#       positive is advertising LANGUAGE overall?
#
# WHY THIS DIAGNOSTIC MATTERS:
# In Week 4 the aggregation method changed the raw recession gap
# by 45% (Harvard, stronger under pooling) and 56% (LM, weaker
# under pooling), in opposite directions. Long records are
# disproportionately financial notices, supplements and
# advertorials, so pooling lets them dominate. All regression
# scripts so far (03a, 03b, 04, 04b) use only the average-of-
# ratios series. This script checks whether the Week 5
# conclusions (weak HAC significance, recession gap shrinking
# once long-run drift is controlled) hold under pooling too.
#
# MODELS (same definitions as script 04c):
#   A. Sentiment = Recession
#   B. Sentiment = Recession + Time Trend
#   C. Sentiment = Recession + Decade Fixed Effects
#   D. Sentiment = Recession + Time Trend + Post1973
#                  + Recession x Post1973
#      (reported as the recession effect pre-1973 and post-1973)
#
# Each model: 2 dictionaries x 2 outcomes x 2 aggregations
# x 2 recession definitions (majority 42 / any-month 48).
#
# COMPARING THE TWO AGGREGATIONS:
# The pooled series has a slightly different mean level from the
# macro series, so raw coefficients are also expressed as % of
# each series' own mean (Week 4 scale normalisation). The
# comparison reports, for each model:
#   - coefficient and HAC p-value under each method
#   - whether the sign agrees
#   - whether significance at 5% agrees
#   - % change in the coefficient when switching to pooled
#
# STANDARD ERRORS:
# HAC / Newey-West, 4 lags (same call as 04, 04b, 04c).
#
# SAMPLE:
# Full corpus, 1948Q1-2014Q4, 268 quarters in every model.
#
# FILES LOADED (small plain-text CSVs already in the repo):
#   week 4 - updated quality filtered analysis/week4_aggregations/
#       quarterly_harvard_macro.csv
#       quarterly_harvard_pooled.csv
#       quarterly_loughran_macro.csv
#       quarterly_loughran_pooled.csv
#       aggregation_sensitivity_results.csv   (sanity check only)
#   week three data cleaning/
#       nber_recession_table.csv
#
# OUTPUTS (in this script's folder):
#   csv outputs/
#       aggregation_method_regression_comparison.csv
#           one row per series x model x effect: macro vs pooled
#           side by side
#   diagnostic images/
#       recession_effect_average_of_ratios_vs_pooled_words.png
#   robustness tables/
#       aggregation_method_regression_summary.txt
#
# INTERPRETATION RULE:
# Descriptive associations only, not causal effects.
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


# ---------------------------------------------------------
# DEFINE FILE LOCATIONS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WEEK5_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEEK5_DIR.parent

WEEK4_AGG_DIR = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
    / "week4_aggregations"
)

RECESSION_FILE = REPO_ROOT / "week three data cleaning" / "nber_recession_table.csv"
WEEK4_RESULTS_FILE = WEEK4_AGG_DIR / "aggregation_sensitivity_results.csv"

SENTIMENT_FILES = {
    ("Harvard", "Average of ratios"): WEEK4_AGG_DIR / "quarterly_harvard_macro.csv",
    ("Harvard", "Pooled words"): WEEK4_AGG_DIR / "quarterly_harvard_pooled.csv",
    ("Loughran-McDonald", "Average of ratios"): WEEK4_AGG_DIR / "quarterly_loughran_macro.csv",
    ("Loughran-McDonald", "Pooled words"): WEEK4_AGG_DIR / "quarterly_loughran_pooled.csv",
}


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = SCRIPT_DIR / "csv outputs"
IMAGE_OUTPUT_DIR = SCRIPT_DIR / "diagnostic images"
TABLE_OUTPUT_DIR = SCRIPT_DIR / "robustness tables"

for folder in [CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

HAC_LAGS = 4
BRETTON_WOODS_BREAK_YEAR = 1973

AGGREGATIONS = ["Average of ratios", "Pooled words"]

RECESSION_DEFINITIONS = {
    "Majority of months (42 quarters)": "Recession_Majority",
    "Any month (48 quarters)": "Recession_Any",
}
PRIMARY_DEFINITION = "Majority of months (42 quarters)"

# (model code, effect name, figure / table label)
EFFECTS = [
    ("A", "Recession (whole period)", "A. Recession only"),
    ("B", "Recession (whole period)", "B. + linear trend"),
    ("C", "Recession (whole period)", "C. + decade fixed effects"),
    ("D", "Recession, pre-1973", "D. Trend + interaction: pre-1973"),
    ("D", "Recession, post-1973", "D. Trend + interaction: post-1973"),
]


# ---------------------------------------------------------
# CHECK THAT INPUT FILES EXIST AND ARE NOT LFS STUBS
# ---------------------------------------------------------

def check_input_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file:\n  {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.startswith("version https://git-lfs"):
        raise RuntimeError(
            f"This file is a Git LFS pointer, not the real data:\n  {path}\n"
            f"Run: git lfs pull --include=\"{path.relative_to(REPO_ROOT)}\""
        )


for path in list(SENTIMENT_FILES.values()) + [RECESSION_FILE, WEEK4_RESULTS_FILE]:
    check_input_file(path)


# ---------------------------------------------------------
# LOAD RECESSION DATA
# ---------------------------------------------------------

recession_data = pd.read_csv(RECESSION_FILE)[
    ["Year_Quarter", "Recession_Majority", "Recession_Any"]
].copy()

for column in ["Recession_Majority", "Recession_Any"]:
    recession_data[column] = (
        recession_data[column].astype(str).str.strip().str.lower() == "true"
    ).astype(int)


# ---------------------------------------------------------
# LOAD SENTIMENT SERIES
# ---------------------------------------------------------
# Converted to percentage points (x100), as in 03a-04c.
# ---------------------------------------------------------

def load_series(path):
    data = pd.read_csv(path)[["Year_Quarter", "Pos", "Neg", "Ads"]].copy()
    data = data.merge(recession_data, on="Year_Quarter", how="inner")
    data = data.sort_values("Year_Quarter").reset_index(drop=True)

    data["Pos"] = 100 * data["Pos"]
    data["Neg"] = 100 * data["Neg"]

    data["Year"] = data["Year_Quarter"].str[:4].astype(int)
    data["Time_Trend"] = np.arange(len(data))
    data["Post1973"] = (data["Year"] >= BRETTON_WOODS_BREAK_YEAR).astype(int)
    data["Decade"] = (data["Year"] // 10 * 10).astype(str) + "s"

    decade_dummies = pd.get_dummies(data["Decade"], prefix="Decade", drop_first=True).astype(int)
    data = pd.concat([data, decade_dummies], axis=1)

    return data, list(decade_dummies.columns)


series = {}
for key, path in SENTIMENT_FILES.items():
    data, decade_columns = load_series(path)
    if len(data) != 268:
        raise ValueError(f"{key}: expected 268 quarters, found {len(data)}")
    series[key] = (data, decade_columns)

# Both aggregations must cover exactly the same quarters.
for dictionary in ["Harvard", "Loughran-McDonald"]:
    q_macro = series[(dictionary, "Average of ratios")][0]["Year_Quarter"].tolist()
    q_pooled = series[(dictionary, "Pooled words")][0]["Year_Quarter"].tolist()
    if q_macro != q_pooled:
        raise ValueError(f"{dictionary}: macro and pooled files cover different quarters")


# ---------------------------------------------------------
# DESCRIPTIVE COMPARISON OF THE TWO SERIES
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("HOW SIMILAR ARE THE TWO QUARTERLY SERIES?")
print("=" * 90)

series_rows = []
for dictionary in ["Harvard", "Loughran-McDonald"]:
    macro = series[(dictionary, "Average of ratios")][0]
    pooled = series[(dictionary, "Pooled words")][0]
    for outcome in ["Pos", "Neg"]:
        corr = macro[outcome].corr(pooled[outcome])
        series_rows.append({
            "Dictionary": dictionary,
            "Outcome": outcome,
            "Mean_macro_pp": macro[outcome].mean(),
            "Mean_pooled_pp": pooled[outcome].mean(),
            "Correlation_macro_vs_pooled": corr,
        })
        print(f"{dictionary:<18} {outcome}:  mean macro {macro[outcome].mean():.4f} pp | "
              f"mean pooled {pooled[outcome].mean():.4f} pp | correlation {corr:.3f}")

series_comparison = pd.DataFrame(series_rows)


# ---------------------------------------------------------
# REGRESSION HELPERS
# ---------------------------------------------------------

def fit_hac(data, outcome, predictors):
    X = sm.add_constant(data[predictors].astype(float))
    y = data[outcome].astype(float)
    ols_model = sm.OLS(y, X).fit()
    hac_model = ols_model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
    return hac_model, list(X.columns)


def coefficient(hac_model, names, name):
    i = names.index(name)
    ci = hac_model.conf_int()[i]
    return hac_model.params[i], hac_model.bse[i], hac_model.pvalues[i], ci[0], ci[1]


def linear_combination(hac_model, names, weights):
    r = np.zeros(len(names))
    for name, weight in weights.items():
        r[names.index(name)] = weight
    test = hac_model.t_test(r)
    lo, hi = np.squeeze(test.conf_int())
    return (float(np.squeeze(test.effect)), float(np.squeeze(test.sd)),
            float(np.squeeze(test.pvalue)), float(lo), float(hi))


# ---------------------------------------------------------
# RUN ALL MODELS
# ---------------------------------------------------------

rows = []

for (dictionary, aggregation), (base_data, decade_columns) in series.items():
    for definition_label, recession_column in RECESSION_DEFINITIONS.items():

        data = base_data.copy()
        data["Recession"] = data[recession_column]
        data["Recession_x_Post1973"] = data["Recession"] * data["Post1973"]

        model_specs = {
            "A": ["Recession"],
            "B": ["Recession", "Time_Trend"],
            "C": ["Recession"] + decade_columns,
            "D": ["Recession", "Time_Trend", "Post1973", "Recession_x_Post1973"],
        }

        for outcome in ["Pos", "Neg"]:
            outcome_mean = data[outcome].mean()

            for model_code, predictors in model_specs.items():
                hac_model, names = fit_hac(data, outcome, predictors)

                common = {
                    "Dictionary": dictionary,
                    "Outcome": outcome,
                    "Aggregation": aggregation,
                    "Recession_Definition": definition_label,
                    "Model_Code": model_code,
                    "R_squared": hac_model.rsquared,
                    "Durbin_Watson": durbin_watson(hac_model.resid),
                    "Outcome_mean_pp": outcome_mean,
                }

                if model_code in ["A", "B", "C"]:
                    est, se, p, lo, hi = coefficient(hac_model, names, "Recession")
                    rows.append({**common, "Effect": "Recession (whole period)",
                                 "Coefficient": est, "HAC_SE": se, "HAC_p_value": p,
                                 "CI_low": lo, "CI_high": hi})
                else:
                    est, se, p, lo, hi = coefficient(hac_model, names, "Recession")
                    rows.append({**common, "Effect": "Recession, pre-1973",
                                 "Coefficient": est, "HAC_SE": se, "HAC_p_value": p,
                                 "CI_low": lo, "CI_high": hi})
                    est, se, p, lo, hi = linear_combination(
                        hac_model, names, {"Recession": 1, "Recession_x_Post1973": 1})
                    rows.append({**common, "Effect": "Recession, post-1973",
                                 "Coefficient": est, "HAC_SE": se, "HAC_p_value": p,
                                 "CI_low": lo, "CI_high": hi})

long_results = pd.DataFrame(rows)
long_results["Effect_pct_of_mean"] = (
    100 * long_results["Coefficient"] / long_results["Outcome_mean_pp"]
)


# ---------------------------------------------------------
# SANITY CHECK AGAINST WEEK 4
# ---------------------------------------------------------
# With a single binary regressor, the Model A coefficient equals
# the recession-minus-expansion difference from Week 4's
# aggregation_sensitivity_results.csv (full corpus rows).
# ---------------------------------------------------------

week4 = pd.read_csv(WEEK4_RESULTS_FILE)
week4 = week4[week4["Corpus"] == "Full corpus"]

print("\n" + "=" * 90)
print("SANITY CHECK: MODEL A REPRODUCES WEEK 4 RECESSION-MINUS-EXPANSION GAPS")
print("=" * 90)

all_ok = True
for _, w in week4.iterrows():
    for outcome, column in [("Pos", "Pos_Diff_pp"), ("Neg", "Neg_Diff_pp")]:
        row = long_results[
            (long_results["Dictionary"] == w["Dictionary"])
            & (long_results["Aggregation"] == w["Aggregation"])
            & (long_results["Outcome"] == outcome)
            & (long_results["Recession_Definition"] == PRIMARY_DEFINITION)
            & (long_results["Model_Code"] == "A")
        ].iloc[0]
        ok = abs(row["Coefficient"] - w[column]) < 0.0006
        all_ok = all_ok and ok
        print(f"{w['Dictionary']:<18} {w['Aggregation']:<18} {outcome}: "
              f"{row['Coefficient']:+.4f} vs Week 4 {w[column]:+.4f}  "
              f"{'OK' if ok else 'MISMATCH'}")

if not all_ok:
    print("\nWARNING: at least one Model A coefficient does not match Week 4.")


# ---------------------------------------------------------
# PUT MACRO AND POOLED SIDE BY SIDE
# ---------------------------------------------------------

key_columns = ["Dictionary", "Outcome", "Recession_Definition", "Model_Code", "Effect"]
value_columns = ["Coefficient", "HAC_SE", "HAC_p_value", "CI_low", "CI_high",
                 "Effect_pct_of_mean", "R_squared", "Durbin_Watson", "Outcome_mean_pp"]

macro = long_results[long_results["Aggregation"] == "Average of ratios"][key_columns + value_columns]
pooled = long_results[long_results["Aggregation"] == "Pooled words"][key_columns + value_columns]

comparison = macro.merge(pooled, on=key_columns, suffixes=("_macro", "_pooled"))

comparison["Same_sign"] = (
    np.sign(comparison["Coefficient_macro"]) == np.sign(comparison["Coefficient_pooled"])
)
comparison["Significant_5pct_macro"] = comparison["HAC_p_value_macro"] < 0.05
comparison["Significant_5pct_pooled"] = comparison["HAC_p_value_pooled"] < 0.05
comparison["Same_significance_verdict"] = (
    comparison["Significant_5pct_macro"] == comparison["Significant_5pct_pooled"]
)
comparison["Pct_change_macro_to_pooled"] = (
    100 * (comparison["Coefficient_pooled"] - comparison["Coefficient_macro"])
    / comparison["Coefficient_macro"].abs()
)

label_map = {(code, effect): label for code, effect, label in EFFECTS}
comparison["Model"] = [label_map[(c, e)] for c, e in zip(comparison["Model_Code"], comparison["Effect"])]

order = {label: i for i, (_, _, label) in enumerate(EFFECTS)}
comparison = comparison.sort_values(
    ["Recession_Definition", "Dictionary", "Outcome", "Model"],
    key=lambda s: s.map(order) if s.name == "Model" else s,
).reset_index(drop=True)

front = key_columns[:3] + ["Model"]
comparison = comparison[front + [c for c in comparison.columns if c not in front + ["Model_Code", "Effect"]]]

comparison_path = CSV_OUTPUT_DIR / "aggregation_method_regression_comparison.csv"
comparison.round(6).to_csv(comparison_path, index=False)


# ---------------------------------------------------------
# PRINT PRIMARY COMPARISON
# ---------------------------------------------------------

primary = comparison[comparison["Recession_Definition"] == PRIMARY_DEFINITION].copy()
primary["Series"] = primary["Dictionary"].str.replace("Loughran-McDonald", "LM") + " " + primary["Outcome"]

print("\n" + "=" * 90)
print(f"MACRO vs POOLED - {PRIMARY_DEFINITION} (coefficients in pp, HAC 4 lags)")
print("=" * 90)
with pd.option_context("display.width", 220, "display.max_rows", 100):
    print(primary[[
        "Series", "Model", "Coefficient_macro", "HAC_p_value_macro",
        "Coefficient_pooled", "HAC_p_value_pooled", "Same_sign", "Same_significance_verdict",
    ]].round(4).to_string(index=False))

n_total = len(comparison)
n_same_sign = int(comparison["Same_sign"].sum())
n_same_sig = int(comparison["Same_significance_verdict"].sum())

print(f"\nAcross all {n_total} comparisons (both recession definitions):")
print(f"  same sign:                {n_same_sign}/{n_total}")
print(f"  same 5% significance call: {n_same_sig}/{n_total}")


# ---------------------------------------------------------
# DIAGNOSTIC FIGURE
# ---------------------------------------------------------
# Recession effect (% of series mean, HAC 95% CI) under each
# aggregation method, primary recession definition.
# ---------------------------------------------------------

AGG_STYLE = {
    "Average of ratios": {"color": "#2a78d6", "marker": "o", "offset": -0.14},
    "Pooled words": {"color": "#1baf7a", "marker": "D", "offset": 0.14},
}

PANELS = [
    ("Harvard", "Pos", "Harvard - positive"),
    ("Loughran-McDonald", "Pos", "Loughran-McDonald - positive"),
    ("Harvard", "Neg", "Harvard - negative"),
    ("Loughran-McDonald", "Neg", "Loughran-McDonald - negative"),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})

fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
axes = axes.flatten()
y_positions = np.arange(len(EFFECTS))[::-1]

for ax, (dictionary, outcome, title) in zip(axes, PANELS):
    for aggregation, style in AGG_STYLE.items():
        for y, (model_code, effect_name, _) in zip(y_positions, EFFECTS):
            row = long_results[
                (long_results["Dictionary"] == dictionary)
                & (long_results["Outcome"] == outcome)
                & (long_results["Aggregation"] == aggregation)
                & (long_results["Recession_Definition"] == PRIMARY_DEFINITION)
                & (long_results["Model_Code"] == model_code)
                & (long_results["Effect"] == effect_name)
            ].iloc[0]
            scale = 100 / row["Outcome_mean_pp"]
            yy = y + style["offset"]
            ax.plot([row["CI_low"] * scale, row["CI_high"] * scale], [yy, yy],
                    color=style["color"], linewidth=2, solid_capstyle="round")
            ax.plot(row["Coefficient"] * scale, yy, marker=style["marker"], markersize=6,
                    color=style["color"], markeredgecolor="white", markeredgewidth=1,
                    linestyle="none")

    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.axhline(1.5, color="#b5b4ae", linewidth=0.8, linestyle="--")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(axis="x", color="#eeede9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels([label for _, _, label in EFFECTS])
for ax in axes[2:]:
    ax.set_xlabel("Recession effect, % of series mean (HAC 95% CI)")

legend_handles = [
    plt.Line2D([0], [0], color=s["color"], marker=s["marker"], linewidth=2,
               markeredgecolor="white", label=label)
    for label, s in AGG_STYLE.items()
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.005), title="Quarterly aggregation")
fig.suptitle("Recession effect under two aggregation methods, 1948-2014 "
             "(268 quarters, majority-of-months recession definition)", y=1.01, fontsize=11)
fig.tight_layout()

figure_path = IMAGE_OUTPUT_DIR / "recession_effect_average_of_ratios_vs_pooled_words.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# WRITTEN SUMMARY
# ---------------------------------------------------------

lines = []
lines.append("WEEK 5 MODEL REVISION 04D")
lines.append("AGGREGATION METHOD ROBUSTNESS: AVERAGE OF RATIOS vs POOLED WORDS")
lines.append("=" * 78)
lines.append("")
lines.append("SAMPLE: full corpus, 1948Q1-2014Q4, 268 quarters in every model.")
lines.append("Inference: HAC / Newey-West, 4 lags. * = p < .05.")
lines.append("Coefficients in percentage points; [%] = % of the series' own mean.")
lines.append("")
lines.append("-" * 78)
lines.append("THE TWO QUARTERLY SERIES")
lines.append("-" * 78)
for _, r in series_comparison.iterrows():
    lines.append(f"{r['Dictionary']} {r['Outcome']}: mean {r['Mean_macro_pp']:.4f} (macro) vs "
                 f"{r['Mean_pooled_pp']:.4f} (pooled) pp; correlation {r['Correlation_macro_vs_pooled']:.3f}")
lines.append("")

for definition in RECESSION_DEFINITIONS:
    lines.append("-" * 78)
    lines.append(f"{definition}")
    lines.append("-" * 78)
    sub = comparison[comparison["Recession_Definition"] == definition]
    for dictionary in ["Harvard", "Loughran-McDonald"]:
        for outcome in ["Pos", "Neg"]:
            lines.append(f"{dictionary} {outcome}")
            for _, r in sub[(sub["Dictionary"] == dictionary) & (sub["Outcome"] == outcome)].iterrows():
                s_m = "*" if r["HAC_p_value_macro"] < 0.05 else " "
                s_p = "*" if r["HAC_p_value_pooled"] < 0.05 else " "
                flag = "" if r["Same_sign"] else "   SIGN DIFFERS"
                lines.append(
                    f"   {r['Model']:<36} macro {r['Coefficient_macro']:+.4f} "
                    f"[{r['Effect_pct_of_mean_macro']:+5.1f}%] p={r['HAC_p_value_macro']:.3f}{s_m} | "
                    f"pooled {r['Coefficient_pooled']:+.4f} "
                    f"[{r['Effect_pct_of_mean_pooled']:+5.1f}%] p={r['HAC_p_value_pooled']:.3f}{s_p}{flag}"
                )
            lines.append("")

lines.append("-" * 78)
lines.append("OVERALL AGREEMENT (both recession definitions)")
lines.append("-" * 78)
lines.append(f"Same sign:                  {n_same_sign}/{n_total}")
lines.append(f"Same 5% significance call:  {n_same_sig}/{n_total}")
lines.append("")
lines.append("-" * 78)
lines.append("HOW TO READ THIS")
lines.append("-" * 78)
lines.append("- If both methods give the same sign and the same significance verdict,")
lines.append("  the regression conclusion does not depend on how ads are combined")
lines.append("  into quarters, even if the magnitudes differ.")
lines.append("- Sign differences are only meaningful when the estimates are clearly")
lines.append("  away from zero; a flip between two near-zero, non-significant")
lines.append("  estimates means 'no detectable effect' under both methods.")
lines.append("- Pooling gives long records more weight. Long records are")
lines.append("  disproportionately financial notices, supplements and advertorials")
lines.append("  (Week 4), so pooled results describe overall advertising language")
lines.append("  and average-of-ratios results describe the typical advertisement.")
lines.append("- All results are descriptive associations, not causal effects.")

summary_path = TABLE_OUTPUT_DIR / "aggregation_method_regression_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("AGGREGATION METHOD REGRESSION CHECK COMPLETE")
print("=" * 90)
for path in [comparison_path, figure_path, summary_path]:
    print(f"  {path.relative_to(REPO_ROOT)}")
