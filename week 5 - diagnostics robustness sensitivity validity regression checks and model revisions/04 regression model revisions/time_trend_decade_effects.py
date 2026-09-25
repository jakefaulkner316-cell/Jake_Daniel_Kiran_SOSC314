# WEEK 5 MODEL REVISION
# TIME TREND, DECADE EFFECTS, POST-1973 INTERACTION, AND RECESSION-DEFINITION ROBUSTNESS
#
# PURPOSE:
# Test whether the recession-sentiment association survives
# alternative ways of accounting for long-run change in
# advertising language, and whether it depends on how
# recession quarters are defined.
#
# MAIN DIAGNOSTIC QUESTION:
# Once long-run drift in advertising language is controlled for,
# is sentiment still different during recession quarters? And is
# that difference the same before and after 1973?
#
# WHY THIS DIAGNOSTIC MATTERS:
# 1. Decade-to-decade variation in positive sentiment is roughly
#    40x the raw recession-expansion gap (Week 3), and recession
#    quarters are unevenly spread over time (nine in the 1970s,
#    none after 2009). A simple recession dummy can therefore
#    pick up long-run drift instead of the business cycle.
#    Script 04b found a very strong linear trend, but only on the
#    1990-2014 sample. Here the trend is tested on all 268
#    quarters, so no comparison is confounded by a sample change.
#
# 2. A linear trend assumes drift is a straight line. Positive
#    sentiment actually falls to a trough around 1970 and then
#    rises. Decade fixed effects allow any shape of drift, so
#    comparing the two shows whether the conclusion depends on
#    the functional form of the time control.
#
# 3. METHODOLOGY.md commits to a pre/post-1973 comparison:
#    business cycles were less synchronised internationally under
#    Bretton Woods (Bordo and Helbling 2010), so U.S. NBER dates
#    may describe The Economist's advertising environment less
#    well before 1973. As agreed in Week 4, this is estimated as
#    an interaction term (Recession x Post-1973) on the full
#    sample, not by splitting the sample in two. The interaction
#    keeps all 268 quarters and directly tests whether the
#    recession effect differs between periods.
#
# 4. METHODOLOGY.md also commits to a recession-definition check.
#    A quarter is a recession quarter when at least 2 of its 3
#    months are in an NBER contraction (42 quarters). The
#    alternative counts any quarter with at least 1 recession
#    month (48 quarters). The 6 quarters where they disagree are
#    the boundary quarters at the start and end of recessions.
#
# 5. Aggregation robustness (Week 4): every model is run on both
#    the "average of ratios" (macro) and "pooled words" quarterly
#    series. Macro is the primary specification, matching scripts
#    03a, 03b, 04 and 04b.
#
# MODELS (each run for 2 dictionaries x 2 outcomes x
#         2 recession definitions x 2 aggregations = 16 series):
#
#   A. Sentiment = Recession
#   B. Sentiment = Recession + Time Trend
#   C. Sentiment = Recession + Decade Fixed Effects
#   D. Sentiment = Recession + Time Trend + Post1973
#                  + Recession x Post1973
#   E. Sentiment = Recession + Decade Fixed Effects + Post1973
#                  + Recession x Post1973
#
#   From D and E we report the recession effect BEFORE 1973
#   (the Recession coefficient) and AFTER 1973 (Recession +
#   Recession x Post1973, tested as a linear combination), plus
#   the p-value of the interaction itself (is the difference
#   between the two periods statistically distinguishable?).
#
# STANDARD ERRORS:
# HAC / Newey-West, 4 lags, using the same statsmodels call as
# scripts 04 and 04b so results are directly comparable. Model A
# with the majority definition and macro aggregation reproduces
# the baseline HAC results in 03b / 04b (check printed below).
#
# SAMPLE:
# Full corpus, 1948Q1-2014Q4, 268 quarters in every model.
# (Same quarterly files as 03a/03b/04/04b.)
#
# FILES LOADED (all small plain-text CSVs already in the repo;
# no Git LFS download is needed):
#   week 4 - updated quality filtered analysis/week4_aggregations/
#       quarterly_harvard_macro.csv
#       quarterly_harvard_pooled.csv
#       quarterly_loughran_macro.csv
#       quarterly_loughran_pooled.csv
#   week three data cleaning/
#       nber_recession_table.csv
#
# OUTPUTS (in this script's folder):
#   csv outputs/
#       time_trend_decade_all_coefficients.csv
#           every coefficient from every model (long format)
#       time_trend_decade_recession_effects.csv
#           one row per model x series: the recession effect(s),
#           HAC SE, p-value, 95% CI, effect as % of mean level,
#           R-squared and Durbin-Watson
#   diagnostic images/
#       recession_effect_across_time_controls_and_definitions.png
#           candidate Week 5 report figure
#   robustness tables/
#       time_trend_decade_recession_definition_summary.txt
#
# INTERPRETATION RULE:
# These are descriptive associations, not causal effects. Use
# "is associated with", "differs during", "is robust to",
# "is sensitive to".
# =========================================================


#  ------------------------
# IMPORT PACKAGES
#  #

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


#  #
# DEFINE FILE LOCATIONS
#  #

SCRIPT_DIR = Path(__file__).resolve().parent
WEEK5_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEEK5_DIR.parent

WEEK4_AGG_DIR = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
    / "week4_aggregations"
)

RECESSION_FILE = (
    REPO_ROOT
    / "week three data cleaning"
    / "nber_recession_table.csv"
)

SENTIMENT_FILES = {
    ("Harvard", "Average of ratios"): WEEK4_AGG_DIR / "quarterly_harvard_macro.csv",
    ("Harvard", "Pooled words"): WEEK4_AGG_DIR / "quarterly_harvard_pooled.csv",
    ("Loughran-McDonald", "Average of ratios"): WEEK4_AGG_DIR / "quarterly_loughran_macro.csv",
    ("Loughran-McDonald", "Pooled words"): WEEK4_AGG_DIR / "quarterly_loughran_pooled.csv",
}


#  #
# DEFINE OUTPUT FOLDERS
#  #

CSV_OUTPUT_DIR = SCRIPT_DIR / "csv outputs"
IMAGE_OUTPUT_DIR = SCRIPT_DIR / "diagnostic images"
TABLE_OUTPUT_DIR = SCRIPT_DIR / "robustness tables"

for folder in [CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


#  #
# SETTINGS
#  #

HAC_LAGS = 4
BRETTON_WOODS_BREAK_YEAR = 1973   # Post1973 = 1 from 1973Q1 onward

RECESSION_DEFINITIONS = {
    "Majority of months (42 quarters)": "Recession_Majority",
    "Any month (48 quarters)": "Recession_Any",
}

PRIMARY_DEFINITION = "Majority of months (42 quarters)"
PRIMARY_AGGREGATION = "Average of ratios"

MODEL_LABELS = {
    "A": "A. Recession only",
    "B": "B. + linear time trend",
    "C": "C. + decade fixed effects",
    "D": "D. + trend + Recession x Post-1973",
    "E": "E. + decade FE + Recession x Post-1973",
}


#  #
# CHECK THAT INPUT FILES EXIST AND ARE NOT LFS STUBS
#  #
# The repo stores most CSVs in Git LFS. If a file was cloned
# without pulling LFS content, it is a tiny text pointer that
# starts with "version https://git-lfs". Catch that early with
# a clear message instead of a confusing pandas error.
#  #

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


for path in list(SENTIMENT_FILES.values()) + [RECESSION_FILE]:
    check_input_file(path)


#  #
# LOAD AND PREPARE RECESSION DATA
#  #

recession_data = pd.read_csv(RECESSION_FILE)

recession_data = recession_data[
    ["Year_Quarter", "Recession_Majority", "Recession_Any"]
].copy()

for column in ["Recession_Majority", "Recession_Any"]:
    recession_data[column] = (
        recession_data[column].astype(str).str.strip().str.lower() == "true"
    ).astype(int)

n_majority = int(recession_data["Recession_Majority"].sum())
n_any = int(recession_data["Recession_Any"].sum())

boundary_quarters = recession_data.loc[
    recession_data["Recession_Majority"] != recession_data["Recession_Any"],
    "Year_Quarter",
].tolist()

print("\n" + "=" * 90)
print("RECESSION DEFINITIONS")
print("=" * 90)
print(f"Majority-of-months recession quarters: {n_majority}")
print(f"Any-month recession quarters:          {n_any}")
print(f"Quarters where the definitions differ: {len(boundary_quarters)}")
print("  " + ", ".join(boundary_quarters))


# LOAD SENTIMENT SERIES AND BUILD TIME VARIABLES

# Sentiment is converted to percentage points (x100) so that
# coefficients are on the same scale as scripts 03a-04b.

def load_series(path):
    data = pd.read_csv(path)
    data = data[["Year_Quarter", "Pos", "Neg", "Ads"]].copy()
    data = data.merge(recession_data, on="Year_Quarter", how="inner")
    data = data.sort_values("Year_Quarter").reset_index(drop=True)

    data["Pos"] = 100 * data["Pos"]
    data["Neg"] = 100 * data["Neg"]

    data["Year"] = data["Year_Quarter"].str[:4].astype(int)

    # Linear time trend: 0 for the first quarter, 1 for the next...
    data["Time_Trend"] = np.arange(len(data))

    # Post-Bretton-Woods indicator.
    data["Post1973"] = (data["Year"] >= BRETTON_WOODS_BREAK_YEAR).astype(int)

    # Decade label, e.g. 1948 -> "1940s". 2010-2014 forms "2010s".
    data["Decade"] = (data["Year"] // 10 * 10).astype(str) + "s"

    return data


series = {key: load_series(path) for key, path in SENTIMENT_FILES.items()}

for key, data in series.items():
    if len(data) != 268:
        raise ValueError(f"{key}: expected 268 quarters, found {len(data)}")

example = series[("Harvard", "Average of ratios")]

print("\n" + "=" * 90)
print("SAMPLE")
print("=" * 90)
print(f"Quarters: {len(example)}  ({example['Year_Quarter'].iloc[0]} to "
      f"{example['Year_Quarter'].iloc[-1]})")
print(f"Pre-1973 quarters:  {int((example['Post1973'] == 0).sum())}")
print(f"Post-1973 quarters: {int((example['Post1973'] == 1).sum())}")

print("\nRecession quarters by period and definition:")
for label, column in RECESSION_DEFINITIONS.items():
    pre = int(example.loc[example["Post1973"] == 0, column].sum())
    post = int(example.loc[example["Post1973"] == 1, column].sum())
    print(f"  {label:<36} pre-1973: {pre:>2}   post-1973: {post:>2}")

print("\nRecession quarters (majority definition) by decade:")
print(example.groupby("Decade")["Recession_Majority"].sum().to_string())


#  #
# REGRESSION HELPER
#  #

def fit_hac(data, outcome, predictors):
    """
    OLS with HAC / Newey-West standard errors (4 lags).
    Uses the same call as scripts 04 and 04b.
    Returns the HAC results and the list of column names.
    """
    X = sm.add_constant(data[predictors].astype(float))
    y = data[outcome].astype(float)

    ols_model = sm.OLS(y, X).fit()
    hac_model = ols_model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)

    return hac_model, list(X.columns)


def linear_combination(hac_model, names, weights):
    """
    Estimate and test a weighted sum of coefficients,
    e.g. Recession + Recession_x_Post1973.
    """
    r = np.zeros(len(names))
    for name, weight in weights.items():
        r[names.index(name)] = weight

    test = hac_model.t_test(r)
    estimate = float(np.squeeze(test.effect))
    se = float(np.squeeze(test.sd))
    p = float(np.squeeze(test.pvalue))
    ci_low, ci_high = np.squeeze(test.conf_int())

    return estimate, se, p, float(ci_low), float(ci_high)


#  #
# RUN ALL MODELS
#  #

coefficient_rows = []   # every coefficient
effect_rows = []        # the recession effect(s) only

for (dictionary, aggregation), base_data in series.items():

    # Decade dummies (1940s = reference category).
    decade_dummies = pd.get_dummies(
        base_data["Decade"], prefix="Decade", drop_first=True
    ).astype(int)
    decade_columns = list(decade_dummies.columns)
    data_with_decades = pd.concat([base_data, decade_dummies], axis=1)

    for definition_label, recession_column in RECESSION_DEFINITIONS.items():

        data = data_with_decades.copy()
        data["Recession"] = data[recession_column]
        data["Recession_x_Post1973"] = data["Recession"] * data["Post1973"]

        model_specs = {
            "A": ["Recession"],
            "B": ["Recession", "Time_Trend"],
            "C": ["Recession"] + decade_columns,
            "D": ["Recession", "Time_Trend", "Post1973", "Recession_x_Post1973"],
            "E": ["Recession"] + decade_columns + ["Post1973", "Recession_x_Post1973"],
        }

        for outcome in ["Pos", "Neg"]:

            outcome_mean = data[outcome].mean()

            for model_code, predictors in model_specs.items():

                hac_model, names = fit_hac(data, outcome, predictors)
                dw = durbin_watson(hac_model.resid)

                common = {
                    "Dictionary": dictionary,
                    "Outcome": outcome,
                    "Aggregation": aggregation,
                    "Recession_Definition": definition_label,
                    "Model_Code": model_code,
                    "Model": MODEL_LABELS[model_code],
                    "N_Quarters": int(hac_model.nobs),
                    "R_squared": hac_model.rsquared,
                    "Durbin_Watson": dw,
                }

                # --- every coefficient -------------------------------
                ci = hac_model.conf_int()
                for i, name in enumerate(names):
                    coefficient_rows.append({
                        **common,
                        "Variable": name,
                        "Coefficient": hac_model.params[i],
                        "HAC_SE": hac_model.bse[i],
                        "HAC_p_value": hac_model.pvalues[i],
                        "CI_low": ci[i][0],
                        "CI_high": ci[i][1],
                    })

                # --- recession effect(s) -----------------------------
                rec_index = names.index("Recession")

                if model_code in ["A", "B", "C"]:
                    effect_rows.append({
                        **common,
                        "Effect": "Recession (whole period)",
                        "Coefficient": hac_model.params[rec_index],
                        "HAC_SE": hac_model.bse[rec_index],
                        "HAC_p_value": hac_model.pvalues[rec_index],
                        "CI_low": ci[rec_index][0],
                        "CI_high": ci[rec_index][1],
                        "Interaction_p_value": np.nan,
                        "Outcome_mean_pp": outcome_mean,
                    })

                else:
                    int_index = names.index("Recession_x_Post1973")
                    interaction_p = hac_model.pvalues[int_index]

                    # Pre-1973 effect = Recession coefficient
                    effect_rows.append({
                        **common,
                        "Effect": "Recession, pre-1973",
                        "Coefficient": hac_model.params[rec_index],
                        "HAC_SE": hac_model.bse[rec_index],
                        "HAC_p_value": hac_model.pvalues[rec_index],
                        "CI_low": ci[rec_index][0],
                        "CI_high": ci[rec_index][1],
                        "Interaction_p_value": interaction_p,
                        "Outcome_mean_pp": outcome_mean,
                    })

                    # Post-1973 effect = Recession + interaction
                    est, se, p, lo, hi = linear_combination(
                        hac_model, names,
                        {"Recession": 1, "Recession_x_Post1973": 1},
                    )
                    effect_rows.append({
                        **common,
                        "Effect": "Recession, post-1973",
                        "Coefficient": est,
                        "HAC_SE": se,
                        "HAC_p_value": p,
                        "CI_low": lo,
                        "CI_high": hi,
                        "Interaction_p_value": interaction_p,
                        "Outcome_mean_pp": outcome_mean,
                    })


coefficients = pd.DataFrame(coefficient_rows)
effects = pd.DataFrame(effect_rows)

# Effect as % of the outcome's own mean level (Week 4 scale
# normalisation), so Harvard and LM can be compared directly.
effects["Effect_pct_of_mean"] = 100 * effects["Coefficient"] / effects["Outcome_mean_pp"]

effects["Significant_5pct"] = effects["HAC_p_value"] < 0.05


#  #
# SANITY CHECK AGAINST EARLIER WEEK 5 RESULTS
#  #
# Model A (majority, macro) must reproduce the baseline HAC
# numbers from 03b / 04b.
#  #

expected_baseline = {
    ("Harvard", "Pos"): (-0.0291, 0.8089),
    ("Harvard", "Neg"): (0.0021, 0.9102),
    ("Loughran-McDonald", "Pos"): (-0.0753, 0.3897),
    ("Loughran-McDonald", "Neg"): (0.0250, 0.2041),
}

print("\n" + "=" * 90)
print("SANITY CHECK: MODEL A REPRODUCES 03b/04b BASELINE (HAC)")
print("=" * 90)

for (dictionary, outcome), (exp_coef, exp_p) in expected_baseline.items():
    row = effects[
        (effects["Dictionary"] == dictionary)
        & (effects["Outcome"] == outcome)
        & (effects["Aggregation"] == PRIMARY_AGGREGATION)
        & (effects["Recession_Definition"] == PRIMARY_DEFINITION)
        & (effects["Model_Code"] == "A")
    ].iloc[0]
    ok = abs(row["Coefficient"] - exp_coef) < 0.0006 and abs(row["HAC_p_value"] - exp_p) < 0.002
    status = "OK" if ok else "MISMATCH - check inputs"
    print(f"{dictionary:<18} {outcome}:  coef {row['Coefficient']:+.4f} (expected {exp_coef:+.4f})"
          f"   p {row['HAC_p_value']:.4f} (expected {exp_p:.4f})   {status}")


#  #
# SAVE CSV OUTPUTS
#  #

coefficients_path = CSV_OUTPUT_DIR / "time_trend_decade_all_coefficients.csv"
effects_path = CSV_OUTPUT_DIR / "time_trend_decade_recession_effects.csv"

coefficients.round(6).to_csv(coefficients_path, index=False)
effects.round(6).to_csv(effects_path, index=False)


#  #
# PRINT THE PRIMARY RESULTS
#  #

def primary_table(frame, definition=PRIMARY_DEFINITION, aggregation=PRIMARY_AGGREGATION):
    sub = frame[
        (frame["Recession_Definition"] == definition)
        & (frame["Aggregation"] == aggregation)
    ].copy()
    sub["Series"] = sub["Dictionary"].str.replace("Loughran-McDonald", "LM") + " " + sub["Outcome"]
    return sub


display_columns = ["Series", "Model_Code", "Effect", "Coefficient", "HAC_p_value",
                   "Effect_pct_of_mean", "Interaction_p_value", "R_squared", "Durbin_Watson"]

print("\n" + "=" * 90)
print(f"RECESSION EFFECTS - {PRIMARY_DEFINITION}, {PRIMARY_AGGREGATION}")
print("(coefficients in percentage points; HAC 4 lags)")
print("=" * 90)
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(primary_table(effects)[display_columns].round(4).to_string(index=False))


# Time-control coefficients (full sample) for the summary.
trend_rows = coefficients[
    (coefficients["Variable"] == "Time_Trend")
    & (coefficients["Model_Code"] == "B")
    & (coefficients["Recession_Definition"] == PRIMARY_DEFINITION)
    & (coefficients["Aggregation"] == PRIMARY_AGGREGATION)
]

print("\n" + "=" * 90)
print("LINEAR TIME TREND, FULL SAMPLE (Model B, pp per quarter)")
print("=" * 90)
for _, row in trend_rows.iterrows():
    print(f"{row['Dictionary']:<18} {row['Outcome']}:  {row['Coefficient']:+.5f} pp/quarter "
          f"({row['Coefficient'] * 40:+.3f} pp per decade), p = {row['HAC_p_value']:.4g}, "
          f"R2 = {row['R_squared']:.3f}")


#  #
# DIAGNOSTIC FIGURE
#  #
# Coefficient plot: recession effect (HAC 95% CI) across the
# time-control specifications, for both recession definitions.
# One panel per dictionary x outcome, primary aggregation.
# Effects are shown as % of each series' mean level so the four
# panels share one comparable x-axis (Week 4 normalisation).
#  #

ROW_ORDER = [
    ("A", "Recession (whole period)", "Recession only"),
    ("B", "Recession (whole period)", "+ linear trend"),
    ("C", "Recession (whole period)", "+ decade fixed effects"),
    ("D", "Recession, pre-1973", "Trend + interaction: pre-1973"),
    ("D", "Recession, post-1973", "Trend + interaction: post-1973"),
    ("E", "Recession, pre-1973", "Decade FE + interaction: pre-1973"),
    ("E", "Recession, post-1973", "Decade FE + interaction: post-1973"),
]

DEFINITION_STYLE = {
    "Majority of months (42 quarters)": {"color": "#2a78d6", "marker": "o", "offset": -0.14},
    "Any month (48 quarters)": {"color": "#eb6834", "marker": "s", "offset": 0.14},
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

fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True)
axes = axes.flatten()

y_positions = np.arange(len(ROW_ORDER))[::-1]

for ax, (dictionary, outcome, title) in zip(axes, PANELS):

    for definition_label, style in DEFINITION_STYLE.items():
        for y, (model_code, effect_name, _) in zip(y_positions, ROW_ORDER):
            row = effects[
                (effects["Dictionary"] == dictionary)
                & (effects["Outcome"] == outcome)
                & (effects["Aggregation"] == PRIMARY_AGGREGATION)
                & (effects["Recession_Definition"] == definition_label)
                & (effects["Model_Code"] == model_code)
                & (effects["Effect"] == effect_name)
            ].iloc[0]

            scale = 100 / row["Outcome_mean_pp"]
            est = row["Coefficient"] * scale
            lo = row["CI_low"] * scale
            hi = row["CI_high"] * scale
            yy = y + style["offset"]

            ax.plot([lo, hi], [yy, yy], color=style["color"], linewidth=2,
                    solid_capstyle="round")
            ax.plot(est, yy, marker=style["marker"], markersize=6,
                    color=style["color"], markeredgecolor="white", markeredgewidth=1,
                    linestyle="none")

    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.axhline(3.5, color="#b5b4ae", linewidth=0.8, linestyle="--")   # whole-period rows above, pre/post-1973 rows below
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(axis="x", color="#eeede9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels([label for _, _, label in ROW_ORDER])

for ax in axes[2:]:
    ax.set_xlabel("Recession effect, % of series mean (HAC 95% CI)")

legend_handles = [
    plt.Line2D([0], [0], color=s["color"], marker=s["marker"], linewidth=2,
               markeredgecolor="white", label=label)
    for label, s in DEFINITION_STYLE.items()
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.005), title="Recession definition")

fig.suptitle(
    "Recession effect on ad sentiment across time controls, 1948-2014 (268 quarters, average of ratios)",
    y=1.01, fontsize=11,
)
fig.tight_layout()

figure_path = IMAGE_OUTPUT_DIR / "recession_effect_across_time_controls_and_definitions.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)


#  #
# WRITTEN SUMMARY
#  #

def fmt_effect(row):
    stars = "*" if row["HAC_p_value"] < 0.05 else ""
    return (f"{row['Coefficient']:+.4f} pp ({row['Effect_pct_of_mean']:+.1f}% of mean), "
            f"HAC p = {row['HAC_p_value']:.3f}{stars}")


def get_effect(dictionary, outcome, model_code, effect_name,
               definition=PRIMARY_DEFINITION, aggregation=PRIMARY_AGGREGATION):
    return effects[
        (effects["Dictionary"] == dictionary)
        & (effects["Outcome"] == outcome)
        & (effects["Aggregation"] == aggregation)
        & (effects["Recession_Definition"] == definition)
        & (effects["Model_Code"] == model_code)
        & (effects["Effect"] == effect_name)
    ].iloc[0]


lines = []
lines.append("WEEK 5 MODEL REVISION 04C")
lines.append("TIME TREND, DECADE EFFECTS, POST-1973 INTERACTION, RECESSION DEFINITION")
lines.append("=" * 78)
lines.append("")
lines.append("SAMPLE: full corpus, 1948Q1-2014Q4, 268 quarters in every model.")
lines.append(f"Recession quarters: {n_majority} (majority of months) / {n_any} (any month).")
lines.append(f"Boundary quarters that differ: {', '.join(boundary_quarters)}")
lines.append(f"Pre-1973 quarters: {int((example['Post1973'] == 0).sum())}; "
             f"post-1973 quarters: {int((example['Post1973'] == 1).sum())}.")
lines.append("Inference: HAC / Newey-West, 4 lags. * = p < .05.")
lines.append("Effects in percentage points and as % of each series' mean level.")
lines.append("")

for definition in RECESSION_DEFINITIONS:
    for aggregation in ["Average of ratios", "Pooled words"]:
        lines.append("-" * 78)
        lines.append(f"{definition} | {aggregation}")
        lines.append("-" * 78)
        for dictionary in ["Harvard", "Loughran-McDonald"]:
            for outcome in ["Pos", "Neg"]:
                lines.append(f"{dictionary} {outcome}")
                for model_code, effect_name, label in ROW_ORDER:
                    row = get_effect(dictionary, outcome, model_code, effect_name,
                                     definition, aggregation)
                    extra = ""
                    if model_code in ["D", "E"] and effect_name.endswith("post-1973"):
                        extra = f"  [interaction p = {row['Interaction_p_value']:.3f}]"
                    lines.append(f"   {label:<36} {fmt_effect(row)}{extra}")
                r2_a = get_effect(dictionary, outcome, "A", "Recession (whole period)",
                                  definition, aggregation)
                r2_b = get_effect(dictionary, outcome, "B", "Recession (whole period)",
                                  definition, aggregation)
                r2_c = get_effect(dictionary, outcome, "C", "Recession (whole period)",
                                  definition, aggregation)
                lines.append(f"   R2: A {r2_a['R_squared']:.3f} | B {r2_b['R_squared']:.3f} | "
                             f"C {r2_c['R_squared']:.3f}    "
                             f"Durbin-Watson: A {r2_a['Durbin_Watson']:.2f} | "
                             f"B {r2_b['Durbin_Watson']:.2f} | C {r2_c['Durbin_Watson']:.2f}")
                lines.append("")

lines.append("-" * 78)
lines.append("LINEAR TIME TREND (Model B, majority definition, average of ratios)")
lines.append("-" * 78)
for _, row in trend_rows.iterrows():
    lines.append(f"{row['Dictionary']} {row['Outcome']}: {row['Coefficient']:+.5f} pp/quarter, "
                 f"HAC p = {row['HAC_p_value']:.4g}")
lines.append("")

lines.append("-" * 78)
lines.append("HOW TO READ THIS")
lines.append("-" * 78)
lines.append("- Compare A with B and C: if the recession coefficient moves a lot once a")
lines.append("  time control is added, part of the raw recession gap reflects long-run")
lines.append("  drift in advertising language rather than the business cycle.")
lines.append("- Compare B with C: a linear trend and decade effects model drift")
lines.append("  differently. Agreement means the conclusion does not depend on how")
lines.append("  drift is modelled; disagreement means it does, and should be reported.")
lines.append("- D and E: the interaction p-value tests whether the recession effect")
lines.append("  differs before vs after 1973. A non-significant interaction does NOT")
lines.append("  show the effects are equal; with few recession quarters per period")
lines.append("  the test has low power.")
lines.append("- Majority vs any-month: differences come only from the 6 boundary")
lines.append("  quarters at the start and end of recessions.")
lines.append("- Durbin-Watson well below 2 means residuals remain serially correlated,")
lines.append("  which is why HAC inference is used throughout.")
lines.append("- All results are descriptive associations, not causal effects.")

summary_path = TABLE_OUTPUT_DIR / "time_trend_decade_recession_definition_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")


#  #
# DONE
#  #

print("\n" + "=" * 90)
print("TIME TREND / DECADE EFFECTS / RECESSION DEFINITION CHECKS COMPLETE")
print("=" * 90)
for path in [coefficients_path, effects_path, figure_path, summary_path]:
    print(f"  {path.relative_to(REPO_ROOT)}")
