# SOSC 314: LDA Topic Share Regression
# Tests whether recession quarters have different topic prevalence.
# Uses decade fixed effects, HAC standard errors, and Holm correction.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "week 5 - diagnostics robustness sensitivity validity regression checks and model revisions/04 topic model diagnostics/csv outputs/topic_prevalence_by_quarter.csv"

CSV_DIR = OUTPUT / "csv outputs"
IMAGE_DIR = OUTPUT / "diagnostic images"
CSV_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load existing topic prevalence data.
df = pd.read_csv(SOURCE)
df["Decade"] = (df["Year"] // 10) * 10
df["Quarter_Date"] = pd.PeriodIndex(df["Year_Quarter"], freq="Q")
df = df.sort_values("Quarter_Date").reset_index(drop=True)

topics = [f"topic_{i}" for i in range(10)]
results = []

# Regress each topic on recession status and decade effects.
for topic in topics:
    model = smf.ols(f"{topic} ~ Recession + C(Decade)", data=df).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}
    )

    coefficient = model.params["Recession"]
    ci_low, ci_high = model.conf_int().loc["Recession"]

    results.append({
        "topic": topic,
        "coefficient_pp": coefficient * 100,
        "se_pp": model.bse["Recession"] * 100,
        "ci_low_pp": ci_low * 100,
        "ci_high_pp": ci_high * 100,
        "p_value": model.pvalues["Recession"],
        "n_quarters": int(model.nobs)
    })

results = pd.DataFrame(results)

# Add stability information from the existing LDA diagnostics.
stability_path = ROOT / "week 5 - diagnostics robustness sensitivity validity regression checks and model revisions/04 topic model diagnostics/csv outputs/topic_recession_prevalence_tests.csv"

stability = pd.read_csv(stability_path)
stability = stability[stability["model"] == "Recession + decade FE"].copy()

stability["topic"] = "topic_" + stability["topic"].astype(str)

results = results.merge(
    stability[["topic", "stable_across_seeds", "top_words"]],
    on="topic",
    how="left",
    validate="one_to_one"
)

print("\nSTABLE TOPICS:")
print(results[results["stable_across_seeds"]][["topic", "top_words"]].to_string(index=False))

# Adjust for testing 10 topics.
results["p_holm"] = multipletests(results["p_value"], method="holm")[1]
results["significant_holm"] = results["p_holm"] < 0.05

results.to_csv(CSV_DIR / "topic_recession_regression_results.csv", index=False)

print("\nTOPIC SHARE REGRESSION RESULTS:")
print(results.round(4).to_string(index=False))

print("\nTopics significant after Holm correction:", results["significant_holm"].sum())

# Create coefficient plot.
plot = results.sort_values("coefficient_pp")

fig, ax = plt.subplots(figsize=(10, 6))

positions = np.arange(len(plot))
colors = ["#D55E00" if x else "#0072B2" for x in plot["significant_holm"]]

ax.barh(positions, plot["coefficient_pp"], color=colors, alpha=0.85)

ax.errorbar(
    plot["coefficient_pp"],
    positions,
    xerr=[
        plot["coefficient_pp"] - plot["ci_low_pp"],
        plot["ci_high_pp"] - plot["coefficient_pp"]
    ],
    fmt="none",
    ecolor="black",
    capsize=3
)

ax.set_yticks(positions)
ax.set_yticklabels([
    f"{row.topic} {'(stable)' if row.stable_across_seeds else '(exploratory)'}"
    for row in plot.itertuples()
])
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("Recession-associated difference in topic share (percentage points)")
ax.set_title("LDA Topic Prevalence During Recessions\nDecade Fixed Effects and HAC Standard Errors")

plt.tight_layout()
plt.savefig(IMAGE_DIR / "topic_recession_effects_hac_holm.png", dpi=300)
plt.close()

print("\nSaved regression table and figure.")