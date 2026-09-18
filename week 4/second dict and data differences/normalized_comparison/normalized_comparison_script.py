import os, subprocess, pandas as pd

REPO = "/content/Jake_Daniel_Kiran_SOSC314"
W3, W4 = "week three data cleaning", "week 4/second dict and data differences/csv"

FILES = {
    ("Harvard", "All words"):      f"{W3}/week_three_quarterly_sentiment.csv",
    ("Harvard", "No stopwords"):   f"{W3}/week_three_quarterly_sentiment_stopwords.csv",
    ("Harvard", "Dictionary only"):f"{W3}/week_three_quarterly_sentiment_harvard_only.csv",
    ("LM", "All words"):           f"{W4}/economist_loughran_mcdonald_quarterly.csv",
    ("LM", "No stopwords"):        f"{W4}/economist_lm_stopwords_quarterly.csv",
    ("LM", "Dictionary only"):     f"{W4}/economist_lm_only_quarterly.csv",
}
NBER = f"{W3}/nber_recession_table.csv"

subprocess.run(f'git lfs pull --include="{",".join(list(FILES.values())+[NBER])}"',
               shell=True, cwd=REPO)

nber = pd.read_csv(f"{REPO}/{NBER}", usecols=["Year_Quarter", "Recession_Status"])

rows = []
for (d, variant), rel in FILES.items():
    q = pd.read_csv(f"{REPO}/{rel}")
    pos = "Positiv_Ratio" if d == "Harvard" else "Positive_Ratio"
    q = q.merge(nber, on="Year_Quarter", how="inner")
    g = q.groupby("Recession_Status")[pos].mean()

    level = 100 * g.mean()
    diff  = 100 * (g["Recession"] - g["Not Recession"])
    rows.append({"Dictionary": d, "Variant": variant,
                 "Level_pp": level, "Diff_pp": diff,
                 "Diff_pct_of_level": 100 * diff / level})

t = pd.DataFrame(rows).round(4)
print(t.to_string(index=False))

# SOSC 314 - WEEK 4: COMPARING EFFECTS ON A COMMON SCALE

# WHY THIS EXISTS
#
# The six dictionary/preprocessing combinations produce sentiment
# ratios on different scales, because each divides by a
# different denominator. Restricting to dictionary vocabulary cuts the
# denominator to a fraction of its original size, so every ratio
# inflates, and so does any difference between groups.
#
# Plotting the raw differences on one axis therefore invites a misleading reading: the Loughran-McDonald dictionary-only bar looks like a
# twenty-fold stronger effect when it is mostly a change of units.
#
# The fix is to express each recession difference relative to its own
# variant's mean level. That number is comparable across variants.
#
# DESCRIPTIVE ONLY. No models are fitted here. Group means without
# controls for the long-run trend; inference is Week 5 work.
#
# Output: normalized_effect_comparison.csv
#         normalized_effect_comparison.png

import os
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

REPO = "/content/Jake_Daniel_Kiran_SOSC314"
OUT_DIR = "/content/week_four_normalized"
os.makedirs(OUT_DIR, exist_ok=True)

W3 = "week three data cleaning"
W4 = "week 4/second dict and data differences/csv"

# All quarterly files
FILES = {
    ("Harvard", "All words"):       f"{W3}/week_three_quarterly_sentiment.csv",
    ("Harvard", "No stopwords"):    f"{W3}/week_three_quarterly_sentiment_stopwords.csv",
    ("Harvard", "Dictionary only"): f"{W3}/week_three_quarterly_sentiment_harvard_only.csv",
    ("Loughran-McDonald", "All words"):       f"{W4}/economist_loughran_mcdonald_quarterly.csv",
    ("Loughran-McDonald", "No stopwords"):    f"{W4}/economist_lm_stopwords_quarterly.csv",
    ("Loughran-McDonald", "Dictionary only"): f"{W4}/economist_lm_only_quarterly.csv",
}
NBER = f"{W3}/nber_recession_table.csv"


def is_pointer(path):
    """An unresolved Git LFS file is a 3-line stub, not real data."""
    try:
        with open(path) as f:
            return f.readline().startswith("version https://git-lfs")
    except (FileNotFoundError, UnicodeDecodeError):
        return True


need = [r for r in list(FILES.values()) + [NBER]
        if is_pointer(os.path.join(REPO, r))]
if need:
    print(f"Downloading {len(need)} small file(s)...")
    subprocess.run(f'git lfs pull --include="{",".join(need)}"',
                   shell=True, cwd=REPO)

nber = pd.read_csv(os.path.join(REPO, NBER),
                   usecols=["Year_Quarter", "Recession_Status"])

# COMPUTE
# For each variant:
#   Level_pp = mean positive share, in percentage points
#   Diff_pp = recession mean minus expansion mean
#   Diff_pct_of_level = the difference relative to that variant's own scale. This is the only column comparable across variants

rows = []

for (dictionary, variant), rel in FILES.items():

    q = pd.read_csv(os.path.join(REPO, rel))

    # The two dictionaries name their columns differently:
    # Harvard General Inquirer -> Positiv_Ratio / Negativ_Ratio
    # Loughran-McDonald -> Positive_Ratio / Negative_Ratio
    pos = "Positiv_Ratio" if dictionary == "Harvard" else "Positive_Ratio"
    neg = "Negativ_Ratio" if dictionary == "Harvard" else "Negative_Ratio"

    q = q.merge(nber, on="Year_Quarter", how="inner")
    g = q.groupby("Recession_Status")[[pos, neg]].mean()

    level = 100 * g[pos].mean()
    diff = 100 * (g.loc["Recession", pos] - g.loc["Not Recession", pos])
    neg_diff = 100 * (g.loc["Recession", neg] - g.loc["Not Recession", neg])

    rows.append({
        "Dictionary": dictionary,
        "Variant": variant,
        "Quarters": len(q),
        "Level_pp": level,
        "Pos_Diff_pp": diff,
        "Neg_Diff_pp": neg_diff,
        "Pos_Diff_pct_of_level": 100 * diff / level,
    })

results = pd.DataFrame(rows).round(4)

print("\n" + "=" * 84)
print("RECESSION EFFECT ON A COMMON SCALE")
print("=" * 84)
print(results.to_string(index=False))

results.to_csv(f"{OUT_DIR}/normalized_effect_comparison.csv", index=False)

# FIGUREs
# Grouped bars: one group per preprocessing variant, one bar per
# dictionary. Because the y-axis is the effect relative to each
# variant's own level, the bars are directly comparable, which the raw
# percentage-point version is not.

ORDER = ["All words", "No stopwords", "Dictionary only"]
DICTS = ["Harvard", "Loughran-McDonald"]

fig, ax = plt.subplots(figsize=(9, 5.5))

width = 0.36
x = range(len(ORDER))

for i, d in enumerate(DICTS):
    vals = [
        results[(results["Dictionary"] == d)
                & (results["Variant"] == v)]["Pos_Diff_pct_of_level"].iloc[0]
        for v in ORDER
    ]
    pos_x = [p + (i - 0.5) * width for p in x]
    bars = ax.bar(pos_x, vals, width, label=d)

    # Label each bar
    for b, v in zip(bars, vals):
        offset = -0.18 if v < 0 else 0.18
        ax.text(b.get_x() + b.get_width() / 2,
                v + offset,
                f"{v:.1f}%",
                ha="center",
                va="top" if v < 0 else "bottom",
                fontsize=9)

ax.margins(y=0.16)

ax.axhline(0, linewidth=0.8, color="black")
ax.set_xticks(list(x))
ax.set_xticklabels(ORDER)
ax.set_xlabel("Preprocessing variant")
ax.set_ylabel("Recession effect, % of that variant's mean level")
ax.set_title("Recession effect on positive sentiment, normalised by scale\n"
             "(negative = less positive language during recessions)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/normalized_effect_comparison.png", dpi=200)
plt.show()

# download

import shutil
shutil.make_archive("/content/week_four_normalized", "zip", OUT_DIR)

print("\nFiles ready:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)

try:
    from google.colab import files
    files.download("/content/week_four_normalized.zip")
