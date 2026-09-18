# WEEK 4: AGGREGATION COMPARISON AND SENSITIVITY CHECK
#
# No models are fitted.
#
# 1. AGGREGATION. Every quarterly number in this project so far was
#    built by averaging the per-advertisement ratios. That treats a
#    10-word advertisement exactly like an 8,062-word one. The
#    alternative is to pool: add up the category words across the whole
#    quarter, add up the total words, and divide once.
#
#      macro (current) = mean of (category words / total words)
#      pooled          = sum(category words) / sum(total words)
#
#    They answer different questions. Macro asks how positive the
#    typical advertisement is. Pooled asks how positive advertising
#    language is overall. Neither is wrong; we report both.
#
# 2. SENSITIVITY. We built corpus quality flags but have never applied
#    them. Here we rerun everything with and without them.
#
# Output: a table of 8 rows (2 dictionaries x 2 aggregations x 2 filter
# states) and one figure.

import os
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

REPO_DIR = "/content/Jake_Daniel_Kiran_SOSC314"
OUT_DIR = "/content/week_four_aggregation"
os.makedirs(OUT_DIR, exist_ok=True)

HARVARD_REL = ("week three data cleaning/Archived Data/"
               "week_three_ads_with_sentiment.csv")
LM_REL = ("week 4/second dict and data differences/csv/"
          "economist_loughran_mcdonald_sentiment.csv")
NBER_REL = "week three data cleaning/nber_recession_table.csv"
FLAGS_REL = "week 4/Ad_filtering/week_four_quality_flags.csv"


def is_pointer(path):
    """An unresolved Git LFS file is a 3-line stub, not real data."""
    try:
        with open(path) as f:
            return f.readline().startswith("version https://git-lfs")
    except (FileNotFoundError, UnicodeDecodeError):
        return True


# Setting up: MAKE SURE THE INPUTS ARE DOWNLOADED

DRIVE_DIR = "/content/drive/MyDrive/SOSC314_data"
drive_harvard = os.path.join(DRIVE_DIR, "week_three_ads_with_sentiment.csv")

if os.path.exists(drive_harvard):
    HARVARD_PATH = drive_harvard
    print("Using the Harvard file cached in Drive.")
else:
    HARVARD_PATH = os.path.join(REPO_DIR, HARVARD_REL)
    need = [r for r in (HARVARD_REL, NBER_REL)
            if is_pointer(os.path.join(REPO_DIR, r))]
    if need:
        print("Downloading:", need)
        subprocess.run(f'git lfs pull --include="{",".join(need)}"',
                       shell=True, cwd=REPO_DIR)

# The NBER table is tiny but might still be in LFS
if is_pointer(os.path.join(REPO_DIR, NBER_REL)):
    subprocess.run(f'git lfs pull --include="{NBER_REL}"',
                   shell=True, cwd=REPO_DIR)

# STEP 1: LOAD
# Only the columns we need. Both files are large, so reading all
# columns would be slow and pointless.
#
# Note the naming difference between the two dictionaries:
#   Harvard General Inquirer -> Positiv_Count / Negativ_Count
#   Loughran-McDonald        -> Positive_Count / Negative_Count

print("\nLoading Harvard ad-level data...")
harvard = pd.read_csv(
    HARVARD_PATH,
    usecols=["Year_Quarter", "Total_Words", "Positiv_Count", "Negativ_Count"],
)
harvard = harvard.rename(columns={
    "Positiv_Count": "Pos_Count",
    "Negativ_Count": "Neg_Count",
})

print("Loading Loughran-McDonald ad-level data...")
lm = pd.read_csv(
    os.path.join(REPO_DIR, LM_REL),
    usecols=["Year_Quarter", "Total_Words", "Positive_Count", "Negative_Count"],
)
lm = lm.rename(columns={
    "Positive_Count": "Pos_Count",
    "Negative_Count": "Neg_Count",
})

flags = pd.read_csv(
    os.path.join(REPO_DIR, FLAGS_REL),
    usecols=["row_id", "is_non_advertiser", "is_multipage", "too_short",
             "any_flag"],
)

nber = pd.read_csv(os.path.join(REPO_DIR, NBER_REL),
                   usecols=["Year_Quarter", "Recession_Status"])

print(f"\nHarvard rows: {len(harvard):,}")
print(f"LM rows:      {len(lm):,}")
print(f"Flags rows:   {len(flags):,}")

assert len(harvard) == len(lm) == len(flags), \
    "Row counts differ. The row_id join is not safe."


# STEP 2: ATTACH THE QUALITY FLAGS

# row_id is positional. Both sentiment files were written with
# index=False from the same filtered corpus, in the same order, which we
# verified at 100% brand agreement. We shall never sort these frames 

for df in (harvard, lm):
    df.reset_index(drop=True, inplace=True)
    df["row_id"] = df.index

harvard = harvard.merge(flags, on="row_id", how="left")
lm = lm.merge(flags, on="row_id", how="left")

print(f"\nAds flagged by any rule: {int(flags['any_flag'].sum()):,} "
      f"({100*flags['any_flag'].mean():.2f}%)")

# --------------------------------------------------------------
# STEP 3: THE TWO AGGREGATION METHODS
# --------------------------------------------------------------


def aggregate(ads, method):
    """Collapse ad-level counts to a quarterly sentiment series.

    macro  : average the per-ad ratios. Every ad counts equally,
             regardless of length.
    pooled : sum the words first, then divide once. Long ads carry
             proportionally more weight.
    """
    d = ads[ads["Total_Words"] > 0].copy()

    if method == "macro":
        d["Pos_Ratio"] = d["Pos_Count"] / d["Total_Words"]
        d["Neg_Ratio"] = d["Neg_Count"] / d["Total_Words"]
        out = (
            d.groupby("Year_Quarter")
            .agg(Pos=("Pos_Ratio", "mean"),
                 Neg=("Neg_Ratio", "mean"),
                 Ads=("Pos_Ratio", "size"))
            .reset_index()
        )
    else:
        g = (
            d.groupby("Year_Quarter")
            .agg(Pos_Words=("Pos_Count", "sum"),
                 Neg_Words=("Neg_Count", "sum"),
                 All_Words=("Total_Words", "sum"),
                 Ads=("Total_Words", "size"))
            .reset_index()
        )
        g["Pos"] = g["Pos_Words"] / g["All_Words"]
        g["Neg"] = g["Neg_Words"] / g["All_Words"]
        out = g[["Year_Quarter", "Pos", "Neg", "Ads"]]

    return out.merge(nber, on="Year_Quarter", how="inner")


def recession_gap(quarterly):
    """Descriptive difference: recession quarters minus expansion
    quarters, in percentage points. No significance test: quarterly
    series are serially correlated, and valid inference is Week 5 work.
    """
    g = quarterly.groupby("Recession_Status")[["Pos", "Neg"]].mean()
    return {
        "Pos_Recession": 100 * g.loc["Recession", "Pos"],
        "Pos_Expansion": 100 * g.loc["Not Recession", "Pos"],
        "Pos_Diff_pp": 100 * (g.loc["Recession", "Pos"]
                              - g.loc["Not Recession", "Pos"]),
        "Neg_Diff_pp": 100 * (g.loc["Recession", "Neg"]
                              - g.loc["Not Recession", "Neg"]),
        "Quarters": len(quarterly),
    }


# STEP 4: RUN ALL EIGHT COMBINATIONS

rows = []
series_store = {}

for dict_name, ads in [("Harvard", harvard), ("Loughran-McDonald", lm)]:
    for filtered in [False, True]:

        subset = ads[~ads["any_flag"].fillna(False)] if filtered else ads
        label_f = "Filtered" if filtered else "Full corpus"

        for method in ["macro", "pooled"]:
            q = aggregate(subset, method)
            series_store[(dict_name, label_f, method)] = q

            r = recession_gap(q)
            r["Dictionary"] = dict_name
            r["Corpus"] = label_f
            r["Aggregation"] = "Average of ratios" if method == "macro" \
                else "Pooled words"
            r["Ads_Used"] = len(subset)
            rows.append(r)

results = pd.DataFrame(rows)[[
    "Dictionary", "Corpus", "Aggregation", "Ads_Used", "Quarters",
    "Pos_Recession", "Pos_Expansion", "Pos_Diff_pp", "Neg_Diff_pp",
]].round(4)

print("\n" + "=" * 78)
print("RECESSION MINUS EXPANSION, BY DICTIONARY / CORPUS / AGGREGATION")
print("=" * 78)
print(results.to_string(index=False))

results.to_csv(f"{OUT_DIR}/aggregation_sensitivity_results.csv", index=False)


# STEP 5: READ THE TWO COMPARISONS

print("\n" + "-" * 78)
print("DOES THE AGGREGATION METHOD MATTER?")
print("-" * 78)
for d in ["Harvard", "Loughran-McDonald"]:
    sub = results[(results["Dictionary"] == d)
                  & (results["Corpus"] == "Full corpus")]
    macro = sub[sub["Aggregation"] == "Average of ratios"]["Pos_Diff_pp"].iloc[0]
    pooled = sub[sub["Aggregation"] == "Pooled words"]["Pos_Diff_pp"].iloc[0]
    flip = "SIGN FLIPS" if macro * pooled < 0 else "same sign"
    print(f"{d:<20} macro {macro:+.4f} pp | pooled {pooled:+.4f} pp | {flip}")

print("\n" + "-" * 78)
print("DOES THE QUALITY FILTER MATTER?")
print("-" * 78)
for d in ["Harvard", "Loughran-McDonald"]:
    for m in ["Average of ratios", "Pooled words"]:
        sub = results[(results["Dictionary"] == d)
                      & (results["Aggregation"] == m)]
        full = sub[sub["Corpus"] == "Full corpus"]["Pos_Diff_pp"].iloc[0]
        filt = sub[sub["Corpus"] == "Filtered"]["Pos_Diff_pp"].iloc[0]
        print(f"{d:<20} {m:<18} full {full:+.4f} | filtered {filt:+.4f} "
              f"| shift {filt-full:+.4f} pp")

# STEP 6: FIGURE
# Two panels, one per dictionary, each showing the quarterly positive
# series under both aggregation methods with recessions shaded. Separate
# panels because the two dictionaries sit on very different scales and a
# shared axis would hide the smaller one.

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, d in zip(axes, ["Harvard", "Loughran-McDonald"]):
    for method, style in [("macro", "-"), ("pooled", "--")]:
        q = series_store[(d, "Full corpus", method)].copy()
        q["t"] = pd.PeriodIndex(q["Year_Quarter"], freq="Q").to_timestamp()
        label = "Average of ratios" if method == "macro" else "Pooled words"
        ax.plot(q["t"], 100 * q["Pos"], style, linewidth=1, label=label)

    q = series_store[(d, "Full corpus", "macro")].copy()
    q["t"] = pd.PeriodIndex(q["Year_Quarter"], freq="Q").to_timestamp()
    for _, r in q[q["Recession_Status"] == "Recession"].iterrows():
        ax.axvspan(r["t"], r["t"] + pd.offsets.QuarterEnd(1),
                   alpha=0.2, color="grey")

    ax.set_title(f"{d}: positive sentiment by quarter, "
                 "recessions shaded")
    ax.set_ylabel("Positive words (% of total)")
    ax.legend(loc="upper left", fontsize=9)

axes[-1].set_xlabel("Quarter")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/aggregation_comparison.png", dpi=200)
plt.show()

# Step7: save and downlod

for (d, c, m), q in series_store.items():
    if c == "Full corpus":
        name = f"quarterly_{d.split('-')[0].lower()}_{m}.csv"
        q.to_csv(f"{OUT_DIR}/{name}", index=False)

import shutil
shutil.make_archive("/content/week_four_aggregation", "zip", OUT_DIR)

print("\nFiles ready:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)

try:
    from google.colab import files
    files.download("/content/week_four_aggregation.zip")
except Exception:
    print("\nDownload blocked. Use the folder icon in the left sidebar.")
