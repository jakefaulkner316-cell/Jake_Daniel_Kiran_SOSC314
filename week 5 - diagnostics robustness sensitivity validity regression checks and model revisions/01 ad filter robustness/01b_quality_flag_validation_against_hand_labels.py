# 01b - WEEK 5 DIAGNOSTIC:
# VALIDATING THE AUTOMATED QUALITY FLAGS AGAINST
# KIRAN'S HAND-LABELLED SAMPLE


# PURPOSE:
# Measure how well the Week 4 automated quality flags identify
# records that are not genuine advertisements, using Kiran's
# 200 hand-labelled records as the reference ("ground truth").
#
# MAIN DIAGNOSTIC QUESTION:
# When the flags remove a record, is it really a non-advertisement
# (precision)? And of the records a human reader judges to be
# non-advertisements, how many do the flags catch (recall)?
#
# WHY THIS DIAGNOSTIC MATTERS:
# Script 01 showed that removing the 3,303 flagged records barely
# changes the sentiment estimates. That is a ROBUSTNESS result: it
# says the flags do not drive the findings. It does not say the
# flags remove the RIGHT records. This script is the VALIDITY
# check: it compares the automated flags with human judgement.
#
# If recall is low, many non-advertisements (house notices, legal
# and financial notices, unreadable OCR) stay in the corpus after
# filtering, and the sentiment measure partly reflects text that
# is not advertising copy. That is a measurement-validity concern
# to report, not something to hide.
#
# THE HAND-LABELLED SAMPLE (Kiran, end of Week 4):
#   - 200 records drawn at random from the first 2,000 rows of
#     week_three_ads_with_sentiment.csv
#   - label: 1 = advertisement, 0 = non-advertisement,
#     with a short written reason for each non-advertisement
#   - Kiran's ID runs from 1 upward in the original row order.
#     The group's join key, row_id, is the 0-based position in the
#     same file. The script tests both alignments (ID - 1 and ID)
#     against Brand / date when those columns are present and uses
#     whichever one matches.
#
# IMPORTANT SAMPLE LIMITATION:
# The corpus file is stored in REVERSE date order (row 0 is
# 2014-12-20; the last row is 1948-01-03). The first 2,000 rows
# therefore cover roughly mid-2012 to end-2014 ONLY. Every
# precision/recall figure below describes the flags' behaviour in
# 2012-2014 advertising and cannot be generalised to the full
# 1948-2014 corpus. The script prints the actual date range of the
# labelled records so the report can state it exactly.
#
# FLAGS BEING VALIDATED (week_four_quality_flags.csv):
#   is_non_advertiser : brand on the hand-built blocklist
#   is_multipage      : 2+ page headers (multi-page supplements)
#   too_short         : fewer than the minimum number of words
#   any_flag          : any of the three
#
# FILES LOADED:
#   week 4 - updated quality filtered analysis/Ad_filtering/
#       week_four_quality_flags.csv        (plain text, in repo)
#   week 5 - .../01 ad filter robustness/hand labelled sample/
#       kiran_200_hand_labelled_ads.csv    (Kiran's sheet, exported
#                                           as CSV and committed)
#
# OUTPUTS (in this script's folder):
#   csv outputs/
#       hand_label_vs_flag_merged_records.csv
#           one row per labelled record: label, reason, every flag,
#           and whether flag and label agree
#       hand_label_vs_flag_confusion_matrix.csv
#       hand_label_vs_flag_validation_metrics.csv
#           precision, recall, specificity, agreement, with 95%
#           Wilson intervals, for any_flag and for each flag
#   diagnostic images/
#       quality_flags_vs_hand_labels.png
#   robustness tables/
#       quality_flag_validation_summary.txt
#
# HOW TO RUN:
#   1. Export Kiran's Google Sheet as CSV (File > Download > .csv)
#   2. Save it as:
#        01 ad filter robustness/hand labelled sample/
#        kiran_200_hand_labelled_ads.csv
#   3. If her column names differ from what is auto-detected below,
#      set LABEL_COLUMN / ID_COLUMN / REASON_COLUMN by hand.
#   4. Run this script from anywhere (paths are anchored to the
#      script's own location).
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# DEFINE FILE LOCATIONS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WEEK5_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEEK5_DIR.parent

FLAGS_FILE = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
    / "Ad_filtering"
    / "week_four_quality_flags.csv"
)

LABELS_FILE = (
    SCRIPT_DIR
    / "hand labelled sample"
    / "kiran_200_hand_labelled_ads.csv"
)


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = SCRIPT_DIR / "csv outputs"
IMAGE_OUTPUT_DIR = SCRIPT_DIR / "diagnostic images"
TABLE_OUTPUT_DIR = SCRIPT_DIR / "robustness tables"

for folder in [CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# COLUMN SETTINGS FOR KIRAN'S FILE
# ---------------------------------------------------------
# Leave as None to auto-detect. Set a column name explicitly if
# auto-detection picks the wrong column.
# ---------------------------------------------------------

ID_COLUMN = None       # her 1-based record ID
LABEL_COLUMN = None    # 1 = advertisement, 0 = non-advertisement
REASON_COLUMN = None   # free-text explanation (optional)
BRAND_COLUMN = None    # optional, used to verify the join
DATE_COLUMN = None     # optional, used to verify the join

FLAG_COLUMNS = ["is_non_advertiser", "is_multipage", "too_short", "any_flag"]

FLAG_LABELS = {
    "any_flag": "Any flag",
    "is_non_advertiser": "Non-advertiser blocklist",
    "is_multipage": "Multi-page record",
    "too_short": "Too short",
}


# ---------------------------------------------------------
# CHECK INPUT FILES
# ---------------------------------------------------------

def check_input_file(path, how_to_fix=""):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file:\n  {path}\n{how_to_fix}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.startswith("version https://git-lfs"):
        raise RuntimeError(
            f"This file is a Git LFS pointer, not the real data:\n  {path}\n"
            f"Run: git lfs pull --include=\"{path.relative_to(REPO_ROOT)}\""
        )


check_input_file(FLAGS_FILE)
check_input_file(
    LABELS_FILE,
    "Export Kiran's Google Sheet as CSV and save it at the path above.",
)


# ---------------------------------------------------------
# LOAD FLAGS
# ---------------------------------------------------------

flags = pd.read_csv(FLAGS_FILE)

for column in FLAG_COLUMNS:
    flags[column] = flags[column].astype(str).str.strip().str.lower() == "true"

if not (flags["row_id"].values == np.arange(len(flags))).all():
    raise ValueError("row_id in the flags file is not the 0-based row position.")

print("\n" + "=" * 90)
print("AUTOMATED QUALITY FLAGS")
print("=" * 90)
print(f"Records: {len(flags):,}")
print(f"Any flag: {int(flags['any_flag'].sum()):,} ({100 * flags['any_flag'].mean():.2f}%)")
print(f"Row 0 date: {flags['DateOfIssue'].iloc[0]} | last row date: {flags['DateOfIssue'].iloc[-1]}")

first_2000 = flags.iloc[:2000]
print(f"First 2,000 rows (Kiran's sampling frame): "
      f"{first_2000['DateOfIssue'].min()} to {first_2000['DateOfIssue'].max()}, "
      f"{int(first_2000['any_flag'].sum())} flagged "
      f"({100 * first_2000['any_flag'].mean():.2f}%)")


# ---------------------------------------------------------
# LOAD KIRAN'S LABELS AND DETECT COLUMNS
# ---------------------------------------------------------

labels_raw = pd.read_csv(LABELS_FILE)
labels_raw.columns = [str(c).strip() for c in labels_raw.columns]

print("\n" + "=" * 90)
print("HAND-LABELLED SAMPLE")
print("=" * 90)
print(f"Rows: {len(labels_raw)}")
print(f"Columns found: {list(labels_raw.columns)}")


def find_column(frame, explicit, keywords, must_be_numeric=False, exclude=()):
    """Return the explicit column if set, else the first column whose
    lower-case name contains one of the keywords."""
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(f"Column {explicit!r} not found in labels file.")
        return explicit
    for keyword in keywords:
        for column in frame.columns:
            if column in exclude:
                continue
            if keyword in column.lower():
                if must_be_numeric and pd.to_numeric(frame[column], errors="coerce").isna().all():
                    continue
                return column
    return None


id_col = find_column(labels_raw, ID_COLUMN, ["row_id", "id", "index", "row"], must_be_numeric=True)
label_col = find_column(labels_raw, LABEL_COLUMN,
                        ["label", "is_ad", "advertisement", "classification", "class", "ad"],
                        must_be_numeric=True, exclude=(id_col,))
reason_col = find_column(labels_raw, REASON_COLUMN,
                         ["reason", "explanation", "note", "comment", "why"],
                         exclude=(id_col, label_col))
brand_col = find_column(labels_raw, BRAND_COLUMN, ["brand"], exclude=(id_col, label_col))
date_col = find_column(labels_raw, DATE_COLUMN, ["date"], exclude=(id_col, label_col))

print(f"\nUsing columns -> ID: {id_col!r} | label: {label_col!r} | reason: {reason_col!r} | "
      f"brand: {brand_col!r} | date: {date_col!r}")

if id_col is None or label_col is None:
    raise KeyError(
        "Could not detect the ID and/or label column. Set ID_COLUMN and "
        "LABEL_COLUMN at the top of this script."
    )

labels = pd.DataFrame({
    "Kiran_ID": pd.to_numeric(labels_raw[id_col], errors="coerce"),
    "Hand_Label": pd.to_numeric(labels_raw[label_col], errors="coerce"),
    "Reason": labels_raw[reason_col].astype("string") if reason_col else pd.NA,
    "Kiran_Brand": labels_raw[brand_col].astype("string") if brand_col else pd.NA,
    "Kiran_Date": labels_raw[date_col].astype("string") if date_col else pd.NA,
})

n_before = len(labels)
labels = labels.dropna(subset=["Kiran_ID", "Hand_Label"]).copy()
if len(labels) < n_before:
    print(f"Dropped {n_before - len(labels)} rows with a missing ID or label.")

labels["Kiran_ID"] = labels["Kiran_ID"].astype(int)
labels["Hand_Label"] = labels["Hand_Label"].astype(int)

bad_labels = sorted(set(labels["Hand_Label"]) - {0, 1})
if bad_labels:
    raise ValueError(f"Labels must be 0 or 1; found {bad_labels}.")

if labels["Kiran_ID"].duplicated().any():
    raise ValueError("Duplicate IDs in the labelled sample.")

# The positive class for validation is "non-advertisement".
labels["Is_Non_Ad"] = labels["Hand_Label"] == 0

print(f"Advertisements (1):     {int((~labels['Is_Non_Ad']).sum())}")
print(f"Non-advertisements (0): {int(labels['Is_Non_Ad'].sum())}")
print(f"ID range: {labels['Kiran_ID'].min()} to {labels['Kiran_ID'].max()}")


# ---------------------------------------------------------
# ALIGN KIRAN'S IDS WITH row_id
# ---------------------------------------------------------
# Her IDs start at 1; row_id starts at 0. The expected mapping is
# row_id = ID - 1. When Brand or date columns are present, both
# candidate offsets are tested and the better match is used.
# ---------------------------------------------------------

def normalise_text(series):
    return series.astype("string").str.strip().str.lower().fillna("").to_numpy(dtype=object)


def match_rate(offset):
    candidate_ids = labels["Kiran_ID"] + offset
    valid = candidate_ids.between(0, len(flags) - 1)
    if not valid.any():
        return 0.0, 0
    matched = flags.set_index("row_id").loc[candidate_ids[valid]]
    checks = []
    if brand_col:
        checks.append(
            normalise_text(labels.loc[valid, "Kiran_Brand"])
            == normalise_text(matched["Brand"])
        )
    if date_col:
        kiran_dates = pd.to_datetime(labels.loc[valid, "Kiran_Date"], errors="coerce")
        flag_dates = pd.to_datetime(matched["DateOfIssue"], errors="coerce")
        checks.append(
            kiran_dates.dt.strftime("%Y-%m-%d").fillna("x").to_numpy(dtype=object)
            == flag_dates.dt.strftime("%Y-%m-%d").fillna("y").to_numpy(dtype=object)
        )
    if not checks:
        return np.nan, int(valid.sum())
    agree = np.logical_and.reduce(checks)
    return float(np.mean(agree)), int(valid.sum())


print("\n" + "=" * 90)
print("ALIGNING KIRAN'S ID WITH row_id")
print("=" * 90)

offset_results = {offset: match_rate(offset) for offset in [-1, 0]}
for offset, (rate, n_valid) in offset_results.items():
    rate_text = "no Brand/date column to check" if np.isnan(rate) else f"{100 * rate:.1f}% of records match"
    print(f"row_id = ID {offset:+d}: {n_valid} IDs in range, {rate_text}")

rates = {k: v[0] for k, v in offset_results.items()}
if all(np.isnan(r) for r in rates.values()):
    OFFSET = -1
    alignment_note = ("Alignment ASSUMED (row_id = ID - 1): the labels file has no Brand or date "
                      "column to verify it.")
else:
    OFFSET = max(rates, key=lambda k: -1 if np.isnan(rates[k]) else rates[k])
    alignment_note = (f"Alignment VERIFIED: row_id = ID {OFFSET:+d} "
                      f"({100 * rates[OFFSET]:.1f}% Brand/date match).")
    if rates[OFFSET] < 0.95:
        alignment_note += " WARNING: match rate below 95%; check the join before reporting."

print(alignment_note)

labels["row_id"] = labels["Kiran_ID"] + OFFSET


# ---------------------------------------------------------
# MERGE LABELS WITH FLAGS
# ---------------------------------------------------------

merged = labels.merge(
    flags[["row_id", "DateOfIssue", "Brand", "token_count"] + FLAG_COLUMNS],
    on="row_id",
    how="left",
    validate="one_to_one",
)

unmatched = merged["any_flag"].isna().sum()
if unmatched:
    print(f"WARNING: {unmatched} labelled records have no matching row_id and are dropped.")
    merged = merged.dropna(subset=["any_flag"]).copy()

for column in FLAG_COLUMNS:
    merged[column] = merged[column].astype(bool)

merged["Agreement_any_flag"] = merged["any_flag"] == merged["Is_Non_Ad"]
merged["Outcome_any_flag"] = np.select(
    [
        merged["any_flag"] & merged["Is_Non_Ad"],
        merged["any_flag"] & ~merged["Is_Non_Ad"],
        ~merged["any_flag"] & merged["Is_Non_Ad"],
    ],
    ["Correctly removed (non-ad, flagged)",
     "Wrongly removed (ad, flagged)",
     "Missed (non-ad, not flagged)"],
    default="Correctly kept (ad, not flagged)",
)

sample_start = merged["DateOfIssue"].min()
sample_end = merged["DateOfIssue"].max()
n = len(merged)

print(f"\nMerged records: {n}")
print(f"Date range of the labelled records: {sample_start} to {sample_end}")
print(f"Median word count of labelled records: {merged['token_count'].median():.0f}")


# ---------------------------------------------------------
# METRICS WITH 95% WILSON INTERVALS
# ---------------------------------------------------------
# With 200 records, and only a handful flagged, every rate is
# uncertain. Wilson intervals behave sensibly for small counts
# and for rates near 0 or 1.
# ---------------------------------------------------------

def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan, np.nan
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


metric_rows = []
confusion_rows = []

for flag in ["any_flag", "is_non_advertiser", "is_multipage", "too_short"]:
    flagged = merged[flag]
    non_ad = merged["Is_Non_Ad"]

    tp = int((flagged & non_ad).sum())     # non-ad, flagged
    fp = int((flagged & ~non_ad).sum())    # ad, flagged
    fn = int((~flagged & non_ad).sum())    # non-ad, not flagged
    tn = int((~flagged & ~non_ad).sum())   # ad, not flagged

    confusion_rows.append({
        "Flag": FLAG_LABELS[flag],
        "NonAd_flagged (true positive)": tp,
        "Ad_flagged (false positive)": fp,
        "NonAd_not_flagged (false negative)": fn,
        "Ad_not_flagged (true negative)": tn,
    })

    metrics = {
        "Precision": (tp, tp + fp,
                      "Of the records the flag removes, share that are non-ads"),
        "Recall": (tp, tp + fn,
                   "Of the non-ads, share that the flag catches"),
        "Specificity": (tn, tn + fp,
                        "Of the genuine ads, share that the flag keeps"),
        "Agreement": (tp + tn, n,
                      "Share of records where flag and hand label agree"),
        "NonAd_share_left_after_filtering": (fn, fn + tn,
                                             "Of the records the flag keeps, share that are non-ads"),
    }

    for metric_name, (k, total, meaning) in metrics.items():
        value, lo, hi = wilson_interval(k, total)
        metric_rows.append({
            "Flag": FLAG_LABELS[flag],
            "Metric": metric_name,
            "Count": k,
            "Denominator": total,
            "Value": value,
            "CI95_low": lo,
            "CI95_high": hi,
            "Meaning": meaning,
        })

confusion = pd.DataFrame(confusion_rows)
metrics_table = pd.DataFrame(metric_rows)

# Cohen's kappa for any_flag: agreement beyond what chance would give.
tp, fp, fn, tn = confusion.iloc[0, 1:].astype(int).tolist()
p_observed = (tp + tn) / n
p_expected = (((tp + fp) / n) * ((tp + fn) / n)) + (((fn + tn) / n) * ((fp + tn) / n))
kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else np.nan

base_rate, base_lo, base_hi = wilson_interval(int(merged["Is_Non_Ad"].sum()), n)

print("\n" + "=" * 90)
print("CONFUSION MATRIX (positive class = non-advertisement)")
print("=" * 90)
print(confusion.to_string(index=False))

print("\n" + "=" * 90)
print("VALIDATION METRICS (95% Wilson intervals)")
print("=" * 90)
with pd.option_context("display.width", 200):
    shown = metrics_table[metrics_table["Flag"] == "Any flag"].copy()
    for _, r in shown.iterrows():
        v = "n/a" if pd.isna(r["Value"]) else f"{100 * r['Value']:.1f}%"
        ci = "" if pd.isna(r["Value"]) else f" [{100 * r['CI95_low']:.1f}%, {100 * r['CI95_high']:.1f}%]"
        print(f"{r['Metric']:<34} {r['Count']:>3}/{r['Denominator']:<3} = {v}{ci}")
print(f"{'Cohen kappa':<34} {kappa:.3f}")
print(f"{'Non-ad share in sample (base rate)':<34} {100 * base_rate:.1f}% "
      f"[{100 * base_lo:.1f}%, {100 * base_hi:.1f}%]")


# ---------------------------------------------------------
# WHY WERE NON-ADS MISSED?
# ---------------------------------------------------------

reason_table = pd.DataFrame()
if reason_col:
    reason_table = (
        merged.loc[merged["Is_Non_Ad"]]
        .assign(Reason_clean=lambda d: d["Reason"].fillna("(no reason given)")
                .str.strip().str.lower())
        .groupby("Reason_clean")
        .agg(Non_ads=("row_id", "size"), Caught_by_flags=("any_flag", "sum"))
        .sort_values("Non_ads", ascending=False)
        .reset_index()
        .rename(columns={"Reason_clean": "Reason"})
    )
    reason_table["Missed"] = reason_table["Non_ads"] - reason_table["Caught_by_flags"]

    print("\n" + "=" * 90)
    print("NON-ADVERTISEMENTS BY KIRAN'S REASON")
    print("=" * 90)
    print(reason_table.head(20).to_string(index=False))


# ---------------------------------------------------------
# SAVE CSV OUTPUTS
# ---------------------------------------------------------

merged_out = merged[[
    "Kiran_ID", "row_id", "DateOfIssue", "Brand", "token_count",
    "Hand_Label", "Is_Non_Ad", "Reason",
    "is_non_advertiser", "is_multipage", "too_short", "any_flag",
    "Agreement_any_flag", "Outcome_any_flag",
]].sort_values("row_id")

merged_path = CSV_OUTPUT_DIR / "hand_label_vs_flag_merged_records.csv"
confusion_path = CSV_OUTPUT_DIR / "hand_label_vs_flag_confusion_matrix.csv"
metrics_path = CSV_OUTPUT_DIR / "hand_label_vs_flag_validation_metrics.csv"

merged_out.to_csv(merged_path, index=False)
confusion.to_csv(confusion_path, index=False)
metrics_table.round(4).to_csv(metrics_path, index=False)


# ---------------------------------------------------------
# DIAGNOSTIC FIGURE
# ---------------------------------------------------------
# Left: what happens to Kiran's non-ads and to her genuine ads
#       under the automated flags (counts).
# Right: Kiran's non-ads by reason, split into caught vs missed
#        (only drawn when a reason column exists).
# ---------------------------------------------------------

CAUGHT = "#2a78d6"
MISSED = "#eb6834"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})

has_reasons = not reason_table.empty
fig, axes = plt.subplots(1, 2 if has_reasons else 1,
                         figsize=(11, 4.2) if has_reasons else (6, 3.6),
                         gridspec_kw={"width_ratios": [1, 1.4]} if has_reasons else None)
axes = np.atleast_1d(axes)

ax = axes[0]
groups = ["Hand-labelled\nnon-advertisements", "Hand-labelled\nadvertisements"]
flagged_counts = [tp, fp]
kept_counts = [fn, tn]
y = np.arange(len(groups))[::-1]

ax.barh(y, flagged_counts, color=CAUGHT, height=0.55, label="Flagged (removed)",
        edgecolor="white", linewidth=2)
ax.barh(y, kept_counts, left=flagged_counts, color=MISSED, height=0.55,
        label="Not flagged (kept)", edgecolor="white", linewidth=2)

for yy, f, k in zip(y, flagged_counts, kept_counts):
    total = f + k
    ax.text(total + max(flagged_counts[0] + kept_counts[0], tn + fp) * 0.01, yy,
            f"{f} of {total} flagged", va="center", color="#52514e", fontsize=8.5)

ax.set_yticks(y)
ax.set_yticklabels(groups)
ax.set_xlabel("Records in the hand-labelled sample")
ax.set_xlim(0, max(tp + fn, fp + tn) * 1.3)
ax.set_title("Automated flags vs hand labels", loc="left", fontsize=10, color="#0b0b0b")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

if has_reasons:
    ax = axes[1]
    top = reason_table.head(8).iloc[::-1]
    yr = np.arange(len(top))
    ax.barh(yr, top["Caught_by_flags"], color=CAUGHT, height=0.55,
            edgecolor="white", linewidth=2, label="Caught by flags")
    ax.barh(yr, top["Missed"], left=top["Caught_by_flags"], color=MISSED, height=0.55,
            edgecolor="white", linewidth=2, label="Missed by flags")
    ax.set_yticks(yr)
    ax.set_yticklabels([r if len(r) <= 45 else r[:42] + "..." for r in top["Reason"]])
    ax.set_xlabel("Hand-labelled non-advertisements")
    ax.set_title("Non-advertisements by Kiran's reason", loc="left", fontsize=10, color="#0b0b0b")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.suptitle(
    f"Validating the Week 4 quality flags against {n} hand-labelled records "
    f"({sample_start[:4]}-{sample_end[:4]} issues only)",
    y=1.02, fontsize=11,
)
fig.tight_layout()

figure_path = IMAGE_OUTPUT_DIR / "quality_flags_vs_hand_labels.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# WRITTEN SUMMARY
# ---------------------------------------------------------

def pct(metric, flag="Any flag"):
    r = metrics_table[(metrics_table["Flag"] == flag) & (metrics_table["Metric"] == metric)].iloc[0]
    if pd.isna(r["Value"]):
        return f"{r['Count']}/{r['Denominator']} (undefined)"
    return (f"{r['Count']}/{r['Denominator']} = {100 * r['Value']:.1f}% "
            f"[95% CI {100 * r['CI95_low']:.1f}-{100 * r['CI95_high']:.1f}%]")


lines = []
lines.append("WEEK 5 DIAGNOSTIC 01B")
lines.append("VALIDATING THE AUTOMATED QUALITY FLAGS AGAINST HAND LABELS")
lines.append("=" * 78)
lines.append("")
lines.append(f"Reference: {n} records hand-labelled by Kiran (1 = ad, 0 = non-ad).")
lines.append(f"Labelled records cover issues from {sample_start} to {sample_end} ONLY.")
lines.append("The corpus file is in reverse date order, so 'the first 2,000 rows' are")
lines.append("the most recent issues, not the earliest.")
lines.append(alignment_note)
lines.append("")
lines.append(f"Non-ad share in the sample (base rate): {int(merged['Is_Non_Ad'].sum())}/{n} = "
             f"{100 * base_rate:.1f}% [95% CI {100 * base_lo:.1f}-{100 * base_hi:.1f}%]")
lines.append(f"Flagged share in the sample: {tp + fp}/{n}")
lines.append(f"Flagged share in the full corpus: {int(flags['any_flag'].sum()):,}/{len(flags):,} = "
             f"{100 * flags['any_flag'].mean():.2f}%")
lines.append("")
lines.append("-" * 78)
lines.append("ANY FLAG vs HAND LABEL (positive class = non-advertisement)")
lines.append("-" * 78)
lines.append(f"Non-ad, flagged (correctly removed): {tp}")
lines.append(f"Ad, flagged (wrongly removed):       {fp}")
lines.append(f"Non-ad, not flagged (missed):        {fn}")
lines.append(f"Ad, not flagged (correctly kept):    {tn}")
lines.append("")
lines.append(f"Precision:   {pct('Precision')}")
lines.append(f"Recall:      {pct('Recall')}")
lines.append(f"Specificity: {pct('Specificity')}")
lines.append(f"Agreement:   {pct('Agreement')}")
lines.append(f"Cohen's kappa: {kappa:.3f}")
lines.append(f"Non-ads left after filtering: {pct('NonAd_share_left_after_filtering')}")
lines.append("")
lines.append("-" * 78)
lines.append("BY INDIVIDUAL FLAG")
lines.append("-" * 78)
lines.append(confusion.to_string(index=False))
lines.append("")

if has_reasons:
    lines.append("-" * 78)
    lines.append("NON-ADVERTISEMENTS BY KIRAN'S REASON (caught vs missed by any flag)")
    lines.append("-" * 78)
    lines.append(reason_table.to_string(index=False))
    lines.append("")

lines.append("-" * 78)
lines.append("HOW TO READ THIS")
lines.append("-" * 78)
lines.append("- Precision answers: when the filter removes something, is it junk?")
lines.append("  Recall answers: how much of the junk does the filter find?")
lines.append("- The flags were designed to be conservative (they remove 1.92% of the")
lines.append("  corpus). Low recall with high precision is therefore the expected")
lines.append("  pattern: the filter removes little, and what it removes is mostly")
lines.append("  correct, but many non-advertisements remain in the corpus.")
lines.append("- With only a handful of flagged records in the sample, precision is")
lines.append("  estimated from very few cases; read its interval, not the point value.")
lines.append("- Kappa near 0 means the flags agree with the human labels little more")
lines.append("  than chance would predict, even if raw agreement looks high (raw")
lines.append("  agreement is inflated because most records are genuine ads).")
lines.append("- The sample covers only the most recent issues, so none of these")
lines.append("  figures can be assumed to hold for earlier decades, where OCR quality")
lines.append("  and the mix of notices and advertorials differ.")
lines.append("- Script 01 showed the flags barely move the sentiment estimates. Read")
lines.append("  together: the filter is not driving the results, but it also does not")
lines.append("  clean the corpus of most non-advertising text, so the sentiment")
lines.append("  measure still partly reflects non-advertising language.")

summary_path = TABLE_OUTPUT_DIR / "quality_flag_validation_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")


print("\n" + "=" * 90)
print("QUALITY FLAG VALIDATION COMPLETE")
print("=" * 90)
for path in [merged_path, confusion_path, metrics_path, figure_path, summary_path]:
    print(f"  {path.relative_to(REPO_ROOT)}")
