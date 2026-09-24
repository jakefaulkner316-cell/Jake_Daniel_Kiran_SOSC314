# =========================================================
# WEEK 5 DIAGNOSTIC 2:
# SENTIMENT DICTIONARY AND PREPROCESSING SENSITIVITY
# =========================================================
#
# PURPOSE:
# This script tests whether the recession-related sentiment
# findings depend on which sentiment dictionary and which
# preprocessing method are used.
#
# DICTIONARIES COMPARED:
# - Harvard General Inquirer
# - Loughran-McDonald
#
# PREPROCESSING METHODS COMPARED:
# - All Words
# - No Stopwords
# - Dictionary Only
#
# MAIN QUESTION:
# Do the direction and magnitude of the recession-related
# positive and negative sentiment effects remain stable across
# reasonable measurement choices?
#
# WHY THIS MATTERS:
# If the main result remains similar across dictionaries and
# preprocessing choices, that strengthens the robustness of
# the finding.
#
# If the result changes substantially, that means the finding
# is sensitive to how sentiment is operationalized.
#
# OUTPUTS:
# - CSV containing all dictionary/preprocessing comparisons
# - CSV summarizing the sensitivity range within each dictionary
# - Positive sentiment sensitivity graph
# - Negative sentiment sensitivity graph
# - Human-readable summary for the Week 5 report and oral check
#
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ---------------------------------------------------------
# DEFINE FILE LOCATIONS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

WEEK5_DIR = SCRIPT_DIR.parent

REPO_ROOT = WEEK5_DIR.parent

WEEK4_DIR = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
)

COMPARISON_FILE = (
    WEEK4_DIR
    / "second dict and data differences"
    / "csv"
    / "harvard_vs_loughran_comparison.csv"
)


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = (
    SCRIPT_DIR
    / "csv outputs"
)

IMAGE_OUTPUT_DIR = (
    SCRIPT_DIR
    / "diagnostic images"
)

TABLE_OUTPUT_DIR = (
    SCRIPT_DIR
    / "robustness tables"
)


CSV_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD WEEK 4 COMPARISON RESULTS
# ---------------------------------------------------------

results = pd.read_csv(
    COMPARISON_FILE
)


print("\n" + "=" * 80)

print(
    "WEEK 4 DICTIONARY AND PREPROCESSING COMPARISON"
)

print("=" * 80)

print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# VERIFY REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "Dictionary",
    "Preprocessing",
    "Positive Difference (pp)",
    "Negative Difference (pp)"
]


missing_columns = [
    column
    for column in required_columns
    if column not in results.columns
]


if missing_columns:

    raise ValueError(
        "The comparison file is missing these columns: "
        + ", ".join(missing_columns)
    )


# ---------------------------------------------------------
# RENAME COLUMNS FOR EASIER CODING
# ---------------------------------------------------------

results = results.rename(
    columns={
        "Positive Difference (pp)":
            "Positive_Diff_pp",

        "Negative Difference (pp)":
            "Negative_Diff_pp"
    }
)


# ---------------------------------------------------------
# ADD DIRECTION LABELS
# ---------------------------------------------------------

results[
    "Positive_Direction"
] = results[
    "Positive_Diff_pp"
].apply(
    lambda x:
    "Lower during recession"
    if x < 0
    else "Higher during recession"
)


results[
    "Negative_Direction"
] = results[
    "Negative_Diff_pp"
].apply(
    lambda x:
    "Higher during recession"
    if x > 0
    else "Lower during recession"
)


# ---------------------------------------------------------
# ADD ABSOLUTE EFFECT SIZE COLUMNS
# ---------------------------------------------------------

results[
    "Positive_Absolute_pp"
] = (
    results[
        "Positive_Diff_pp"
    ]
    .abs()
)


results[
    "Negative_Absolute_pp"
] = (
    results[
        "Negative_Diff_pp"
    ]
    .abs()
)


# ---------------------------------------------------------
# CHECK DIRECTIONAL ROBUSTNESS
# ---------------------------------------------------------

positive_all_negative = (
    results[
        "Positive_Diff_pp"
    ] < 0
).all()


negative_all_positive = (
    results[
        "Negative_Diff_pp"
    ] > 0
).all()


print("\n" + "=" * 80)

print(
    "DIRECTIONAL ROBUSTNESS CHECK"
)

print("=" * 80)

print(
    "Positive sentiment is lower during recessions "
    "for every specification:",
    positive_all_negative
)

print(
    "Negative sentiment is higher during recessions "
    "for every specification:",
    negative_all_positive
)


# ---------------------------------------------------------
# CALCULATE SENSITIVITY RANGE WITHIN EACH DICTIONARY
# ---------------------------------------------------------
#
# This measures how much the estimated recession effect
# changes when preprocessing changes.
# ---------------------------------------------------------

sensitivity_ranges = (
    results
    .groupby(
        "Dictionary"
    )
    .agg(
        Positive_Min=(
            "Positive_Diff_pp",
            "min"
        ),

        Positive_Max=(
            "Positive_Diff_pp",
            "max"
        ),

        Negative_Min=(
            "Negative_Diff_pp",
            "min"
        ),

        Negative_Max=(
            "Negative_Diff_pp",
            "max"
        )
    )
    .reset_index()
)


sensitivity_ranges[
    "Positive_Range_pp"
] = (
    sensitivity_ranges[
        "Positive_Max"
    ]
    -
    sensitivity_ranges[
        "Positive_Min"
    ]
)


sensitivity_ranges[
    "Negative_Range_pp"
] = (
    sensitivity_ranges[
        "Negative_Max"
    ]
    -
    sensitivity_ranges[
        "Negative_Min"
    ]
)


print("\n" + "=" * 80)

print(
    "SENSITIVITY RANGE WITHIN EACH DICTIONARY"
)

print("=" * 80)

print(
    sensitivity_ranges.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE FULL SENSITIVITY RESULTS
# ---------------------------------------------------------

results_output = (
    CSV_OUTPUT_DIR
    / "preprocessing_and_dictionary_sensitivity_results.csv"
)


results.to_csv(
    results_output,
    index=False
)


print(
    "\nSAVED:"
)

print(
    results_output
)


# ---------------------------------------------------------
# SAVE SENSITIVITY RANGE RESULTS
# ---------------------------------------------------------

range_output = (
    CSV_OUTPUT_DIR
    / "preprocessing_sensitivity_ranges.csv"
)


sensitivity_ranges.to_csv(
    range_output,
    index=False
)


print(
    "\nSAVED:"
)

print(
    range_output
)


# =========================================================
# POSITIVE SENTIMENT SENSITIVITY GRAPH
# =========================================================

positive_plot = (
    results.copy()
)


positive_plot[
    "Method"
] = (
    positive_plot[
        "Dictionary"
    ]
    + "\n"
    + positive_plot[
        "Preprocessing"
    ]
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


ax.bar(
    positive_plot[
        "Method"
    ],
    positive_plot[
        "Positive_Diff_pp"
    ]
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_title(
    "Sensitivity of Recession-Related Positive Sentiment "
    "to Dictionary and Preprocessing Choice"
)


ax.set_xlabel(
    "Dictionary and Preprocessing Method"
)


ax.set_ylabel(
    "Recession Minus Non-Recession Difference "
    "(Percentage Points)"
)


plt.xticks(
    rotation=25,
    ha="right"
)


plt.tight_layout()


positive_image = (
    IMAGE_OUTPUT_DIR
    / "positive_sentiment_sensitivity_comparison.png"
)


plt.savefig(
    positive_image,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    positive_image
)


# =========================================================
# NEGATIVE SENTIMENT SENSITIVITY GRAPH
# =========================================================

negative_plot = (
    results.copy()
)


negative_plot[
    "Method"
] = (
    negative_plot[
        "Dictionary"
    ]
    + "\n"
    + negative_plot[
        "Preprocessing"
    ]
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


ax.bar(
    negative_plot[
        "Method"
    ],
    negative_plot[
        "Negative_Diff_pp"
    ]
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_title(
    "Sensitivity of Recession-Related Negative Sentiment "
    "to Dictionary and Preprocessing Choice"
)


ax.set_xlabel(
    "Dictionary and Preprocessing Method"
)


ax.set_ylabel(
    "Recession Minus Non-Recession Difference "
    "(Percentage Points)"
)


plt.xticks(
    rotation=25,
    ha="right"
)


plt.tight_layout()


negative_image = (
    IMAGE_OUTPUT_DIR
    / "negative_sentiment_sensitivity_comparison.png"
)


plt.savefig(
    negative_image,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    negative_image
)


# =========================================================
# CREATE HUMAN-READABLE SUMMARY
# =========================================================

summary_file = (
    TABLE_OUTPUT_DIR
    / "preprocessing_sensitivity_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 DICTIONARY AND PREPROCESSING "
        "SENSITIVITY DIAGNOSTIC\n"
    )

    file.write(
        "=" * 65
        + "\n\n"
    )

    file.write(
        "DIAGNOSTIC QUESTION\n"
    )

    file.write(
        "-" * 65
        + "\n"
    )

    file.write(
        "Do the recession-related sentiment findings remain "
        "similar when the sentiment dictionary and preprocessing "
        "method change?\n\n"
    )


    file.write(
        "DIRECTIONAL ROBUSTNESS\n"
    )

    file.write(
        "-" * 65
        + "\n"
    )

    file.write(
        "Positive sentiment is lower during recessions under "
        f"every tested specification: {positive_all_negative}\n\n"
    )

    file.write(
        "Negative sentiment is higher during recessions under "
        f"every tested specification: {negative_all_positive}\n\n"
    )


    file.write(
        "SPECIFICATION RESULTS\n"
    )

    file.write(
        "-" * 65
        + "\n\n"
    )


    for _, row in results.iterrows():

        file.write(
            f"Dictionary: {row['Dictionary']}\n"
        )

        file.write(
            f"Preprocessing: {row['Preprocessing']}\n"
        )

        file.write(
            "Positive recession difference: "
            f"{row['Positive_Diff_pp']:+.4f} pp\n"
        )

        file.write(
            "Negative recession difference: "
            f"{row['Negative_Diff_pp']:+.4f} pp\n"
        )

        file.write(
            "\n"
        )


    file.write(
        "INTERPRETATION\n"
    )

    file.write(
        "-" * 65
        + "\n\n"
    )

    file.write(
        "The positive-sentiment result is directionally robust "
        "because every tested combination of dictionary and "
        "preprocessing method shows lower positive advertising "
        "language during recession quarters.\n\n"
    )

    file.write(
        "However, the magnitude of the positive effect changes "
        "across specifications, meaning the estimated effect size "
        "is sensitive to how sentiment is measured.\n\n"
    )

    file.write(
        "Negative sentiment is less stable than positive sentiment. "
        "Harvard changes direction depending on preprocessing, while "
        "Loughran-McDonald shows higher negative language during "
        "recessions across all three preprocessing methods.\n\n"
    )

    file.write(
        "The dictionary-only specification produces much larger "
        "percentage-point effects because the denominator contains "
        "only words recognized by the dictionary. These results are "
        "therefore on a different scale from the all-words and "
        "no-stopwords specifications.\n"
    )


print(
    "\nSAVED:"
)

print(
    summary_file
)


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "DICTIONARY AND PREPROCESSING SENSITIVITY "
    "DIAGNOSTIC COMPLETE"
)

print("=" * 80)

print(
    "\nCreated:"
)

print(
    "1. Full dictionary/preprocessing sensitivity CSV"
)

print(
    "2. Sensitivity range CSV"
)

print(
    "3. Positive sentiment sensitivity graph"
)

print(
    "4. Negative sentiment sensitivity graph"
)

print(
    "5. Human-readable sensitivity summary"
)