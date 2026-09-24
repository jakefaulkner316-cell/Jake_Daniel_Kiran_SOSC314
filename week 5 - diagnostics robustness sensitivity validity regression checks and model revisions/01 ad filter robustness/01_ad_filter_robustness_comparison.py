# =========================================================
# WEEK 5 DIAGNOSTIC 1:
# AD FILTER ROBUSTNESS COMPARISON
# =========================================================
#
# PURPOSE:
# This script tests whether applying the Week 4 advertisement
# quality filters materially changes our main recession-related
# sentiment findings.
#
# WEEK 4 created quality flags for advertisements that may be
# problematic observations. These include:
#
# - likely non-advertisers
# - multipage advertisements
# - advertisements that are too short
# - advertisements triggering any quality flag
#
# The full corpus contains 172,219 advertisements.
# The quality-filtered corpus contains 168,916 advertisements.
# Therefore, 3,303 advertisements are removed by the quality
# filtering procedure.
#
# WHAT WE COMPARE:
#
# 1. Harvard General Inquirer
#    - Full corpus
#    - Quality-filtered corpus
#
# 2. Loughran-McDonald
#    - Full corpus
#    - Quality-filtered corpus
#
# We examine each dictionary using two aggregation methods:
#
# A. Average of ratios ("macro")
#    Each advertisement receives equal weight.
#    We calculate each advertisement's sentiment ratio first
#    and then average those ratios within each quarter.
#
# B. Pooled words
#    All category words and all total words are summed within
#    the quarter before calculating the sentiment ratio.
#    Longer advertisements therefore receive more weight.
#
# MAIN DIAGNOSTIC QUESTION:
#
# Does removing the 3,303 quality-flagged advertisements
# materially change the recession-minus-non-recession
# difference in positive or negative sentiment?
#
# INTERPRETATION:
#
# If the before-filtering and after-filtering estimates are
# very similar, the sentiment result is robust to the
# advertisement-quality filtering decision.
#
# If the estimates change substantially, or if their direction
# changes, then the original result may be sensitive to the
# composition and quality of the advertisement corpus.
#
# WEEK 5 REQUIREMENTS ADDRESSED:
#
# - Robustness / sensitivity checks
# - Discussion of validity and possible failure modes
# - Diagnostics appropriate to our dictionary method
# - Revisions informed by diagnostic findings
#
# OUTPUTS:
#
# csv outputs/
#   ad_filter_before_vs_after_sentiment_robustness.csv
#
# diagnostic images/
#   positive_sentiment_before_vs_after_ad_filtering.png
#   negative_sentiment_before_vs_after_ad_filtering.png
#
# robustness tables/
#   ad_filter_robustness_summary.txt
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
#
# Using the location of this Python script makes the paths
# work even if the script is launched from a different folder.
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

WEEK5_DIR = SCRIPT_DIR.parent

REPO_ROOT = WEEK5_DIR.parent

WEEK4_DIR = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
)

WEEK4_AGGREGATION_DIR = (
    WEEK4_DIR
    / "week4_aggregations"
)

RESULTS_FILE = (
    WEEK4_AGGREGATION_DIR
    / "aggregation_sensitivity_results.csv"
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


# Create the folders if they do not already exist.

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
# LOAD WEEK 4 SENSITIVITY RESULTS
# ---------------------------------------------------------
#
# Week 4 already performed the expensive advertisement-level
# calculations and correctly applied the quality flags.
#
# We therefore use those validated results rather than
# unnecessarily reprocessing all 172,219 advertisements.
# ---------------------------------------------------------

results = pd.read_csv(
    RESULTS_FILE
)


# ---------------------------------------------------------
# VERIFY REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "Dictionary",
    "Corpus",
    "Aggregation",
    "Ads_Used",
    "Quarters",
    "Pos_Recession",
    "Pos_Expansion",
    "Pos_Diff_pp",
    "Neg_Diff_pp"
]


missing_columns = [
    column
    for column in required_columns
    if column not in results.columns
]


if missing_columns:

    raise ValueError(
        "The Week 4 results file is missing these columns: "
        + ", ".join(missing_columns)
    )


# ---------------------------------------------------------
# PRINT ORIGINAL WEEK 4 RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "WEEK 4 AGGREGATION AND QUALITY-FILTER RESULTS"
)

print("=" * 80)

print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# VERIFY FULL AND FILTERED CORPUS SIZES
# ---------------------------------------------------------

full_ads = (
    results.loc[
        results["Corpus"] == "Full corpus",
        "Ads_Used"
    ]
    .iloc[0]
)

filtered_ads = (
    results.loc[
        results["Corpus"] == "Filtered",
        "Ads_Used"
    ]
    .iloc[0]
)

ads_removed = (
    full_ads
    - filtered_ads
)

percent_removed = (
    ads_removed
    / full_ads
) * 100


print("\n" + "=" * 80)

print(
    "ADVERTISEMENT QUALITY FILTER CHECK"
)

print("=" * 80)

print(
    f"Full corpus: {int(full_ads):,} ads"
)

print(
    f"Filtered corpus: {int(filtered_ads):,} ads"
)

print(
    f"Ads removed: {int(ads_removed):,}"
)

print(
    f"Percent of corpus removed: {percent_removed:.2f}%"
)


# ---------------------------------------------------------
# CALCULATE BEFORE-VS-AFTER FILTERING CHANGES
# ---------------------------------------------------------
#
# We compare:
#
# filtered estimate - full corpus estimate
#
# Values close to zero mean filtering has little effect.
#
# Positive sentiment values are recession minus expansion.
# Negative sentiment values are also recession minus expansion.
# ---------------------------------------------------------

comparison_rows = []


for dictionary in [
    "Harvard",
    "Loughran-McDonald"
]:

    for aggregation in [
        "Average of ratios",
        "Pooled words"
    ]:

        full_row = results[
            (
                results["Dictionary"]
                == dictionary
            )
            &
            (
                results["Corpus"]
                == "Full corpus"
            )
            &
            (
                results["Aggregation"]
                == aggregation
            )
        ].iloc[0]


        filtered_row = results[
            (
                results["Dictionary"]
                == dictionary
            )
            &
            (
                results["Corpus"]
                == "Filtered"
            )
            &
            (
                results["Aggregation"]
                == aggregation
            )
        ].iloc[0]


        positive_full = (
            full_row["Pos_Diff_pp"]
        )

        positive_filtered = (
            filtered_row["Pos_Diff_pp"]
        )

        negative_full = (
            full_row["Neg_Diff_pp"]
        )

        negative_filtered = (
            filtered_row["Neg_Diff_pp"]
        )


        positive_change = (
            positive_filtered
            - positive_full
        )

        negative_change = (
            negative_filtered
            - negative_full
        )


        # Check whether filtering changes the direction
        # of the recession effect.

        positive_same_direction = (
            positive_full
            * positive_filtered
            >= 0
        )

        negative_same_direction = (
            negative_full
            * negative_filtered
            >= 0
        )


        comparison_rows.append(
            {
                "Dictionary":
                    dictionary,

                "Aggregation":
                    aggregation,

                "Full_Ads":
                    int(
                        full_row[
                            "Ads_Used"
                        ]
                    ),

                "Filtered_Ads":
                    int(
                        filtered_row[
                            "Ads_Used"
                        ]
                    ),

                "Ads_Removed":
                    int(
                        full_row[
                            "Ads_Used"
                        ]
                        -
                        filtered_row[
                            "Ads_Used"
                        ]
                    ),

                "Positive_Full_pp":
                    positive_full,

                "Positive_Filtered_pp":
                    positive_filtered,

                "Positive_Filter_Change_pp":
                    positive_change,

                "Positive_Absolute_Change_pp":
                    abs(
                        positive_change
                    ),

                "Positive_Same_Direction":
                    positive_same_direction,

                "Negative_Full_pp":
                    negative_full,

                "Negative_Filtered_pp":
                    negative_filtered,

                "Negative_Filter_Change_pp":
                    negative_change,

                "Negative_Absolute_Change_pp":
                    abs(
                        negative_change
                    ),

                "Negative_Same_Direction":
                    negative_same_direction
            }
        )


# Turn the comparison results into a DataFrame.

filter_comparison = pd.DataFrame(
    comparison_rows
)


# Round numerical values for readability.

numeric_columns = [
    "Positive_Full_pp",
    "Positive_Filtered_pp",
    "Positive_Filter_Change_pp",
    "Positive_Absolute_Change_pp",
    "Negative_Full_pp",
    "Negative_Filtered_pp",
    "Negative_Filter_Change_pp",
    "Negative_Absolute_Change_pp"
]


filter_comparison[
    numeric_columns
] = (
    filter_comparison[
        numeric_columns
    ]
    .round(4)
)


# ---------------------------------------------------------
# PRINT FILTERING ROBUSTNESS RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "CHANGE CAUSED BY ADVERTISEMENT QUALITY FILTERING"
)

print("=" * 80)

print(
    filter_comparison.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE COMPARISON RESULTS AS CSV
# ---------------------------------------------------------

comparison_csv_path = (
    CSV_OUTPUT_DIR
    / "ad_filter_before_vs_after_sentiment_robustness.csv"
)

filter_comparison.to_csv(
    comparison_csv_path,
    index=False
)


print(
    "\nSAVED:"
)

print(
    comparison_csv_path
)


# ---------------------------------------------------------
# CREATE LABELS FOR GRAPHS
# ---------------------------------------------------------

plot_data = (
    filter_comparison.copy()
)

plot_data["Method"] = (
    plot_data["Dictionary"]
    + "\n"
    + plot_data["Aggregation"]
)


x_positions = list(
    range(
        len(
            plot_data
        )
    )
)

bar_width = 0.35


# =========================================================
# POSITIVE SENTIMENT ROBUSTNESS GRAPH
# =========================================================
#
# Each pair of bars compares the recession effect before
# and after quality filtering.
#
# A negative value means positive language is lower during
# recession quarters than during non-recession quarters.
# =========================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)


ax.bar(
    [
        position
        - bar_width / 2
        for position in x_positions
    ],
    plot_data[
        "Positive_Full_pp"
    ],
    width=bar_width,
    label="Full corpus"
)


ax.bar(
    [
        position
        + bar_width / 2
        for position in x_positions
    ],
    plot_data[
        "Positive_Filtered_pp"
    ],
    width=bar_width,
    label="Quality filtered"
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(
    plot_data[
        "Method"
    ],
    rotation=20,
    ha="right"
)


ax.set_title(
    "Effect of Advertisement Quality Filtering on "
    "Recession-Related Positive Sentiment"
)


ax.set_xlabel(
    "Dictionary and Aggregation Method"
)


ax.set_ylabel(
    "Recession Minus Non-Recession Difference "
    "(Percentage Points)"
)


ax.legend()


plt.tight_layout()


positive_image_path = (
    IMAGE_OUTPUT_DIR
    / "positive_sentiment_before_vs_after_ad_filtering.png"
)


plt.savefig(
    positive_image_path,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    positive_image_path
)


# =========================================================
# NEGATIVE SENTIMENT ROBUSTNESS GRAPH
# =========================================================
#
# A positive value means negative language is more common
# during recession quarters.
#
# A negative value means negative language is slightly less
# common during recession quarters.
# =========================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)


ax.bar(
    [
        position
        - bar_width / 2
        for position in x_positions
    ],
    plot_data[
        "Negative_Full_pp"
    ],
    width=bar_width,
    label="Full corpus"
)


ax.bar(
    [
        position
        + bar_width / 2
        for position in x_positions
    ],
    plot_data[
        "Negative_Filtered_pp"
    ],
    width=bar_width,
    label="Quality filtered"
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(
    plot_data[
        "Method"
    ],
    rotation=20,
    ha="right"
)


ax.set_title(
    "Effect of Advertisement Quality Filtering on "
    "Recession-Related Negative Sentiment"
)


ax.set_xlabel(
    "Dictionary and Aggregation Method"
)


ax.set_ylabel(
    "Recession Minus Non-Recession Difference "
    "(Percentage Points)"
)


ax.legend()


plt.tight_layout()


negative_image_path = (
    IMAGE_OUTPUT_DIR
    / "negative_sentiment_before_vs_after_ad_filtering.png"
)


plt.savefig(
    negative_image_path,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    negative_image_path
)


# =========================================================
# CREATE HUMAN-READABLE ROBUSTNESS SUMMARY
# =========================================================
#
# This text file gives us notes that can later be used in:
#
# - the Week 5 written report
# - the Week 6 oral check
# - the statistical interpretation section
# =========================================================

summary_path = (
    TABLE_OUTPUT_DIR
    / "ad_filter_robustness_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 AD FILTER ROBUSTNESS DIAGNOSTIC\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Full corpus: {int(full_ads):,} advertisements\n"
    )

    file.write(
        f"Quality-filtered corpus: {int(filtered_ads):,} advertisements\n"
    )

    file.write(
        f"Advertisements removed: {int(ads_removed):,}\n"
    )

    file.write(
        f"Percent removed: {percent_removed:.2f}%\n\n"
    )


    file.write(
        "ROBUSTNESS RESULTS\n"
    )

    file.write(
        "-" * 60
        + "\n\n"
    )


    for _, row in filter_comparison.iterrows():

        file.write(
            f"Dictionary: {row['Dictionary']}\n"
        )

        file.write(
            f"Aggregation: {row['Aggregation']}\n"
        )

        file.write(
            "Positive recession effect before filtering: "
            f"{row['Positive_Full_pp']:+.4f} pp\n"
        )

        file.write(
            "Positive recession effect after filtering: "
            f"{row['Positive_Filtered_pp']:+.4f} pp\n"
        )

        file.write(
            "Change caused by filtering: "
            f"{row['Positive_Filter_Change_pp']:+.4f} pp\n"
        )

        file.write(
            "Positive effect keeps same direction: "
            f"{row['Positive_Same_Direction']}\n"
        )

        file.write(
            "Negative recession effect before filtering: "
            f"{row['Negative_Full_pp']:+.4f} pp\n"
        )

        file.write(
            "Negative recession effect after filtering: "
            f"{row['Negative_Filtered_pp']:+.4f} pp\n"
        )

        file.write(
            "Change caused by filtering: "
            f"{row['Negative_Filter_Change_pp']:+.4f} pp\n"
        )

        file.write(
            "Negative effect keeps same direction: "
            f"{row['Negative_Same_Direction']}\n"
        )

        file.write(
            "\n"
        )


    file.write(
        "INTERPRETATION\n"
    )

    file.write(
        "-" * 60
        + "\n\n"
    )

    file.write(
        "The quality filter removes 3,303 advertisements, "
        "or roughly 1.9 percent of the full corpus.\n\n"
    )

    file.write(
        "The main positive-sentiment recession result remains "
        "negative across both dictionaries and both aggregation "
        "methods after filtering. This indicates that the finding "
        "of lower positive advertising language during recessions "
        "is robust to the Week 4 advertisement-quality filters.\n\n"
    )

    file.write(
        "The Loughran-McDonald results change very little after "
        "filtering, indicating particularly strong stability to "
        "the quality-filter decision.\n\n"
    )

    file.write(
        "Harvard negative sentiment using the average-of-ratios "
        "method changes from a very small positive estimate to a "
        "very small negative estimate. Because both values are "
        "extremely close to zero, this should be interpreted as "
        "an unstable near-zero relationship rather than a strong "
        "substantive reversal.\n"
    )


print(
    "\nSAVED:"
)

print(
    summary_path
)


# ---------------------------------------------------------
# FINAL COMPLETION MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "AD FILTER ROBUSTNESS DIAGNOSTIC COMPLETE"
)

print("=" * 80)

print(
    "\nCreated:"
)

print(
    "1. Before-vs-after robustness CSV"
)

print(
    "2. Positive sentiment diagnostic graph"
)

print(
    "3. Negative sentiment diagnostic graph"
)

print(
    "4. Human-readable robustness summary"
)