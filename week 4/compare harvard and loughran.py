import pandas as pd
import matplotlib.pyplot as plt


# Load the three Harvard quarterly datasets.
harvard_all_words = pd.read_csv(
    "../week three data cleaning/week_three_quarterly_sentiment.csv"
)

harvard_stopwords = pd.read_csv(
    "../week three data cleaning/week_three_quarterly_sentiment_stopwords.csv"
)

harvard_only = pd.read_csv(
    "../week three data cleaning/week_three_quarterly_sentiment_harvard_only.csv"
)


# Load the three Loughran-McDonald quarterly datasets.
lm_all_words = pd.read_csv(
    "second dict and data differences/csv/economist_loughran_mcdonald_quarterly.csv"
)

lm_stopwords = pd.read_csv(
    "second dict and data differences/csv/economist_lm_stopwords_quarterly.csv"
)

lm_only = pd.read_csv(
    "second dict and data differences/csv/economist_lm_only_quarterly.csv"
)


# Load the recession data.
economic = pd.read_csv(
    "../week three data cleaning/nber_recession_table.csv"
)
# Merge each Harvard dataset with recession data.
harvard_all_words = harvard_all_words.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

harvard_stopwords = harvard_stopwords.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

harvard_only = harvard_only.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)


# Merge each Loughran-McDonald dataset with recession data.
lm_all_words = lm_all_words.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

lm_stopwords = lm_stopwords.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

lm_only = lm_only.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)
# Create a function that calculates the recession minus non-recession
# difference for positive and negative sentiment.
def get_recession_difference(data, positive_column, negative_column):

    comparison = (
        data
        .groupby("Recession_Status")[
            [positive_column, negative_column]
        ]
        .mean()
    )

    positive_difference = (
        comparison.loc["Recession", positive_column]
        - comparison.loc["Not Recession", positive_column]
    )

    negative_difference = (
        comparison.loc["Recession", negative_column]
        - comparison.loc["Not Recession", negative_column]
    )

    return positive_difference, negative_difference

    # Harvard differences.
harvard_all_positive, harvard_all_negative = get_recession_difference(
    harvard_all_words,
    "Positiv_Ratio",
    "Negativ_Ratio"
)

harvard_stop_positive, harvard_stop_negative = get_recession_difference(
    harvard_stopwords,
    "Positiv_Ratio",
    "Negativ_Ratio"
)

harvard_only_positive, harvard_only_negative = get_recession_difference(
    harvard_only,
    "Positiv_Ratio",
    "Negativ_Ratio"
)


# Loughran-McDonald differences.
lm_all_positive, lm_all_negative = get_recession_difference(
    lm_all_words,
    "Positive_Ratio",
    "Negative_Ratio"
)

lm_stop_positive, lm_stop_negative = get_recession_difference(
    lm_stopwords,
    "Positive_Ratio",
    "Negative_Ratio"
)

lm_only_positive, lm_only_negative = get_recession_difference(
    lm_only,
    "Positive_Ratio",
    "Negative_Ratio"
)

comparison_table = pd.DataFrame({
    "Dictionary": [
        "Harvard",
        "Harvard",
        "Harvard",
        "Loughran-McDonald",
        "Loughran-McDonald",
        "Loughran-McDonald"
    ],

    "Preprocessing": [
        "All Words",
        "No Stopwords",
        "Dictionary Only",
        "All Words",
        "No Stopwords",
        "Dictionary Only"
    ],

    "Positive Difference (pp)": [
        harvard_all_positive * 100,
        harvard_stop_positive * 100,
        harvard_only_positive * 100,
        lm_all_positive * 100,
        lm_stop_positive * 100,
        lm_only_positive * 100
    ],

    "Negative Difference (pp)": [
        harvard_all_negative * 100,
        harvard_stop_negative * 100,
        harvard_only_negative * 100,
        lm_all_negative * 100,
        lm_stop_negative * 100,
        lm_only_negative * 100
    ]
})

comparison_table = comparison_table.round(4)

print("\nHARVARD VS LOUGHRAN-MCDONALD:")
print(comparison_table)

import os
import numpy as np


# Create output folders if they do not already exist.
os.makedirs("second dict and data differences/images", exist_ok=True)
os.makedirs("second dict and data differences/tables", exist_ok=True)
os.makedirs("second dict and data differences/csv", exist_ok=True)


# Save the comparison table as a CSV as well.
comparison_table.to_csv(
    "second dict and data differences/csv/harvard_vs_loughran_comparison.csv",
    index=False
)


# Create a cleaner label that combines dictionary and preprocessing choice.
comparison_table["Label"] = (
    comparison_table["Dictionary"]
    + " - "
    + comparison_table["Preprocessing"]
)


# -----------------------------
# GRAPH
# -----------------------------
x = np.arange(len(comparison_table))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar(
    x - width / 2,
    comparison_table["Positive Difference (pp)"],
    width,
    label="Positive Difference"
)

ax.bar(
    x + width / 2,
    comparison_table["Negative Difference (pp)"],
    width,
    label="Negative Difference"
)

ax.set_title(
    "Harvard vs. Loughran-McDonald: Recession Effect on Advertising Sentiment"
)
ax.set_xlabel("Dictionary and Preprocessing Method")
ax.set_ylabel("Difference in Percentage Points\n(Recession - Not Recession)")
ax.set_xticks(x)
ax.set_xticklabels(
    comparison_table["Label"],
    rotation=45,
    ha="right"
)
ax.axhline(y=0, linewidth=1)
ax.legend()

plt.tight_layout()

plt.savefig(
    "second dict and data differences/images/harvard_vs_loughran_comparison_graph.png",
    dpi=300
)

plt.close()


# -----------------------------
# TABLE IMAGE
# -----------------------------
table_for_image = comparison_table[
    [
        "Dictionary",
        "Preprocessing",
        "Positive Difference (pp)",
        "Negative Difference (pp)"
    ]
].copy()

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

table = ax.table(
    cellText=table_for_image.values,
    colLabels=table_for_image.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

plt.title(
    "Harvard vs. Loughran-McDonald Recession Comparison",
    pad=20
)

plt.savefig(
    "second dict and data differences/tables/harvard_vs_loughran_comparison_table.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()

print("\nGRAPH AND TABLE IMAGE CREATED")