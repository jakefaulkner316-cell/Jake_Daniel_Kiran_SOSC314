import pandas as pd
import matplotlib.pyplot as plt
import re

from nltk.corpus import stopwords


# ---------------------------------------------------------
# LOAD UPDATED WEEK 4 ADVERTISEMENT DATA
# ---------------------------------------------------------

# This is the updated advertisement dataset containing the
# Week 4 quality flags and filtering information.
ads = pd.read_csv(
    "Ad_filtering/week_four_quality_flags.csv"
)


# ---------------------------------------------------------
# LOAD HARVARD GENERAL INQUIRER
# ---------------------------------------------------------

harvard = pd.read_excel(
    "../Initial file imports/inquireraugmented.xls",
    engine="xlrd"
)


# ---------------------------------------------------------
# LOAD LOUGHRAN-MCDONALD MASTER DICTIONARY
# ---------------------------------------------------------

lm_dictionary = pd.read_csv(
    "second dict and data differences/csv/Loughran-McDonald_MasterDictionary_1993-2025.csv"
)


# ---------------------------------------------------------
# LOAD RECESSION DATA
# ---------------------------------------------------------

recession_data = pd.read_csv(
    "../week three data cleaning/nber_recession_table.csv"
)


# ---------------------------------------------------------
# LOAD ENGLISH STOPWORDS
# ---------------------------------------------------------

stop_words = set(
    stopwords.words("english")
)


# ---------------------------------------------------------
# CHECK THAT EVERYTHING LOADED CORRECTLY
# ---------------------------------------------------------

print("\nUPDATED AD DATA:")
print(ads.head())

print("\nAD DATA COLUMNS:")
print(ads.columns)

print("\nNUMBER OF ADS:")
print(len(ads))


print("\nHARVARD DICTIONARY SIZE:")
print(harvard.shape)


print("\nLOUGHRAN-MCDONALD DICTIONARY SIZE:")
print(lm_dictionary.shape)


print("\nRECESSION DATA:")
print(recession_data.head())

# ---------------------------------------------------------
# LOAD ORIGINAL ECONOMIST TEXT
# ---------------------------------------------------------

# The Week 4 quality file contains the quality flags, but it does not
# contain the OCR advertisement text. We therefore reload the original
# Economist dataset and recreate the same post-1948 branded corpus.

raw_ads = pd.read_csv(
    "../Initial file imports/The Economist Historical Advertisements – Master Dataset.csv"
)

# Keep the variables needed for the analysis.
raw_ads = raw_ads[
    [
        "DateOfIssue",
        "Brand",
        "Brand is generic (e.g. 'Notices')",
        "OCR_GoogleVision_original"
    ]
].copy()

# Convert the date variable.
raw_ads["DateOfIssue"] = pd.to_datetime(
    raw_ads["DateOfIssue"],
    errors="coerce"
)

# Create year and quarter variables.
raw_ads["Year"] = raw_ads["DateOfIssue"].dt.year
raw_ads["Quarter"] = raw_ads["DateOfIssue"].dt.quarter
raw_ads["Year_Quarter"] = (
    raw_ads["DateOfIssue"]
    .dt.to_period("Q")
    .astype(str)
)

# Recreate the same corpus used in the earlier sentiment analysis.
raw_ads = raw_ads[
    raw_ads["Year"] >= 1948
].copy()

raw_ads = raw_ads[
    raw_ads["Brand is generic (e.g. 'Notices')"] == False
].copy()

# Reset the row numbers so they match the row_id used in
# the Week 4 quality-flag dataset.
raw_ads = raw_ads.reset_index(drop=True)
raw_ads["row_id"] = raw_ads.index


# ---------------------------------------------------------
# MERGE WEEK 4 QUALITY FLAGS WITH THE AD TEXT
# ---------------------------------------------------------

ads_with_quality = raw_ads.merge(
    ads,
    on="row_id",
    how="inner",
    suffixes=("", "_quality")
)

print("\nADS AFTER MERGING QUALITY FLAGS:")
print(len(ads_with_quality))


# Keep only advertisements that did not trigger a Week 4 quality flag.
quality_filtered_ads = ads_with_quality[
    ads_with_quality["any_flag"] == False
].copy()

print("\nQUALITY FILTERING SUMMARY:")
print("Before quality filtering:", len(ads_with_quality))
print("Removed:", ads_with_quality["any_flag"].sum())
print("Remaining:", len(quality_filtered_ads))


# ---------------------------------------------------------
# PREPARE LOUGHRAN-MCDONALD DICTIONARY
# ---------------------------------------------------------

lm_categories = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
    "Complexity"
]

# Convert dictionary words to lowercase.
lm_dictionary["Word"] = (
    lm_dictionary["Word"]
    .astype(str)
    .str.lower()
)

# Create a set of words for each LM category.
lm_category_word_sets = {}

for category in lm_categories:

    lm_category_word_sets[category] = set(
        lm_dictionary.loc[
            lm_dictionary[category] > 0,
            "Word"
        ]
    )


# ---------------------------------------------------------
# CLEAN OCR TEXT
# ---------------------------------------------------------

def clean_ad_text(text):

    text = str(text).lower()

    words = re.findall(
        r"[a-z']+",
        text
    )

    return words


# ---------------------------------------------------------
# COUNT LOUGHRAN-MCDONALD CATEGORIES
# ---------------------------------------------------------

def count_lm_categories(words):

    results = {}

    for category, word_set in lm_category_word_sets.items():

        results[category] = sum(
            1 for word in words
            if word in word_set
        )

    return results


# ---------------------------------------------------------
# CALCULATE CATEGORY RATIOS
# ---------------------------------------------------------

def calculate_lm_ratios(words, category_counts):

    total_words = len(words)

    ratios = {}

    for category, count in category_counts.items():

        if total_words > 0:
            ratios[category] = count / total_words
        else:
            ratios[category] = 0

    return ratios


# ---------------------------------------------------------
# RUN LM ANALYSIS ON QUALITY-FILTERED ADS
# ---------------------------------------------------------

lm_results = []

for text in quality_filtered_ads["OCR_GoogleVision_original"]:

    words = clean_ad_text(text)

    counts = count_lm_categories(
        words
    )

    ratios = calculate_lm_ratios(
        words,
        counts
    )

    result = {}

    for category, ratio in ratios.items():
        result[f"{category}_Ratio"] = ratio

    lm_results.append(result)


lm_results = pd.DataFrame(
    lm_results
)

quality_filtered_lm = pd.concat(
    [
        quality_filtered_ads.reset_index(drop=True),
        lm_results
    ],
    axis=1
)


# ---------------------------------------------------------
# AGGREGATE TO QUARTER
# ---------------------------------------------------------

lm_ratio_columns = [
    column for column in quality_filtered_lm.columns
    if column.endswith("_Ratio")
]

lm_quarterly = (
    quality_filtered_lm
    .groupby("Year_Quarter")[lm_ratio_columns]
    .mean()
    .reset_index()
)


# ---------------------------------------------------------
# MERGE WITH RECESSION DATA
# ---------------------------------------------------------

lm_quarterly = lm_quarterly.merge(
    recession_data[
        [
            "Year_Quarter",
            "Recession_Majority"
        ]
    ],
    on="Year_Quarter",
    how="inner"
)


# ---------------------------------------------------------
# COMPARE RECESSION VS NON-RECESSION QUARTERS
# ---------------------------------------------------------

lm_category_comparison = (
    lm_quarterly
    .groupby("Recession_Majority")[lm_ratio_columns]
    .mean()
)

# Recession minus non-recession.
lm_category_difference = (
    lm_category_comparison.loc[True]
    - lm_category_comparison.loc[False]
) * 100

# Make labels easier to read.
lm_category_difference.index = (
    lm_category_difference.index
    .str.replace("_Ratio", "", regex=False)
)


# ---------------------------------------------------------
# CREATE IMAGE
# ---------------------------------------------------------

# Create a numeric x-axis position for each quarter.
x = range(len(lm_quarterly))

plt.figure(figsize=(15, 8))

# Plot every Loughran-McDonald category over time.
for column in lm_ratio_columns:

    label = column.replace(
        "_Ratio",
        ""
    )

    plt.plot(
        x,
        lm_quarterly[column] * 100,
        label=label
    )


# Make the x-axis readable by only displaying every 20th quarter.
tick_positions = list(
    range(
        0,
        len(lm_quarterly),
        20
    )
)

tick_labels = (
    lm_quarterly["Year_Quarter"]
    .iloc[tick_positions]
)


plt.xticks(
    tick_positions,
    tick_labels,
    rotation=45,
    ha="right"
)


plt.title(
    "Loughran-McDonald Advertising Language Over Time\n"
    "Week 4 Quality-Filtered Economist Advertisements"
)

plt.xlabel(
    "Quarter"
)

plt.ylabel(
    "Average Share of Advertisement Words (%)"
)

plt.legend(
    title="Language Category"
)

plt.tight_layout()


# Save the new time-series graph.
plt.savefig(
    "images/week4_quality_filtered_lm_categories_over_time.png",
    dpi=300
)

plt.close()


print(
    "\nCREATED: images/week4_quality_filtered_lm_categories_over_time.png"
)