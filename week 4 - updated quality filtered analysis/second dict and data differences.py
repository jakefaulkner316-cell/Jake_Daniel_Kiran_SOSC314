import pandas as pd


# Load the Loughran-McDonald Master Dictionary.
lm_dictionary = pd.read_csv(
    "second dict and data differences/csv/Loughran-McDonald_MasterDictionary_1993-2025.csv"
)


# Show the first five rows to make sure the file loaded correctly.
#print("\nFIRST FIVE ROWS:")
#print(lm_dictionary.head())


# Show the column names so we can see which sentiment categories are available.
#print("\nCOLUMN NAMES:")
#print(lm_dictionary.columns)


# Show the size of the dictionary.
#print("\nDICTIONARY SIZE:")
#print(lm_dictionary.shape)

# Word:
# The actual dictionary word that will be matched against the Economist advertisement text.
#
# Negative:
# Words with a negative meaning in financial or economic contexts.
#
# Positive:
# Words with a positive meaning in financial or economic contexts.
#
# Uncertainty:
# Words that express uncertainty, doubt, risk, or lack of clarity.
#
# Litigious:
# Words related to legal disputes, lawsuits, regulation, or legal risk.
#
# Strong_Modal:
# Strong modal words that express a high level of certainty, obligation,
# or necessity, such as language similar to "must" or "will".
#
# Weak_Modal:
# Weak modal words that express possibility, uncertainty, or weaker commitment,
# such as language similar to "may", "might", or "could".
#
# Constraining:
# Words that suggest limits, restrictions, constraints, or barriers.
#
# Complexity:
# Words associated with complexity, difficulty, or complicated situations.
#
# These categories are useful because the Loughran-McDonald dictionary was
# designed for financial and economic language. This gives us a second
# dictionary-based measurement approach that we can compare with the
# broader Harvard General Inquirer results.

lm_categories = [
    "Word",
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
    "Complexity"
]

lm_cleaned = lm_dictionary[lm_categories].copy()

# Convert the dictionary words to lowercase so they can be matched
# consistently with the cleaned Economist advertisement text.
lm_cleaned["Word"] = (
    lm_cleaned["Word"]
    .astype(str)
    .str.lower()
)

lm_cleaned.to_csv(
    "second dict and data differences/csv/loughran_mcdonald_cleaned.csv",
    index=False
)
# Load the same Economist advertisement corpus used

ads = pd.read_csv(
    "../Initial file imports/The Economist Historical Advertisements – Master Dataset.csv"
)
# Keep only the variables needed for the sentiment analysis.
ads = ads[
    [
        "DateOfIssue",
        "Brand",
        "Brand is generic (e.g. 'Notices')",
        "OCR_GoogleVision_original"
    ]
].copy()

# Convert the date into a usable format.
ads["DateOfIssue"] = pd.to_datetime(
    ads["DateOfIssue"],
    errors="coerce"
)

# Create year and quarter variables.
ads["Year"] = ads["DateOfIssue"].dt.year
ads["Quarter"] = ads["DateOfIssue"].dt.quarter
ads["Year_Quarter"] = ads["DateOfIssue"].dt.to_period("Q")

# Keep only advertisements from 1948 onward.
ads = ads[
    ads["Year"] >= 1948
].copy()

# Keep only branded advertisements.
ads = ads[
    ads["Brand is generic (e.g. 'Notices')"] == False
].copy()
# These are the Loughran-McDonald categories we want to measure.
lm_sentiment_categories = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
    "Complexity"
]


# Create a set of words for each Loughran-McDonald category.
# In this dictionary, a value greater than 0 means that the word
# belongs to that category.
lm_category_word_sets = {}

for category in lm_sentiment_categories:

    lm_category_word_sets[category] = set(
        lm_cleaned.loc[
            lm_cleaned[category] > 0,
            "Word"
        ]
    )


# Check how many words are in each category.
print("\nLOUGHRAN-MCDONALD CATEGORY SIZES:")

for category, words in lm_category_word_sets.items():
    print(category, len(words))
import re


# Convert the Economist OCR text to lowercase and remove punctuation
# so words can be matched consistently with the Loughran-McDonald dictionary.
def clean_ad_text(text):

    text = str(text).lower()

    words = re.findall(
        r"[a-z']+",
        text
    )

    return words


# Count how many words in one advertisement belong to each
# Loughran-McDonald category.
def count_lm_categories(words):

    results = {}

    for category, word_set in lm_category_word_sets.items():

        count = sum(
            1 for word in words
            if word in word_set
        )

        results[category] = count

    return results


# Normalize the category counts by the total number of words
# so longer advertisements are not automatically given higher scores.
def calculate_lm_ratios(words, category_counts):

    total_words = len(words)

    ratios = {}

    for category, count in category_counts.items():

        if total_words > 0:
            ratios[category] = count / total_words

        else:
            ratios[category] = 0

    return ratios


# Test the process on the first advertisement.
test_words = clean_ad_text(
    ads["OCR_GoogleVision_original"].iloc[0]
)

test_counts = count_lm_categories(
    test_words
)

test_ratios = calculate_lm_ratios(
    test_words,
    test_counts
)

print("\nFIRST AD LOUGHRAN-MCDONALD COUNTS:")
print(test_counts)

print("\nFIRST AD LOUGHRAN-MCDONALD RATIOS:")
print(test_ratios)
# Run the Loughran-McDonald analysis for every Economist advertisement.
lm_results = []

for text in ads["OCR_GoogleVision_original"]:

    words = clean_ad_text(text)

    category_counts = count_lm_categories(
        words
    )

    category_ratios = calculate_lm_ratios(
        words,
        category_counts
    )

    result = {
        "Total_Words": len(words)
    }

    # Add the raw category counts.
    for category, count in category_counts.items():
        result[f"{category}_Count"] = count

    # Add the normalized category ratios.
    for category, ratio in category_ratios.items():
        result[f"{category}_Ratio"] = ratio

    lm_results.append(result)


# Turn the results into a dataframe.
lm_results = pd.DataFrame(
    lm_results
)


# Combine the original advertisement information
# with the Loughran-McDonald measurements.
ads_with_lm = pd.concat(
    [
        ads.reset_index(drop=True),
        lm_results
    ],
    axis=1
)


# Save the advertisement-level Loughran-McDonald dataset.
ads_with_lm.to_csv(
    "second dict and data differences/csv/economist_loughran_mcdonald_sentiment.csv",
    index=False
)

print("\nLOUGHRAN-MCDONALD AD-LEVEL DATASET CREATED:")
print(ads_with_lm.head())

print("\nNUMBER OF ADS PROCESSED:")
print(len(ads_with_lm))

# Create a quarterly Loughran-McDonald sentiment dataset.

lm_ratio_columns = [
    column for column in ads_with_lm.columns
    if column.endswith("_Ratio")
]

quarterly_lm = (
    ads_with_lm
    .groupby("Year_Quarter")[lm_ratio_columns]
    .mean()
    .reset_index()
)

quarterly_lm_counts = (
    ads_with_lm
    .groupby("Year_Quarter")
    .size()
    .reset_index(name="Advertisement_Count")
)

quarterly_lm = quarterly_lm.merge(
    quarterly_lm_counts,
    on="Year_Quarter"
)

quarterly_lm.to_csv(
    "second dict and data differences/csv/economist_loughran_mcdonald_quarterly.csv",
    index=False
)

print("\nLOUGHRAN-MCDONALD QUARTERLY DATASET CREATED:")
print(quarterly_lm.head())

# Create the no-stopword and LM-only ad-level datasets.

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

all_lm_words = set()

for word_set in lm_category_word_sets.values():
    all_lm_words.update(word_set)


def no_stopwords_version(words):
    return [
        word for word in words
        if word not in stop_words
    ]


def lm_only_version(words):
    return [
        word for word in words
        if word in all_lm_words
    ]


lm_no_stopwords_results = []
lm_only_results = []

for text in ads["OCR_GoogleVision_original"]:

    words = clean_ad_text(text)

    # No-stopwords version.
    words_no_stopwords = no_stopwords_version(words)

    counts_no_stopwords = count_lm_categories(
        words_no_stopwords
    )

    ratios_no_stopwords = calculate_lm_ratios(
        words_no_stopwords,
        counts_no_stopwords
    )

    result_no_stopwords = {
        "Total_Words": len(words_no_stopwords)
    }

    for category, count in counts_no_stopwords.items():
        result_no_stopwords[f"{category}_Count"] = count

    for category, ratio in ratios_no_stopwords.items():
        result_no_stopwords[f"{category}_Ratio"] = ratio

    lm_no_stopwords_results.append(
        result_no_stopwords
    )


    # LM-only version.
    words_lm_only = lm_only_version(words)

    counts_lm_only = count_lm_categories(
        words_lm_only
    )

    ratios_lm_only = calculate_lm_ratios(
        words_lm_only,
        counts_lm_only
    )

    result_lm_only = {
        "Total_Words": len(words_lm_only)
    }

    for category, count in counts_lm_only.items():
        result_lm_only[f"{category}_Count"] = count

    for category, ratio in ratios_lm_only.items():
        result_lm_only[f"{category}_Ratio"] = ratio

    lm_only_results.append(
        result_lm_only
    )


# Convert the result lists into dataframes.
lm_no_stopwords_results = pd.DataFrame(
    lm_no_stopwords_results
)

lm_only_results = pd.DataFrame(
    lm_only_results
)


# Combine with the original ad information.
ads_lm_no_stopwords = pd.concat(
    [
        ads.reset_index(drop=True),
        lm_no_stopwords_results
    ],
    axis=1
)

ads_lm_only = pd.concat(
    [
        ads.reset_index(drop=True),
        lm_only_results
    ],
    axis=1
)


# Save both ad-level datasets.
ads_lm_no_stopwords.to_csv(
    "second dict and data differences/csv/economist_lm_stopwords.csv",
    index=False
)

ads_lm_only.to_csv(
    "second dict and data differences/csv/economist_lm_only.csv",
    index=False
)

print("\nSTOPWORD AND LM-ONLY AD-LEVEL DATASETS CREATED")

# Create quarterly versions of the stopword and LM-only datasets.

lm_ratio_columns = [
    column for column in ads_lm_no_stopwords.columns
    if column.endswith("_Ratio")
]


def make_quarterly_lm_dataset(ad_data):

    quarterly = (
        ad_data
        .groupby("Year_Quarter")[lm_ratio_columns]
        .mean()
        .reset_index()
    )

    quarterly_ad_counts = (
        ad_data
        .groupby("Year_Quarter")
        .size()
        .reset_index(name="Advertisement_Count")
    )

    quarterly = quarterly.merge(
        quarterly_ad_counts,
        on="Year_Quarter"
    )

    return quarterly


quarterly_lm_stopwords = make_quarterly_lm_dataset(
    ads_lm_no_stopwords
)

quarterly_lm_only = make_quarterly_lm_dataset(
    ads_lm_only
)


quarterly_lm_stopwords.to_csv(
    "second dict and data differences/csv/economist_lm_stopwords_quarterly.csv",
    index=False
)

quarterly_lm_only.to_csv(
    "second dict and data differences/csv/economist_lm_only_quarterly.csv",
    index=False
)

print("\nSTOPWORD AND LM-ONLY QUARTERLY DATASETS CREATED")