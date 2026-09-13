import pandas as pd


#Load the data
ads = pd.read_csv(
    "week_three_cleaned_ads.csv"
)


harvard = pd.read_csv(
    "week_three_cleaned_harvard.csv"
)
#The categoires that we didn't scrub out
sentiment_categories = [
    "Positiv",
    "Negativ",
    "Affil",
    "Hostile",
    "Power",
    "Strong",
    "Weak",
    "Pleasur",
    "Pain",
    "Active",
    "Passive",
    "EMOT",
    "Virtue",
    "Vice",
    "Role",
    "Solve",
    "Goal",
    "Means",
    "MeansLw",
    "Complet"
]


#create a set of words that are in each category: basically if it has this trait within the dataset, then it'll filter into a list with all of the others that share that specifc trait
category_word_sets = {}

for category in sentiment_categories:

    category_word_sets[category] = set(
        harvard.loc[
            harvard[category].notna(),
            "Entry"
        ]
        .dropna()
        .astype(str)
        .str.lower()
    )

#First and foremost, I need to convert the OCR data into only lowercase letters that way the strings are the same. I also can remove punctuation for the same reason. It is important to note that we are losing word order and sentence structure with the methodology that we are doing right now, but this is consistent with a bag-of-words method
import re
def clean_ad_text(text):
    text = str(text).lower()
    words = re.findall(r"[a-z']+", text)
    return words
test_words = clean_ad_text(
   ads["OCR_GoogleVision_original"].iloc[0])
#This was used to test to make sure that it was working
#print("\nFIRST 30 CLEANED WORDS:")
#print(test_words[:30])
# Count how many words in one advertisement belong to each Harvard category.
# Count how many words in one advertisement belong to each Harvard category.
def count_categories(words):

    results = {}

    for category, word_set in category_word_sets.items():
        count = sum(
            1 for word in words
            if word in word_set
        )

        results[category] = count

    return results


# Test the category-counting function on the first advertisement.
test_category_counts = count_categories(test_words)

print("\nCATEGORY COUNTS FOR FIRST AD:")
print(test_category_counts)

# We are now able to see the count of each word type; however, longer advertisments should have more in each category so we need to normalize the results. We will use ratios for this process 
def calculate_category_ratios(words, category_counts):

    total_words = len(words)

    ratios = {}

    for category, count in category_counts.items():

        if total_words > 0:
            ratios[category] = count / total_words
        else:
            ratios[category] = 0

    return ratios


# Test the ratio calculation on the first advertisement.
test_category_ratios = calculate_category_ratios(
    test_words,
    test_category_counts
)

print("\nCATEGORY RATIOS FOR FIRST AD:")
print(test_category_ratios)
#We now want to create a new sheet that includes the count and ratio for every advertisment
all_results = []

for text in ads["OCR_GoogleVision_original"]:

    words = clean_ad_text(text)

    category_counts = count_categories(words)

    category_ratios = calculate_category_ratios(
        words,
        category_counts
    )

    # Save the total word count for each advertisement.
    result = {
        "Total_Words": len(words)
    }

    # Add the raw category counts.
    for category, count in category_counts.items():
        result[f"{category}_Count"] = count

    # Add the normalized category ratios.
    for category, ratio in category_ratios.items():
        result[f"{category}_Ratio"] = ratio

    all_results.append(result)


# Turn the results into a dataframe.
sentiment_results = pd.DataFrame(all_results)


# Combine the original advertisement information with the new sentiment measurements.
ads_with_sentiment = pd.concat(
    [
        ads.reset_index(drop=True),
        sentiment_results
    ],
    axis=1
)


# Save the completed advertisement-level sentiment dataset.
ads_with_sentiment.to_csv(
    "week_three_ads_with_sentiment.csv",
    index=False
)

print("\nSENTIMENT DATASET CREATED:")
print(ads_with_sentiment.head())
#because we are doing this by quarters we will now aggregate the data into that format
ratio_columns = [
    column for column in ads_with_sentiment.columns
    if column.endswith("_Ratio")
]

quarterly_sentiment = (
    ads_with_sentiment
    .groupby("Year_Quarter")[ratio_columns]
    .mean()
    .reset_index()
)

# Also keep track of how many advertisements are in each quarter.
quarterly_ad_counts = (
    ads_with_sentiment
    .groupby("Year_Quarter")
    .size()
    .reset_index(name="Advertisement_Count")
)

# Merge the quarterly sentiment averages with the number of advertisements.
quarterly_sentiment = quarterly_sentiment.merge(
    quarterly_ad_counts,
    on="Year_Quarter"
)

# Save the quarterly sentiment dataset.
quarterly_sentiment.to_csv(
    "week_three_quarterly_sentiment.csv",
    index=False
)

print("\nQUARTERLY SENTIMENT DATASET CREATED:")
print(quarterly_sentiment.head())
