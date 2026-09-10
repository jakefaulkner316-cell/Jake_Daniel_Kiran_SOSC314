import pandas as pd
import matplotlib.pyplot as plt

# Load the three advertisement-level sentiment datasets.
ads_original = pd.read_csv(
    "week_three_ads_with_sentiment.csv"
)

ads_no_stopwords = pd.read_csv(
    "week_three_ads_with_sentiment_stopwords.csv"
)

ads_harvard_only = pd.read_csv(
    "week_three_ads_with_sentiment_harvard_only.csv"
)

# Load economic/recession data.
economic = pd.read_csv(
    "../early statistical analysis/us_recession_table.csv"
)
# Find all of the Harvard ratio columns.
ratio_columns = [
    column for column in ads_original.columns
    if column.endswith("_Ratio")
]

# Create a function that turns advertisement-level data
# into quarterly average sentiment data.
def make_quarterly_dataset(ad_data):

    quarterly = (
        ad_data
        .groupby("Year_Quarter")[ratio_columns]
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


# Create quarterly versions of all three datasets.
quarterly_original = make_quarterly_dataset(
    ads_original
)

quarterly_no_stopwords = make_quarterly_dataset(
    ads_no_stopwords
)

quarterly_harvard_only = make_quarterly_dataset(
    ads_harvard_only
)


# Save all three quarterly datasets.
quarterly_original.to_csv(
    "week_three_quarterly_sentiment.csv",
    index=False
)

quarterly_no_stopwords.to_csv(
    "week_three_quarterly_sentiment_stopwords.csv",
    index=False
)

quarterly_harvard_only.to_csv(
    "week_three_quarterly_sentiment_harvard_only.csv",
    index=False
)

print("\nALL THREE QUARTERLY DATASETS CREATED")