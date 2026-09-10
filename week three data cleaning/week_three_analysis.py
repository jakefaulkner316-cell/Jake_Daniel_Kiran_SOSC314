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

combined_original = quarterly_original.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

combined_no_stopwords = quarterly_no_stopwords.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

combined_harvard_only = quarterly_harvard_only.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)


# Compare the average Harvard category ratios between recession
# and non-recession quarters.
def recession_comparison(data):

    comparison = (
        data
        .groupby("Recession_Status")[ratio_columns]
        .mean()
    )

    return comparison


original_comparison = recession_comparison(
    combined_original
)

no_stopwords_comparison = recession_comparison(
    combined_no_stopwords
)

harvard_only_comparison = recession_comparison(
    combined_harvard_only
)
# Compare positive and negative sentiment across the three methods.
sentiment_graph = pd.DataFrame({
    "All Words Positive": [
        original_comparison.loc["Not Recession", "Positiv_Ratio"] * 100,
        original_comparison.loc["Recession", "Positiv_Ratio"] * 100
    ],
    "All Words Negative": [
        original_comparison.loc["Not Recession", "Negativ_Ratio"] * 100,
        original_comparison.loc["Recession", "Negativ_Ratio"] * 100
    ],
    "No Stopwords Positive": [
        no_stopwords_comparison.loc["Not Recession", "Positiv_Ratio"] * 100,
        no_stopwords_comparison.loc["Recession", "Positiv_Ratio"] * 100
    ],
    "No Stopwords Negative": [
        no_stopwords_comparison.loc["Not Recession", "Negativ_Ratio"] * 100,
        no_stopwords_comparison.loc["Recession", "Negativ_Ratio"] * 100
    ],
    "Harvard Only Positive": [
        harvard_only_comparison.loc["Not Recession", "Positiv_Ratio"] * 100,
        harvard_only_comparison.loc["Recession", "Positiv_Ratio"] * 100
    ],
    "Harvard Only Negative": [
        harvard_only_comparison.loc["Not Recession", "Negativ_Ratio"] * 100,
        harvard_only_comparison.loc["Recession", "Negativ_Ratio"] * 100
    ]
}, index=["Not Recession", "Recession"])

sentiment_graph.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Positive and Negative Advertising Language by Recession Status")
plt.xlabel("Economic Condition")
plt.ylabel("Percent of Words")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "sentiment_methods_recession_comparison.png",
    dpi=300
)

plt.close()

difference_graph = pd.DataFrame({
    "All Words": [
        (
            original_comparison.loc["Recession", "Positiv_Ratio"]
            - original_comparison.loc["Not Recession", "Positiv_Ratio"]
        ) * 100,
        (
            original_comparison.loc["Recession", "Negativ_Ratio"]
            - original_comparison.loc["Not Recession", "Negativ_Ratio"]
        ) * 100
    ],

    "No Stopwords": [
        (
            no_stopwords_comparison.loc["Recession", "Positiv_Ratio"]
            - no_stopwords_comparison.loc["Not Recession", "Positiv_Ratio"]
        ) * 100,
        (
            no_stopwords_comparison.loc["Recession", "Negativ_Ratio"]
            - no_stopwords_comparison.loc["Not Recession", "Negativ_Ratio"]
        ) * 100
    ],

    "Harvard Only": [
        (
            harvard_only_comparison.loc["Recession", "Positiv_Ratio"]
            - harvard_only_comparison.loc["Not Recession", "Positiv_Ratio"]
        ) * 100,
        (
            harvard_only_comparison.loc["Recession", "Negativ_Ratio"]
            - harvard_only_comparison.loc["Not Recession", "Negativ_Ratio"]
        ) * 100
    ]
}, index=["Positive", "Negative"])


difference_graph.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Change in Advertising Sentiment During Recessions"
)

plt.xlabel("Sentiment Category")
plt.ylabel("Recession Difference (Percentage Points)")
plt.xticks(rotation=0)

plt.axhline(
    y=0,
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    "sentiment_recession_difference_by_method.png",
    dpi=300
)

plt.close()

# Short meanings for the Harvard categories.
category_meanings = {
    "Positiv_Ratio": "Positive language",
    "Negativ_Ratio": "Negative language",
    "Affil_Ratio": "Affiliation and relationships",
    "Hostile_Ratio": "Hostility and conflict",
    "Power_Ratio": "Power and control",
    "Strong_Ratio": "Strength",
    "Weak_Ratio": "Weakness",
    "Pleasur_Ratio": "Pleasure",
    "Pain_Ratio": "Pain",
    "Active_Ratio": "Active language",
    "Passive_Ratio": "Passive language",
    "EMOT_Ratio": "Emotion",
    "Virtue_Ratio": "Virtue",
    "Vice_Ratio": "Vice",
    "Role_Ratio": "Social roles",
    "Solve_Ratio": "Problem solving",
    "Goal_Ratio": "Goals",
    "Means_Ratio": "Means or methods",
    "MeansLw_Ratio": "Lower-level means or methods",
    "Complet_Ratio": "Completion or accomplishment"
}


# Create a clean comparison table for one preprocessing method.
def make_category_table(comparison):

    table = comparison.transpose().reset_index()

    table["Not Recession (%)"] = (
        table["Not Recession"] * 100
    ).round(3)

    table["Recession (%)"] = (
        table["Recession"] * 100
    ).round(3)

    table["Difference (pp)"] = (
        (table["Recession"] - table["Not Recession"]) * 100
    ).round(3)

    table["Meaning"] = (
        table["index"].map(category_meanings)
    )

    table["Category"] = (
        table["index"]
        .str.replace("_Ratio", "", regex=False)
    )

    return table[
        [
            "Category",
            "Not Recession (%)",
            "Recession (%)",
            "Difference (pp)",
            "Meaning"
        ]
    ]


# Create the three tables.
all_words_table = make_category_table(
    original_comparison
)

no_stopwords_table = make_category_table(
    no_stopwords_comparison
)

harvard_only_table = make_category_table(
    harvard_only_comparison
)
def save_table_image(table_data, title, filename):

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.axis("off")

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="left"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    plt.title(
        title,
        pad=20
    )

    plt.savefig(
        filename,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()


save_table_image(
    all_words_table,
    "Harvard Category Comparison: All Words",
    "harvard_table_all_words.png"
)

save_table_image(
    no_stopwords_table,
    "Harvard Category Comparison: No Stopwords",
    "harvard_table_no_stopwords.png"
)

save_table_image(
    harvard_only_table,
    "Harvard Category Comparison: Harvard Only",
    "harvard_table_harvard_only.png"
)