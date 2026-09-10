import pandas as pd


# Load the quarterly advertisement sentiment data.
sentiment = pd.read_csv(
    "week_three_quarterly_sentiment.csv"
)
# Load economic/recession data.
economic = pd.read_csv(
    "../early statistical analysis/us_recession_table.csv"
)

# Merge the quarterly sentiment data with the US economic data.
combined = sentiment.merge(
    economic,
    on="Year_Quarter",
    how="inner"
)

# Check that the merge worked.
#print("\nCOMBINED DATA:")
#print(combined.head())

#print("\nNUMBER OF QUARTERS:")
#print(len(combined))
# Compare average positive and negative advertising sentiment
# between recession and non-recession quarters.
recession_comparison = (
    combined
    .groupby("Recession_Status")[
        ["Positiv_Ratio", "Negativ_Ratio"]
    ]
    .mean()
)

#print("\nAVERAGE SENTIMENT BY RECESSION STATUS:")
#print(recession_comparison)
# table.
recession_comparison_table = recession_comparison.reset_index()

# Calculate the difference between recession and non-recession sentiment.
positive_difference = (
    recession_comparison.loc["Recession", "Positiv_Ratio"]
    - recession_comparison.loc["Not Recession", "Positiv_Ratio"]
)

negative_difference = (
    recession_comparison.loc["Recession", "Negativ_Ratio"]
    - recession_comparison.loc["Not Recession", "Negativ_Ratio"]
)

print("\nDIFFERENCE IN AVERAGE SENTIMENT:")
print("Positive difference:", positive_difference)
print("Negative difference:", negative_difference)

# Save the comparison table.
recession_comparison_table.to_csv(
    "recession_sentiment_comparison.csv",
    index=False
)
# Compare all Harvard category ratios between recession and non-recession quarters.
ratio_columns = [
    column for column in combined.columns
    if column.endswith("_Ratio")
]

all_category_comparison = (
    combined
    .groupby("Recession_Status")[ratio_columns]
    .mean()
    .transpose()
)

# Calculate the recession minus non-recession difference for every category.
all_category_comparison["Difference"] = (
    all_category_comparison["Recession"]
    - all_category_comparison["Not Recession"]
)

# Sort from the largest positive difference to the largest negative difference.
all_category_comparison = all_category_comparison.sort_values(
    "Difference",
    ascending=False
)

print("\nALL HARVARD CATEGORY DIFFERENCES:")
print(all_category_comparison)

all_category_comparison.to_csv(
    "all_harvard_recession_comparison.csv"
)
# Create a cleaner table for the Harvard category recession comparison.
harvard_recession_table = (
    all_category_comparison
    .reset_index()
    .rename(columns={
        "index": "Harvard_Category",
        "Not Recession": "Not_Recession_Average",
        "Recession": "Recession_Average"
    })
)

# Round the values so the table is easier to read.
harvard_recession_table[
    [
        "Not_Recession_Average",
        "Recession_Average",
        "Difference"
    ]
] = harvard_recession_table[
    [
        "Not_Recession_Average",
        "Recession_Average",
        "Difference"
    ]
].round(6)

print("\nHARVARD CATEGORY RECESSION COMPARISON TABLE:")
print(harvard_recession_table)

# Save the table as a CSV file.
harvard_recession_table.to_csv(
    "harvard_recession_comparison_table.csv",
    index=False
)
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
    "Virtue_Ratio": "Virtue or positive moral language",
    "Vice_Ratio": "Vice or negative moral language",
    "Role_Ratio": "Social roles",
    "Solve_Ratio": "Problem solving",
    "Goal_Ratio": "Goals",
    "Means_Ratio": "Means or methods",
    "MeansLw_Ratio": "Lower-level means or methods",
    "Complet_Ratio": "Completion or accomplishment"
}

# Make a simple table showing how each category changes during recessions.
recession_table = all_category_comparison.reset_index()

recession_table["Difference_Percent"] = (
    recession_table["Difference"] * 100
).round(4)

recession_table["Meaning"] = (
    recession_table["index"].map(category_meanings)
)

recession_table["Category"] = (
    recession_table["index"]
    .str.replace("_Ratio", "", regex=False)
)

recession_table = recession_table[
    [
        "Category",
        "Difference_Percent",
        "Meaning"
    ]
]

print("\nRECESSION CATEGORY DIFFERENCES:")
print(recession_table)

recession_table.to_csv(
    "recession_category_differences.csv",
    index=False
)
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

recession_table = all_category_comparison.reset_index()

recession_table["Not Recession (%)"] = (
    recession_table["Not Recession"] * 100
).round(3)

recession_table["Recession (%)"] = (
    recession_table["Recession"] * 100
).round(3)

recession_table["Difference (pp)"] = (
    recession_table["Difference"] * 100
).round(3)

recession_table["Meaning"] = (
    recession_table["index"].map(category_meanings)
)

recession_table["Category"] = (
    recession_table["index"]
    .str.replace("_Ratio", "", regex=False)
)

recession_table = recession_table[
    [
        "Category",
        "Not Recession (%)",
        "Recession (%)",
        "Difference (pp)",
        "Meaning"
    ]
]

# Turn the table into an image.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

table = ax.table(
    cellText=recession_table.values,
    colLabels=recession_table.columns,
    loc="center",
    cellLoc="left"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

plt.title(
    "Harvard Category Use During Recession and Non-Recession Quarters",
    pad=20
)

plt.savefig(
    "harvard_recession_category_table_percentages.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()
#There is an issue with all of this in that I included grammer words like the, of, and...etc; therefore, now I am removing them and will be rerunning everything. 
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))