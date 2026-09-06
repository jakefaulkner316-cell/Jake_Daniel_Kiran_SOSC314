# Import panda that we can work with the data.
import pandas as pd
#Import matlab that we we can graph
import matplotlib.pyplot as plt

df = pd.read_csv(
    "../Initial file imports/The Economist Historical Advertisements – Master Dataset.csv",)
print(df.head())
#Shows the names of the columns 
print(df.columns)

# Variables in the dataset:

# Filename:
# Name of the file for each ad. We do not need it so we are removing it, especailly as it is reduandant given there is already a brand column.

# URLs_TheEconomistPageScans:
#This links back to the original scanned Economist page, but because we have the OCR, we don't need it

# DateOfIssue:
# This tells us when the ad was published. We will keep this as it allows us to use this data as a time-series data.
# Bounding Box relative X1, Y1, X2, Y2:
# These tell us where the ad was located on the scanned page. It is not relevant to our topic so we are removing it.

# Brand:
# This tells us which company or advertiser the ad came from. We are keeping this because it could be useful later if we want to compare different companies or types of advertisers.

# Brand is generic (e.g. 'Notices'):
# This tells us whether something is a real brand or more of a generic notice. We are keeping this because it may help us filter what type of ad it is

# OCR_GoogleVision_original:
# This is the text that Google OCR pulled from the advertisement image. This is needed as this is how we will do our sentiment analysis

# Categories (from Google Natural Language API classify_text based on OCR text):
#Google did this to try to figure out what the add is about. We will not be using it, as we are wanting to primarily to examine the langauge in the ads themselves
#Limit the data to the columns that we want to use
important_columns = df[
    [
        "DateOfIssue",
        "Brand",
        "Brand is generic (e.g. 'Notices')",
        "OCR_GoogleVision_original"
    ]
]

# Show the first five rows of only these selected columns.
print(important_columns.head())

# Turn date of issue into a set of years and quarters, as in the US we define recession by a period of 2 quarters or more of real GDP lowering
# We use errors="coerce" so that if there are any broken or unreadable dates, it will be marked as missing instead of stopping the entire program.
important_columns["DateOfIssue"] = pd.to_datetime(
    important_columns["DateOfIssue"],
    errors="coerce"
)
# Create a Year variable:
important_columns["Year"] = important_columns["DateOfIssue"].dt.year

# Create a Quarter variable (as definitions require that we have it when we compare with economic data)
important_columns["Quarter"] = important_columns["DateOfIssue"].dt.quarter

# Create a combined Year-Quarter variable.
# For example: 2008Q4
important_columns["Year_Quarter"] = (
    important_columns["DateOfIssue"].dt.to_period("Q")
)
# Check the out put:
print(
    important_columns[
        ["DateOfIssue", "Year", "Quarter", "Year_Quarter"]
    ].head()
)
# Examine how many advertisements appear in each year across the entire dataset.
ads_per_year = important_columns.groupby("Year").size()

# Print the number of advertisements for each year.
print("\nADVERTISEMENTS PER YEAR:")
print(ads_per_year)


# Make a graph showing the number of advertisements year over year published in the dataset.
ads_per_year.plot()

plt.title("Advertisements per Year, 1843-2014")
plt.xlabel("Year")
plt.ylabel("Number of Advertisements")

plt.savefig("ads_per_year.png")

# Close the yearly graph that way it does not overlap with the next one
plt.close()


# Count how many advertisements appear in each quarter
ads_per_quarter = important_columns.groupby("Year_Quarter").size()

# Print the number of advertisements for each quarter.
print("\nADVERTISEMENTS PER QUARTER:")
print(ads_per_quarter)


# Make a graph showing the number of advertisements published in each quarter across the full dataset.
ads_per_quarter.plot()

plt.title("Advertisements per Quarter, 1843-2014")
plt.xlabel("Year and Quarter")
plt.ylabel("Number of Advertisements")

# Save the quarterly graph as an image file.
plt.savefig("ads_per_quarter.png")
plt.close()
# Create yearly and quarterly graphs for only the post-1948 advertisements.
post_1948 = important_columns[
    important_columns["Year"] >= 1948
]

ads_per_year_post_1948 = post_1948.groupby("Year").size()

ads_per_year_post_1948.plot()

plt.title("Advertisements per Year, 1948-2014")
plt.xlabel("Year")
plt.ylabel("Number of Advertisements")

plt.savefig("ads_per_year_1948_2014.png")
plt.close()


ads_per_quarter_post_1948 = post_1948.groupby("Year_Quarter").size()

ads_per_quarter_post_1948.plot()

plt.title("Advertisements per Quarter, 1948-2014")
plt.xlabel("Year and Quarter")
plt.ylabel("Number of Advertisements")

plt.savefig("ads_per_quarter_1948_2014.png")
plt.close()