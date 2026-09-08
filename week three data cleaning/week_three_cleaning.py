import pandas as pd
df = pd.read_csv(
    "../Initial file imports/The Economist Historical Advertisements – Master Dataset.csv")
   
# Variables in the dataset:

# Filename:
# Name of the file for each ad. We do not need it so we are removing it, especailly as it is reduandant given there is already a brand column.

# URLs_TheEconomistPageScans:
#This links back to the original scanned Economist page, but because we have the OCR, we don't need it

# DateOfIssue:
# This tells us when the ad was published. We will keep this as it allows us to use this data as a time-series data.
# Bounding Box:  tell us where the ad was located on the scanned page. It is not relevant to our topic so we are removing it.

# Brand:
# This tells us which company or advertiser the ad came from. We are keeping this because it could be useful later if we want to compare different companies or types of advertisers.

# Brand is generic (e.g. 'Notices'):
# This tells us whether something is a real brand or more of a generic notice. We are keeping this because it may help us filter what type of ad it is

# OCR_GoogleVision_original:
# This is the text that Google OCR pulled from the advertisement image. This is needed as this is how we will do our sentiment analysis

# Categories (from Google Natural Language API classify_text based on OCR text):
#Google did this to try to figure out what the add is about. We will not be using it, as we are wanting to primarily to examine the langauge in the ads themselves
#Limit the data to the columns that we want to use    # Keep only the columns that are useful for our research question.
cleaned_ads = df[
    [
        "DateOfIssue",
        "Brand",
        "Brand is generic (e.g. 'Notices')",
        "OCR_GoogleVision_original"
    ]
].copy()
# coerce so that way if the date isnt there it wont create issues; changing the format so it is easier to work with
cleaned_ads["DateOfIssue"] = pd.to_datetime(
    cleaned_ads["DateOfIssue"],
    errors="coerce"
)
#setting it to quarters that way it lines up with the reporting
cleaned_ads["Year"] = cleaned_ads["DateOfIssue"].dt.year
cleaned_ads["Quarter"] = cleaned_ads["DateOfIssue"].dt.quarter
cleaned_ads["Year_Quarter"] = cleaned_ads["DateOfIssue"].dt.to_period("Q")
print(
    cleaned_ads[
        ["DateOfIssue", "Year", "Quarter", "Year_Quarter"]
    ].head()
)
#Filter to years after 1948
post_1948_ads = cleaned_ads[
    cleaned_ads["Year"] >= 1948
].copy()


#only branded advertisements 
branded_ads = post_1948_ads[
    post_1948_ads["Brand is generic (e.g. 'Notices')"] == False
].copy()
branded_ads.to_csv(
    "week_three_cleaned_ads.csv",
    index=False
)

harvard = pd.read_excel(
    "../Initial file imports/inquireraugmented.xls",
    engine="xlrd")
# Despite out main unit of analysis being just the amount of positive and negative words used in the advertisment, the dataset includes many other categories that we won't scrub out now incase we want to use them later. Below is a list of the ones that we are choosing to keep
# Entry:
# The actual word or dictionary entry.
# Positiv:
# Words with a positive meaning or evaluation.
# Negativ:
# Words with a negative meaning or evaluation.
# Affil:
# Language related to affiliation, relationships, cooperation, or social connection.
# Hostile:
# Language related to hostility, conflict, aggression, or opposition.
# Power:
# Language related to power, authority, influence, or control.
# Strong:
# Words associated with strength, intensity, confidence, or force.
# Weak:
# Words associated with weakness, uncertainty, vulnerability, or lack of strength.
# Pleasur:
# Language related to pleasure, enjoyment, satisfaction, or positive experience.
# Pain:
# Language related to pain, discomfort, suffering, or negative experience.
# Active:
# Language associated with action, activity, movement, or doing something.
# Passive:
# Language associated with passivity, inactivity, or having something done to someone.
# EMOT:
# Words associated with emotion or emotional expression.
# Virtue:
# Words associated with positive moral qualities or socially desirable behavior.
# Vice:
# Words associated with negative moral qualities or socially undesirable behavior.
# Role:
# Words referring to social roles, positions, or types of people.
# Solve:
# Language related to solving problems or finding solutions.
# Goal:
# Language related to goals, aims, objectives, or desired outcomes.
# Means:
# Language related to methods, resources, or ways of achieving something.
# MeansLw:
# A related lower-level means category that captures language about methods or resources used to accomplish goals.
# Complet:
# Language associated with completion, accomplishment, finishing, or fulfillment.
harvard_columns = [
    "Entry",
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
cleaned_harvard = harvard[harvard_columns].copy()
cleaned_harvard.to_csv(
    "week_three_cleaned_harvard.csv",
    index=False
)
