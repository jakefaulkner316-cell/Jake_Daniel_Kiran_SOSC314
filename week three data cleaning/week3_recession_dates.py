
# This script builds a variable from the official NBER chronology, which is different
# from the technical rule of two-consecutive-recessionary-quarters we implemented. 
#As a result, this new form of flagging recessions uncovers two recessionary periods that
# were not flagged before: 2001 and 1960-61. In total, these two variables (GDPC1 and GDP_Growth, which 
#are preserved in the final .csv to enable future comparisons) disagree on the definition of 6 quartes out of the 248.


#This script builds the recession variable our advertisement data is
# matched against. It is data construction and measurement. No models
# are fitted here. We make here a measurement comparison, not a model.

# "us economic data.py" defines a recession as two consecutive quarters
# of negative real GDP growth. That is the popular "technical recession"
# rule, and it is NOT what the NBER uses. The NBER dating committee
# looks at employment, real income, industrial production and sales, not
# GDP alone, and it dates turning points by month rather than quarter.
#
# The two rules disagree. The 2001 recession is an official NBER
# recession but has no two consecutive quarters of negative real GDP
# growth, so the technical rule misses it entirely. The 1960-61
# recession has the same problem.
#
# The output has the same column names as us_recession_table.csv
# (Year_Quarter, GDPC1, GDP_Growth, Recession_Status), so switching
# week_three_analysis.py over to NBER dating is a one-line change:
#
#   economic = pd.read_csv("nber_recession_table.csv")
#
# Recession_Status uses the same "Recession" / "Not Recession" text
# labels, so the groupby and the .loc lookups in that script keep
# working untouched.
#
# Output: nber_recession_table.csv

import pandas as pd

# STEP 1: The official NBER business cycle chronology
# Source: NBER Business Cycle Dating Committee
# https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
# We hardcode these rather than downloading FRED's USREC series. There are only twelve post-war recessions, the list has 
# not changed since 2021, and hardcoding means the script runs locally.
#
# Each pair is (peak month, trough month).
# The 'peak' is the last month of the expansion.
# The 'trough' is the last month of the contraction.

NBER_CYCLES = [
    ("1948-11", "1949-10"),
    ("1953-07", "1954-05"),
    ("1957-08", "1958-04"),
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

# STEP 2: Monthly indicator
# MEASUREMENT DECISION 1: does the peak month itself count?
#
# The NBER says the peak is the last month of expansion. Strictly,
# the recession starts the month AFTER the peak and runs through the
# trough. FRED's USREC series, on the other hand, codes the peak month itself as
# recessionary. The difference is one month per turning point, which
# almost never survives aggregation to quarters, but we name it as a
# variable so the choice is visible and reversible rather than buried inside the loop used in other previous analyses.

PEAK_MONTH_IS_RECESSION = False

# Monthly calendar for the whole analysis window. 1948 matches the advertisement sample; 2014 is the last year of The Economist dataset.
months = pd.period_range(start="1948-01", end="2014-12", freq="M")

recession_months = pd.DataFrame({"Month": months})
recession_months["NBER_Recession"] = False

for peak_text, trough_text in NBER_CYCLES:
    peak = pd.Period(peak_text, freq="M")
    trough = pd.Period(trough_text, freq="M")
    if PEAK_MONTH_IS_RECESSION:
        start = peak
    else:
        start = peak + 1
    in_this_recession = (
        (recession_months["Month"] >= start)
        & (recession_months["Month"] <= trough)
    )

    # Combine with "|" so flagging one recession does not erase those already flagged.
    recession_months["NBER_Recession"] = (
        recession_months["NBER_Recession"] | in_this_recession
    )

print("MONTHLY INDICATOR")
print("Total months:", len(recession_months))
print("Recession months:", recession_months["NBER_Recession"].sum())
print(
    "Share of months in recession:",
    round(recession_months["NBER_Recession"].mean(), 4)
)

# STEP 3: Collapse to quarters
# MEASUREMENT DECISION 2: how to classify a quarter that straddles a turning point
#
# Our advertisement data is quarterly, so the indicator must be too. But
# a quarter can hold 0, 1, 2 or 3 recession months. We count them and
# build two versions:
#   Recession_Status  : 2 or 3 recession months  (majority, our main measure)
#   Recession_Any     : 1 or more recession months
# Majority is the primary measure because it calls a quarter
# recessionary only when most of it was. Recession_Any is more generous and catches transition quarters. 
# Keeping both doesn't compromise anything and offers a new opportunity of robustness checks.

recession_months["Quarter"] = (
    recession_months["Month"].dt.to_timestamp().dt.to_period("Q")
)

quarterly = (
    recession_months
    .groupby("Quarter")
    .agg(
        Recession_Months=("NBER_Recession", "sum"),
        Months_In_Quarter=("NBER_Recession", "size")
    )
    .reset_index()
)

quarterly = quarterly.rename(columns={"Quarter": "Year_Quarter"})
quarterly["Recession_Majority"] = quarterly["Recession_Months"] >= 2
quarterly["Recession_Any"] = quarterly["Recession_Months"] >= 1

# Text labels matching us_recession_table.csv exactly, so the existing
# analysis script needs no changes beyond the input filename.
quarterly["Recession_Status"] = quarterly["Recession_Majority"].map({
    True: "Recession",
    False: "Not Recession"
})

quarterly["Recession_Status_Any"] = quarterly["Recession_Any"].map({
    True: "Recession",
    False: "Not Recession"
})

print()
print("QUARTERLY INDICATOR")
print("Total quarters:", len(quarterly))
print("Recession quarters (majority rule):", quarterly["Recession_Majority"].sum())
print("Recession quarters (any-month rule):", quarterly["Recession_Any"].sum())
print(
    "Quarters where the two rules disagree:",
    (quarterly["Recession_Majority"] != quarterly["Recession_Any"]).sum()
)

# STEP 4: Attach GDP and compare against the rule

# We bring in the same FRED series the old script used, for two reasons.
# First, it keeps GDPC1 and GDP_Growth in the output so the schema
# matches us_recession_table.csv. Second, it lets us document exactly
# which quarters the two definitions disagree on, which is the evidence for why we switched.

try:
    economic_data = pd.read_csv("../Initial file imports/fredgraph.csv")

    economic_data["observation_date"] = pd.to_datetime(
        economic_data["observation_date"]
    )

    economic_data["Year_Quarter"] = (
        economic_data["observation_date"].dt.to_period("Q")
    )

    economic_data["GDP_Growth"] = economic_data["GDPC1"].pct_change() * 100

    economic_data["Negative_GDP"] = economic_data["GDP_Growth"] < 0

    economic_data["Technical_Recession"] = (
        economic_data["Negative_GDP"]
        & economic_data["Negative_GDP"].shift(1)
    )

    gdp_side = economic_data[
        ["Year_Quarter", "GDPC1", "GDP_Growth", "Technical_Recession"]
    ]

    quarterly = quarterly.merge(gdp_side, on="Year_Quarter", how="left")

    both_present = quarterly.dropna(subset=["Technical_Recession"])

    agreement = (
        both_present["Recession_Majority"]
        == both_present["Technical_Recession"]
    )

    print()
    print("NBER RULE VERSUS TECHNICAL RULE")
    print("Quarters where both definitions exist:", len(both_present))
    print("Quarters where they agree:", agreement.sum())
    print("Quarters where they disagree:", (~agreement).sum())

    # Quarters the NBER calls recessionary that the GDP rule misses.
    missed = both_present[
        both_present["Recession_Majority"]
        & (both_present["Technical_Recession"] == False)
    ]

    print()
    print("QUARTERS THE TECHNICAL RULE MISSES:")
    print(missed["Year_Quarter"].tolist())

    # And the reverse.
    extra = both_present[
        (both_present["Recession_Majority"] == False)
        & both_present["Technical_Recession"]
    ]

    print()
    print("QUARTERS THE TECHNICAL RULE ADDS:")
    print(extra["Year_Quarter"].tolist())

except FileNotFoundError:
    print()
    print("fredgraph.csv not found. Skipping the technical-rule comparison.")
    print("The NBER indicator is still complete and usable.")
    quarterly["GDPC1"] = pd.NA
    quarterly["GDP_Growth"] = pd.NA

# STEP 5: Save

output_columns = [
    "Year_Quarter",
    "GDPC1",
    "GDP_Growth",
    "Recession_Status",
    "Recession_Status_Any",
    "Recession_Months",
    "Recession_Majority",
    "Recession_Any"
]

recession_table = quarterly[output_columns]

recession_table.to_csv("nber_recession_table.csv", index=False)

print()
print("SAVED: nber_recession_table.csv")
print()
print("To use it, change one line in week_three_analysis.py:")
print('  economic = pd.read_csv("nber_recession_table.csv")')
print()
print(recession_table.head(12))
