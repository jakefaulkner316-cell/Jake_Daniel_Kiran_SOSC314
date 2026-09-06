# Import pandas so we can work with the US economic data.
import pandas as pd

# Load the FRED economic dataset.
economic_data = pd.read_csv(
    "../Initial file imports/fredgraph.csv"
)

# Turn the observation date into a date format that Python can work with.
economic_data["observation_date"] = pd.to_datetime(
    economic_data["observation_date"]
)

# Create year, quarter, and combined year-quarter variables.
economic_data["Year"] = economic_data["observation_date"].dt.year
economic_data["Quarter"] = economic_data["observation_date"].dt.quarter
economic_data["Year_Quarter"] = (
    economic_data["observation_date"].dt.to_period("Q")
)

# Calculate the percent change in real GDP from one quarter to the next.
economic_data["GDP_Growth"] = (
    economic_data["GDPC1"].pct_change() * 100
)

# Mark whether GDP declined in each quarter.
economic_data["Negative_GDP"] = (
    economic_data["GDP_Growth"] < 0
)

# Define a technical recession as two consecutive quarters
# of negative real GDP growth.
economic_data["Technical_Recession"] = (
    economic_data["Negative_GDP"]
    & economic_data["Negative_GDP"].shift(1)
)

# Create a readable recession label.
economic_data["Recession_Status"] = economic_data[
    "Technical_Recession"
].map({
    True: "Recession",
    False: "Not Recession"
})

# Create a cleaner table with only the main variables.
recession_table = economic_data[
    [
        "Year_Quarter",
        "GDPC1",
        "GDP_Growth",
        "Recession_Status"
    ]
]

print("\nUS RECESSION TABLE:")
print(recession_table)

# Save the table as a CSV file.
recession_table.to_csv(
    "us_recession_table.csv",
    index=False
)