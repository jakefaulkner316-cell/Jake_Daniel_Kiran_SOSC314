# =========================================================
# WEEK 5 REGRESSION ANALYSIS:
# BASELINE RECESSION-SENTIMENT REGRESSIONS
# =========================================================
#
# PURPOSE:
# This script introduces the first formal regression models
# for the project.
#
# Until this point, the project has primarily compared average
# sentiment during recession and non-recession quarters.
#
# This script estimates that relationship using ordinary least
# squares (OLS) regression.
#
# MAIN QUESTION:
#
# Is recession status associated with quarterly positive or
# negative advertising sentiment?
#
# DICTIONARIES TESTED:
#
# 1. Harvard General Inquirer
# 2. Loughran-McDonald
#
# OUTCOMES TESTED:
#
# 1. Positive sentiment
# 2. Negative sentiment
#
# BASELINE MODEL:
#
# Sentiment = intercept + recession indicator + error
#
# The recession coefficient tells us how much average quarterly
# sentiment differs during recession quarters relative to
# non-recession quarters.
#
# IMPORTANT:
#
# These are the baseline OLS models.
#
# A separate Week 5 diagnostic file will later test:
#
# - heteroskedasticity
# - robust standard errors
# - whether statistical inference changes
#
# OUTPUTS:
#
# csv outputs/
#   baseline_recession_sentiment_regressions.csv
#
# robustness tables/
#   baseline_regression_results.txt
#
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

import pandas as pd
import statsmodels.api as sm

from pathlib import Path


# ---------------------------------------------------------
# DEFINE FILE LOCATIONS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

WEEK5_DIR = SCRIPT_DIR.parent

REPO_ROOT = WEEK5_DIR.parent


WEEK4_AGGREGATION_DIR = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
    / "week4_aggregations"
)


HARVARD_FILE = (
    WEEK4_AGGREGATION_DIR
    / "quarterly_harvard_macro.csv"
)


LM_FILE = (
    WEEK4_AGGREGATION_DIR
    / "quarterly_loughran_macro.csv"
)


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = (
    SCRIPT_DIR
    / "csv outputs"
)

TABLE_OUTPUT_DIR = (
    SCRIPT_DIR
    / "robustness tables"
)


CSV_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD QUARTERLY SENTIMENT DATA
# ---------------------------------------------------------

harvard = pd.read_csv(
    HARVARD_FILE
)

lm = pd.read_csv(
    LM_FILE
)


# ---------------------------------------------------------
# INSPECT DATA
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "HARVARD QUARTERLY DATA"
)

print("=" * 80)

print(
    harvard.head()
)

print(
    "\nHarvard columns:"
)

print(
    harvard.columns.tolist()
)


print("\n" + "=" * 80)

print(
    "LOUGHRAN-MCDONALD QUARTERLY DATA"
)

print("=" * 80)

print(
    lm.head()
)

print(
    "\nLoughran-McDonald columns:"
)

print(
    lm.columns.tolist()
)


# ---------------------------------------------------------
# CREATE NUMERIC RECESSION INDICATOR
# ---------------------------------------------------------
#
# Recession = 1
# Not Recession = 0
#
# This allows recession status to be used directly as an
# independent variable in the regression.
# ---------------------------------------------------------

harvard["Recession"] = (
    harvard["Recession_Status"]
    .map(
        {
            "Recession": 1,
            "Not Recession": 0
        }
    )
)


lm["Recession"] = (
    lm["Recession_Status"]
    .map(
        {
            "Recession": 1,
            "Not Recession": 0
        }
    )
)


# ---------------------------------------------------------
# CHECK RECESSION COUNTS
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "RECESSION QUARTER COUNTS"
)

print("=" * 80)

print(
    "\nHarvard:"
)

print(
    harvard["Recession"].value_counts()
)

print(
    "\nLoughran-McDonald:"
)

print(
    lm["Recession"].value_counts()
)


# ---------------------------------------------------------
# FUNCTION TO RUN BASELINE OLS REGRESSION
# ---------------------------------------------------------

def run_baseline_regression(
    data,
    outcome,
    dictionary_name
):

    # Independent variable:
    # recession status

    X = data[
        [
            "Recession"
        ]
    ]

    # Add intercept to regression.

    X = sm.add_constant(
        X
    )

    # Dependent variable:
    # positive or negative sentiment.

    y = data[
        outcome
    ]


    # Estimate ordinary least squares regression.

    model = sm.OLS(
        y,
        X,
        missing="drop"
    ).fit()


    # Extract recession coefficient.

    coefficient = (
        model.params[
            "Recession"
        ]
    )


    # Extract ordinary OLS standard error.

    standard_error = (
        model.bse[
            "Recession"
        ]
    )


    # Extract p-value.

    p_value = (
        model.pvalues[
            "Recession"
        ]
    )


    # Extract confidence interval.

    confidence_interval = (
        model.conf_int()
        .loc[
            "Recession"
        ]
    )


    result = {
        "Dictionary":
            dictionary_name,

        "Outcome":
            outcome,

        "Observations":
            int(
                model.nobs
            ),

        "Recession_Coefficient":
            coefficient,

        "OLS_Standard_Error":
            standard_error,

        "OLS_P_Value":
            p_value,

        "CI_95_Lower":
            confidence_interval.iloc[0],

        "CI_95_Upper":
            confidence_interval.iloc[1],

        "R_Squared":
            model.rsquared
    }


    return model, result


# ---------------------------------------------------------
# RUN FOUR BASELINE REGRESSIONS
# ---------------------------------------------------------
#
# Harvard Positive
# Harvard Negative
# LM Positive
# LM Negative
# ---------------------------------------------------------

regression_results = []

models = {}


# Harvard positive

model, result = run_baseline_regression(
    harvard,
    "Pos",
    "Harvard"
)

models[
    "Harvard Positive"
] = model

regression_results.append(
    result
)


# Harvard negative

model, result = run_baseline_regression(
    harvard,
    "Neg",
    "Harvard"
)

models[
    "Harvard Negative"
] = model

regression_results.append(
    result
)


# Loughran-McDonald positive

model, result = run_baseline_regression(
    lm,
    "Pos",
    "Loughran-McDonald"
)

models[
    "Loughran-McDonald Positive"
] = model

regression_results.append(
    result
)


# Loughran-McDonald negative

model, result = run_baseline_regression(
    lm,
    "Neg",
    "Loughran-McDonald"
)

models[
    "Loughran-McDonald Negative"
] = model

regression_results.append(
    result
)


# ---------------------------------------------------------
# CREATE RESULTS DATAFRAME
# ---------------------------------------------------------

regression_table = pd.DataFrame(
    regression_results
)


# Convert coefficients from proportions into percentage points
# so they are directly comparable to the Week 4 results.

percentage_point_columns = [
    "Recession_Coefficient",
    "OLS_Standard_Error",
    "CI_95_Lower",
    "CI_95_Upper"
]


regression_table[
    percentage_point_columns
] = (
    regression_table[
        percentage_point_columns
    ]
    * 100
)


regression_table = regression_table.rename(
    columns={
        "Recession_Coefficient":
            "Recession_Coefficient_pp",

        "OLS_Standard_Error":
            "OLS_Standard_Error_pp",

        "CI_95_Lower":
            "CI_95_Lower_pp",

        "CI_95_Upper":
            "CI_95_Upper_pp"
    }
)


# ---------------------------------------------------------
# PRINT BASELINE REGRESSION RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "BASELINE OLS RECESSION-SENTIMENT REGRESSIONS"
)

print("=" * 80)

print(
    regression_table.round(
        6
    ).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# PRINT FULL MODEL SUMMARIES
# ---------------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 80)

    print(
        name.upper()
    )

    print("=" * 80)

    print(
        model.summary()
    )


# ---------------------------------------------------------
# SAVE REGRESSION TABLE
# ---------------------------------------------------------

csv_output = (
    CSV_OUTPUT_DIR
    / "baseline_recession_sentiment_regressions.csv"
)


regression_table.to_csv(
    csv_output,
    index=False
)


print(
    "\nSAVED:"
)

print(
    csv_output
)


# ---------------------------------------------------------
# SAVE HUMAN-READABLE RESULTS
# ---------------------------------------------------------

text_output = (
    TABLE_OUTPUT_DIR
    / "baseline_regression_results.txt"
)


with open(
    text_output,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 BASELINE RECESSION-SENTIMENT REGRESSIONS\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        "MODEL\n"
    )

    file.write(
        "Sentiment = Intercept + Recession + Error\n\n"
    )

    file.write(
        "Recession is coded as:\n"
    )

    file.write(
        "1 = recession quarter\n"
    )

    file.write(
        "0 = non-recession quarter\n\n"
    )


    for _, row in regression_table.iterrows():

        file.write(
            f"Dictionary: {row['Dictionary']}\n"
        )

        file.write(
            f"Outcome: {row['Outcome']}\n"
        )

        file.write(
            "Recession coefficient: "
            f"{row['Recession_Coefficient_pp']:+.4f} pp\n"
        )

        file.write(
            "OLS standard error: "
            f"{row['OLS_Standard_Error_pp']:.4f} pp\n"
        )

        file.write(
            "p-value: "
            f"{row['OLS_P_Value']:.6f}\n"
        )

        file.write(
            "95% confidence interval: "
            f"[{row['CI_95_Lower_pp']:+.4f}, "
            f"{row['CI_95_Upper_pp']:+.4f}] pp\n"
        )

        file.write(
            "R-squared: "
            f"{row['R_Squared']:.6f}\n"
        )

        file.write(
            "\n"
        )


print(
    "\nSAVED:"
)

print(
    text_output
)


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 80)

print(
    "BASELINE REGRESSION ANALYSIS COMPLETE"
)

print("=" * 80)

print(
    "\nNext step:"
)

print(
    "Test heteroskedasticity and compare ordinary OLS "
    "standard errors with heteroskedasticity-robust "
    "standard errors."
)