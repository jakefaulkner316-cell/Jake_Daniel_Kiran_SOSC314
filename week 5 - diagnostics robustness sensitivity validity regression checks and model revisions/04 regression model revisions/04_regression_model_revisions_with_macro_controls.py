# =========================================================
# WEEK 5 MODEL REVISION:
# REGRESSION MODELS WITH MACROECONOMIC CONTROLS
# =========================================================
#
# PURPOSE:
# This script revises the baseline recession-sentiment regression
# by adding GDP growth as a macroeconomic control.
#
# BASELINE MODEL:
#
# Sentiment = Intercept + Recession + Error
#
# REVISED MODELS:
#
# Model 1:
# Sentiment = Intercept + Recession + Error
#
# Model 2:
# Sentiment = Intercept + Recession + GDP Growth + Error
#
# Model 3:
# Sentiment = Intercept + GDP Growth + Error
#
# WHY THIS MATTERS:
#
# Recession status is a broad binary indicator.
# GDP growth provides a continuous measure of economic conditions.
#
# Adding GDP growth lets us test whether the recession coefficient
# remains similar after accounting for variation in the strength
# or weakness of the economy.
#
# Because the earlier regression diagnostics suggested serial
# correlation in the quarterly data, HAC / Newey-West standard
# errors are used for inference in the revised models.
#
# DICTIONARIES:
# - Harvard General Inquirer
# - Loughran-McDonald
#
# OUTCOMES:
# - Positive sentiment
# - Negative sentiment
#
# OUTPUTS:
#
# csv outputs/
#   revised_regression_models_with_gdp_growth.csv
#
# diagnostic images/
#   recession_coefficient_across_model_specifications.png
#
# robustness tables/
#   revised_model_summary.txt
#
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
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


RECESSION_FILE = (
    REPO_ROOT
    / "week three data cleaning"
    / "nber_recession_table.csv"
)


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = (
    SCRIPT_DIR
    / "csv outputs"
)

IMAGE_OUTPUT_DIR = (
    SCRIPT_DIR
    / "diagnostic images"
)

TABLE_OUTPUT_DIR = (
    SCRIPT_DIR
    / "robustness tables"
)


CSV_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

harvard = pd.read_csv(
    HARVARD_FILE
)

lm = pd.read_csv(
    LM_FILE
)

recession_data = pd.read_csv(
    RECESSION_FILE
)


# ---------------------------------------------------------
# MERGE GDP GROWTH INTO SENTIMENT DATA
# ---------------------------------------------------------

macro_columns = [
    "Year_Quarter",
    "GDP_Growth",
    "Recession_Majority"
]


harvard = harvard.merge(
    recession_data[
        macro_columns
    ],
    on="Year_Quarter",
    how="inner"
)


lm = lm.merge(
    recession_data[
        macro_columns
    ],
    on="Year_Quarter",
    how="inner"
)


# ---------------------------------------------------------
# CREATE NUMERIC RECESSION INDICATOR
# ---------------------------------------------------------

harvard[
    "Recession"
] = (
    harvard[
        "Recession_Majority"
    ]
    .astype(int)
)


lm[
    "Recession"
] = (
    lm[
        "Recession_Majority"
    ]
    .astype(int)
)


# ---------------------------------------------------------
# CHECK DATA
# ---------------------------------------------------------

print("\n" + "=" * 90)

print(
    "HARVARD REGRESSION DATA"
)

print("=" * 90)

print(
    harvard.head()
)


print("\n" + "=" * 90)

print(
    "LOUGHRAN-MCDONALD REGRESSION DATA"
)

print("=" * 90)

print(
    lm.head()
)


print("\nHARVARD OBSERVATIONS:")

print(
    len(harvard)
)


print("\nLOUGHRAN-MCDONALD OBSERVATIONS:")

print(
    len(lm)
)


# ---------------------------------------------------------
# FUNCTION TO RUN HAC REGRESSION
# ---------------------------------------------------------

def run_hac_regression(
    data,
    outcome,
    predictors,
    dictionary_name,
    model_name
):

    # Keep only needed variables.

    model_data = (
        data[
            [outcome] + predictors
        ]
        .dropna()
        .copy()
    )


    # Independent variables.

    X = model_data[
        predictors
    ]


    # Add intercept.

    X = sm.add_constant(
        X
    )


    # Dependent variable.

    y = model_data[
        outcome
    ]


    # Estimate ordinary least squares model.

    model = sm.OLS(
        y,
        X
    ).fit()


    # Re-estimate inference using HAC / Newey-West
    # standard errors.
    #
    # maxlags=4 allows correlation across approximately
    # one year of quarterly observations.

    hac_model = model.get_robustcov_results(
        cov_type="HAC",
        maxlags=4
    )


    parameter_names = list(
        model.params.index
    )


    results = []


    for variable in predictors:

        variable_index = (
            parameter_names.index(
                variable
            )
        )


        coefficient = (
            model.params[
                variable
            ]
        )


        hac_se = (
            hac_model.bse[
                variable_index
            ]
        )


        hac_p = (
            hac_model.pvalues[
                variable_index
            ]
        )


        hac_ci = (
            hac_model.conf_int()[
                variable_index
            ]
        )


        results.append(
            {
                "Dictionary":
                    dictionary_name,

                "Outcome":
                    outcome,

                "Model":
                    model_name,

                "Variable":
                    variable,

                "Observations":
                    int(
                        model.nobs
                    ),

                "Coefficient":
                    coefficient,

                "HAC_SE":
                    hac_se,

                "HAC_P_Value":
                    hac_p,

                "HAC_CI_Lower":
                    hac_ci[0],

                "HAC_CI_Upper":
                    hac_ci[1],

                "R_Squared":
                    model.rsquared
            }
        )


    return results


# ---------------------------------------------------------
# DEFINE MODEL SPECIFICATIONS
# ---------------------------------------------------------

model_specs = [
    (
        "Model 1: Recession only",
        [
            "Recession"
        ]
    ),

    (
        "Model 2: Recession + GDP Growth",
        [
            "Recession",
            "GDP_Growth"
        ]
    ),

    (
        "Model 3: GDP Growth only",
        [
            "GDP_Growth"
        ]
    )
]


# ---------------------------------------------------------
# RUN MODELS
# ---------------------------------------------------------

all_results = []


datasets = [
    (
        harvard,
        "Harvard"
    ),

    (
        lm,
        "Loughran-McDonald"
    )
]


outcomes = [
    "Pos",
    "Neg"
]


for data, dictionary_name in datasets:

    for outcome in outcomes:

        for model_name, predictors in model_specs:

            model_results = (
                run_hac_regression(
                    data=data,
                    outcome=outcome,
                    predictors=predictors,
                    dictionary_name=dictionary_name,
                    model_name=model_name
                )
            )

            all_results.extend(
                model_results
            )


# ---------------------------------------------------------
# CREATE RESULTS DATAFRAME
# ---------------------------------------------------------

results = pd.DataFrame(
    all_results
)


# ---------------------------------------------------------
# CONVERT SENTIMENT COEFFICIENTS TO PERCENTAGE POINTS
# ---------------------------------------------------------
#
# Sentiment outcomes are stored as proportions.
#
# For recession coefficients:
# multiply by 100 to interpret as percentage points.
#
# GDP growth is already measured in percentage terms,
# so its coefficient represents sentiment change per
# one percentage-point change in GDP growth.
# ---------------------------------------------------------

results[
    "Coefficient_pp"
] = (
    results[
        "Coefficient"
    ]
    * 100
)


results[
    "HAC_SE_pp"
] = (
    results[
        "HAC_SE"
    ]
    * 100
)


results[
    "HAC_CI_Lower_pp"
] = (
    results[
        "HAC_CI_Lower"
    ]
    * 100
)


results[
    "HAC_CI_Upper_pp"
] = (
    results[
        "HAC_CI_Upper"
    ]
    * 100
)


# ---------------------------------------------------------
# ADD SIGNIFICANCE INDICATOR
# ---------------------------------------------------------

results[
    "Significant_05"
] = (
    results[
        "HAC_P_Value"
    ] < 0.05
)


# ---------------------------------------------------------
# PRINT FULL RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 100)

print(
    "REVISED REGRESSION MODELS WITH HAC STANDARD ERRORS"
)

print("=" * 100)

print(
    results[
        [
            "Dictionary",
            "Outcome",
            "Model",
            "Variable",
            "Observations",
            "Coefficient_pp",
            "HAC_SE_pp",
            "HAC_P_Value",
            "HAC_CI_Lower_pp",
            "HAC_CI_Upper_pp",
            "R_Squared",
            "Significant_05"
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE FULL RESULTS
# ---------------------------------------------------------

results_file = (
    CSV_OUTPUT_DIR
    / "revised_regression_models_with_gdp_growth.csv"
)


results.to_csv(
    results_file,
    index=False
)


print(
    "\nSAVED:"
)

print(
    results_file
)


# ---------------------------------------------------------
# CREATE RECESSION-COEFFICIENT COMPARISON
# ---------------------------------------------------------
#
# Compare the recession coefficient in:
#
# Model 1:
# Recession only
#
# Model 2:
# Recession + GDP growth
#
# This directly shows whether adding GDP growth changes
# the estimated recession relationship.
# ---------------------------------------------------------

recession_results = (
    results[
        results[
            "Variable"
        ] == "Recession"
    ]
    .copy()
)


recession_results[
    "Label"
] = (
    recession_results[
        "Dictionary"
    ]
    + "\n"
    + recession_results[
        "Outcome"
    ]
    + "\n"
    + recession_results[
        "Model"
    ]
)


# ---------------------------------------------------------
# GRAPH RECESSION COEFFICIENTS
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 8)
)


ax.bar(
    recession_results[
        "Label"
    ],
    recession_results[
        "Coefficient_pp"
    ]
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_title(
    "Recession Coefficient Across Baseline and "
    "GDP-Controlled Models"
)


ax.set_xlabel(
    "Dictionary, Sentiment Outcome, and Model"
)


ax.set_ylabel(
    "Recession Coefficient "
    "(Percentage Points)"
)


plt.xticks(
    rotation=35,
    ha="right"
)


plt.tight_layout()


image_file = (
    IMAGE_OUTPUT_DIR
    / "recession_coefficient_across_model_specifications.png"
)


plt.savefig(
    image_file,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    image_file
)


# ---------------------------------------------------------
# CREATE HUMAN-READABLE SUMMARY
# ---------------------------------------------------------

summary_file = (
    TABLE_OUTPUT_DIR
    / "revised_model_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 REGRESSION MODEL REVISION\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        "MODELS TESTED\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    file.write(
        "Model 1: Sentiment = Recession\n"
    )

    file.write(
        "Model 2: Sentiment = Recession + GDP Growth\n"
    )

    file.write(
        "Model 3: Sentiment = GDP Growth\n\n"
    )

    file.write(
        "All inference uses HAC/Newey-West standard errors "
        "with four quarterly lags.\n\n"
    )


    file.write(
        "RECESSION COEFFICIENT COMPARISON\n"
    )

    file.write(
        "-" * 70
        + "\n\n"
    )


    for _, row in recession_results.iterrows():

        file.write(
            f"Dictionary: {row['Dictionary']}\n"
        )

        file.write(
            f"Outcome: {row['Outcome']}\n"
        )

        file.write(
            f"Model: {row['Model']}\n"
        )

        file.write(
            "Recession coefficient: "
            f"{row['Coefficient_pp']:+.4f} pp\n"
        )

        file.write(
            "HAC standard error: "
            f"{row['HAC_SE_pp']:.4f} pp\n"
        )

        file.write(
            "HAC p-value: "
            f"{row['HAC_P_Value']:.6f}\n"
        )

        file.write(
            "R-squared: "
            f"{row['R_Squared']:.6f}\n"
        )

        file.write(
            "\n"
        )


    file.write(
        "INTERPRETATION GUIDE\n"
    )

    file.write(
        "-" * 70
        + "\n\n"
    )

    file.write(
        "If the recession coefficient changes substantially "
        "after GDP growth is added, part of the original "
        "recession association may reflect broader changes "
        "in economic growth rather than recession status alone.\n\n"
    )

    file.write(
        "If the coefficient remains similar, the recession "
        "relationship is more robust to controlling for GDP growth.\n\n"
    )

    file.write(
        "The GDP-only model helps determine whether sentiment "
        "responds more directly to continuous changes in economic "
        "growth rather than to the binary recession classification.\n"
    )


print(
    "\nSAVED:"
)

print(
    summary_file
)


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 100)

print(
    "REGRESSION MODEL REVISION COMPLETE"
)

print("=" * 100)

print(
    "\nCreated:"
)

print(
    "1. Revised regression results CSV"
)

print(
    "2. Recession coefficient comparison graph"
)

print(
    "3. Human-readable model revision summary"
)