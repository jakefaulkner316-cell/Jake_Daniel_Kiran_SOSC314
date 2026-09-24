# =========================================================
# WEEK 5 MODEL REVISION 04B:
# RECESSION VS GDP GROWTH WITH INFLATION AND TIME TREND
# =========================================================
#
# PURPOSE:
# This script tests whether the relationship between economic
# conditions and advertising sentiment remains similar when:
#
# 1. The sample is restricted to years where U.S. inflation data
#    are available in the World Bank dataset.
#
# 2. Inflation is added as a macroeconomic control.
#
# 3. A linear time trend is added to account for long-run changes
#    in advertising language across the historical period.
#
# IMPORTANT SAMPLE ISSUE:
#
# The World Bank U.S. inflation data begin in 1990.
# Therefore, models including inflation use the shorter
# 1990+ sample.
#
# To avoid confusing "adding inflation" with "changing the sample,"
# this script also estimates the simple recession and GDP-growth
# models on the SAME restricted sample.
#
# MODELS:
#
# RECESSION MODELS
#
# Model 1:
# Sentiment = Recession
# Full historical sample
#
# Model 2:
# Sentiment = Recession
# Restricted inflation-available sample
#
# Model 3:
# Sentiment = Recession + Inflation + Time Trend
# Restricted inflation-available sample
#
#
# GDP GROWTH MODELS
#
# Model 4:
# Sentiment = GDP Growth
# Restricted inflation-available sample
#
# Model 5:
# Sentiment = GDP Growth + Inflation + Time Trend
# Restricted inflation-available sample
#
#
# STANDARD ERRORS:
#
# All models use HAC / Newey-West standard errors with 4 lags
# because the data are quarterly and previous diagnostics
# indicated serial correlation.
#
# DICTIONARIES:
#
# - Harvard General Inquirer
# - Loughran-McDonald
#
# OUTCOMES:
#
# - Positive sentiment
# - Negative sentiment
#
# OUTPUTS:
#
# csv outputs/
#   inflation_time_trend_model_comparison.csv
#
# diagnostic images/
#   recession_coefficients_full_vs_restricted_vs_controls.png
#   gdp_growth_coefficients_restricted_vs_controls.png
#
# robustness tables/
#   inflation_time_trend_model_summary.txt
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


WORLD_BANK_FILE = (
    REPO_ROOT
    / "Initial file imports"
    / "world_bank_economic_data.csv"
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

world_bank = pd.read_csv(
    WORLD_BANK_FILE
)


# ---------------------------------------------------------
# PREPARE UNITED STATES INFLATION DATA
# ---------------------------------------------------------

us_inflation = (
    world_bank[
        world_bank["Country_Code"] == "USA"
    ][
        [
            "Year",
            "Inflation_CPI_percent"
        ]
    ]
    .copy()
)


# Keep only years with actual inflation values.

us_inflation = (
    us_inflation[
        us_inflation[
            "Inflation_CPI_percent"
        ].notna()
    ]
    .copy()
)


print("\n" + "=" * 90)

print(
    "UNITED STATES INFLATION DATA"
)

print("=" * 90)

print(
    us_inflation.head()
)

print(
    "\nFirst inflation year:",
    us_inflation["Year"].min()
)

print(
    "Last inflation year:",
    us_inflation["Year"].max()
)


# ---------------------------------------------------------
# ADD YEAR TO QUARTERLY SENTIMENT DATA
# ---------------------------------------------------------
#
# Example:
# 1948Q1 -> 1948
# ---------------------------------------------------------

for data in [
    harvard,
    lm
]:

    data["Year"] = (
        data[
            "Year_Quarter"
        ]
        .str[:4]
        .astype(int)
    )


# ---------------------------------------------------------
# MERGE GDP GROWTH AND RECESSION DATA
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
# MERGE ANNUAL INFLATION ONTO QUARTERS
# ---------------------------------------------------------
#
# Each quarter within a given year receives the same
# annual CPI inflation value.
# ---------------------------------------------------------

harvard = harvard.merge(
    us_inflation,
    on="Year",
    how="left"
)


lm = lm.merge(
    us_inflation,
    on="Year",
    how="left"
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
# CREATE LINEAR TIME TREND
# ---------------------------------------------------------
#
# The first quarter is 0, the next is 1, and so on.
#
# This controls for long-run changes in advertising language.
# ---------------------------------------------------------

harvard = (
    harvard
    .sort_values(
        "Year_Quarter"
    )
    .reset_index(
        drop=True
    )
)


lm = (
    lm
    .sort_values(
        "Year_Quarter"
    )
    .reset_index(
        drop=True
    )
)


harvard[
    "Time_Trend"
] = range(
    len(
        harvard
    )
)


lm[
    "Time_Trend"
] = range(
    len(
        lm
    )
)


# ---------------------------------------------------------
# CREATE RESTRICTED INFLATION-AVAILABLE SAMPLE
# ---------------------------------------------------------

harvard_restricted = (
    harvard[
        harvard[
            "Inflation_CPI_percent"
        ].notna()
    ]
    .copy()
)


lm_restricted = (
    lm[
        lm[
            "Inflation_CPI_percent"
        ].notna()
    ]
    .copy()
)


# ---------------------------------------------------------
# PRINT SAMPLE SIZES
# ---------------------------------------------------------

print("\n" + "=" * 90)

print(
    "SAMPLE SIZE CHECK"
)

print("=" * 90)

print(
    "Harvard full sample:",
    len(
        harvard
    )
)

print(
    "Harvard restricted inflation sample:",
    len(
        harvard_restricted
    )
)

print(
    "LM full sample:",
    len(
        lm
    )
)

print(
    "LM restricted inflation sample:",
    len(
        lm_restricted
    )
)


print(
    "\nRestricted sample first quarter:"
)

print(
    harvard_restricted[
        "Year_Quarter"
    ].min()
)


print(
    "\nRestricted sample last quarter:"
)

print(
    harvard_restricted[
        "Year_Quarter"
    ].max()
)


# ---------------------------------------------------------
# FUNCTION TO RUN HAC MODEL
# ---------------------------------------------------------

def run_hac_model(
    data,
    outcome,
    predictors,
    dictionary_name,
    model_name,
    sample_name
):

    model_data = (
        data[
            [outcome]
            + predictors
        ]
        .dropna()
        .copy()
    )


    X = model_data[
        predictors
    ]


    X = sm.add_constant(
        X
    )


    y = model_data[
        outcome
    ]


    ols_model = sm.OLS(
        y,
        X
    ).fit()


    hac_model = (
        ols_model
        .get_robustcov_results(
            cov_type="HAC",
            maxlags=4
        )
    )


    parameter_names = list(
        ols_model.params.index
    )


    output_rows = []


    for predictor in predictors:

        predictor_index = (
            parameter_names.index(
                predictor
            )
        )


        hac_ci = (
            hac_model
            .conf_int()[
                predictor_index
            ]
        )


        output_rows.append(
            {
                "Dictionary":
                    dictionary_name,

                "Outcome":
                    outcome,

                "Sample":
                    sample_name,

                "Model":
                    model_name,

                "Variable":
                    predictor,

                "Observations":
                    int(
                        ols_model.nobs
                    ),

                "Coefficient":
                    ols_model.params[
                        predictor
                    ],

                "HAC_SE":
                    hac_model.bse[
                        predictor_index
                    ],

                "HAC_P_Value":
                    hac_model.pvalues[
                        predictor_index
                    ],

                "HAC_CI_Lower":
                    hac_ci[0],

                "HAC_CI_Upper":
                    hac_ci[1],

                "R_Squared":
                    ols_model.rsquared
            }
        )


    return output_rows


# ---------------------------------------------------------
# RUN ALL MODEL SPECIFICATIONS
# ---------------------------------------------------------

all_results = []


datasets = [
    (
        harvard,
        harvard_restricted,
        "Harvard"
    ),

    (
        lm,
        lm_restricted,
        "Loughran-McDonald"
    )
]


outcomes = [
    "Pos",
    "Neg"
]


for full_data, restricted_data, dictionary_name in datasets:

    for outcome in outcomes:

        # -------------------------------------------------
        # MODEL 1:
        # RECESSION ONLY, FULL SAMPLE
        # -------------------------------------------------

        all_results.extend(
            run_hac_model(
                data=full_data,
                outcome=outcome,
                predictors=[
                    "Recession"
                ],
                dictionary_name=dictionary_name,
                model_name="Recession only",
                sample_name="Full sample"
            )
        )


        # -------------------------------------------------
        # MODEL 2:
        # RECESSION ONLY, RESTRICTED SAMPLE
        # -------------------------------------------------

        all_results.extend(
            run_hac_model(
                data=restricted_data,
                outcome=outcome,
                predictors=[
                    "Recession"
                ],
                dictionary_name=dictionary_name,
                model_name="Recession only",
                sample_name="Restricted inflation sample"
            )
        )


        # -------------------------------------------------
        # MODEL 3:
        # RECESSION + INFLATION + TIME TREND
        # -------------------------------------------------

        all_results.extend(
            run_hac_model(
                data=restricted_data,
                outcome=outcome,
                predictors=[
                    "Recession",
                    "Inflation_CPI_percent",
                    "Time_Trend"
                ],
                dictionary_name=dictionary_name,
                model_name=(
                    "Recession + Inflation + Time Trend"
                ),
                sample_name="Restricted inflation sample"
            )
        )


        # -------------------------------------------------
        # MODEL 4:
        # GDP GROWTH ONLY, RESTRICTED SAMPLE
        # -------------------------------------------------

        all_results.extend(
            run_hac_model(
                data=restricted_data,
                outcome=outcome,
                predictors=[
                    "GDP_Growth"
                ],
                dictionary_name=dictionary_name,
                model_name="GDP Growth only",
                sample_name="Restricted inflation sample"
            )
        )


        # -------------------------------------------------
        # MODEL 5:
        # GDP GROWTH + INFLATION + TIME TREND
        # -------------------------------------------------

        all_results.extend(
            run_hac_model(
                data=restricted_data,
                outcome=outcome,
                predictors=[
                    "GDP_Growth",
                    "Inflation_CPI_percent",
                    "Time_Trend"
                ],
                dictionary_name=dictionary_name,
                model_name=(
                    "GDP Growth + Inflation + Time Trend"
                ),
                sample_name="Restricted inflation sample"
            )
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
# ADD SIGNIFICANCE FLAG
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

print("\n" + "=" * 120)

print(
    "INFLATION AND TIME TREND MODEL COMPARISON"
)

print("=" * 120)

print(
    results[
        [
            "Dictionary",
            "Outcome",
            "Sample",
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

csv_path = (
    CSV_OUTPUT_DIR
    / "inflation_time_trend_model_comparison.csv"
)


results.to_csv(
    csv_path,
    index=False
)


print(
    "\nSAVED:"
)

print(
    csv_path
)


# =========================================================
# RECESSION COEFFICIENT COMPARISON GRAPH
# =========================================================

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
        "Sample"
    ]
    + "\n"
    + recession_results[
        "Model"
    ]
)


fig, ax = plt.subplots(
    figsize=(16, 8)
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
    "Recession Coefficients Across Full, Restricted, "
    "and Inflation-Controlled Models"
)


ax.set_ylabel(
    "Recession Coefficient "
    "(Percentage Points)"
)


ax.set_xlabel(
    "Dictionary, Outcome, Sample, and Model"
)


plt.xticks(
    rotation=45,
    ha="right"
)


plt.tight_layout()


recession_image = (
    IMAGE_OUTPUT_DIR
    / "recession_coefficients_full_vs_restricted_vs_controls.png"
)


plt.savefig(
    recession_image,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    recession_image
)


# =========================================================
# GDP GROWTH COEFFICIENT COMPARISON GRAPH
# =========================================================

gdp_results = (
    results[
        results[
            "Variable"
        ] == "GDP_Growth"
    ]
    .copy()
)


gdp_results[
    "Label"
] = (
    gdp_results[
        "Dictionary"
    ]
    + "\n"
    + gdp_results[
        "Outcome"
    ]
    + "\n"
    + gdp_results[
        "Model"
    ]
)


fig, ax = plt.subplots(
    figsize=(14, 8)
)


ax.bar(
    gdp_results[
        "Label"
    ],
    gdp_results[
        "Coefficient_pp"
    ]
)


ax.axhline(
    y=0,
    linewidth=1
)


ax.set_title(
    "GDP Growth Coefficients With and Without "
    "Inflation and Time Trend Controls"
)


ax.set_ylabel(
    "Sentiment Change per One Percentage-Point "
    "Increase in GDP Growth"
)


ax.set_xlabel(
    "Dictionary, Outcome, and Model"
)


plt.xticks(
    rotation=35,
    ha="right"
)


plt.tight_layout()


gdp_image = (
    IMAGE_OUTPUT_DIR
    / "gdp_growth_coefficients_restricted_vs_controls.png"
)


plt.savefig(
    gdp_image,
    dpi=300
)


plt.close()


print(
    "\nSAVED:"
)

print(
    gdp_image
)


# =========================================================
# CREATE HUMAN-READABLE SUMMARY
# =========================================================

summary_path = (
    TABLE_OUTPUT_DIR
    / "inflation_time_trend_model_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 INFLATION AND TIME TREND MODEL REVISION\n"
    )

    file.write(
        "=" * 75
        + "\n\n"
    )


    file.write(
        "PURPOSE\n"
    )

    file.write(
        "-" * 75
        + "\n"
    )

    file.write(
        "Test whether recession and GDP-growth sentiment "
        "relationships remain similar after accounting for "
        "inflation and long-run time trends.\n\n"
    )


    file.write(
        "IMPORTANT SAMPLE LIMITATION\n"
    )

    file.write(
        "-" * 75
        + "\n"
    )

    file.write(
        "World Bank U.S. CPI inflation begins in 1990. "
        "Therefore, inflation models use a shorter sample "
        "than the full historical regression.\n\n"
    )

    file.write(
        "To separate the effect of adding controls from the "
        "effect of changing the sample, recession-only models "
        "are estimated on both the full and restricted samples.\n\n"
    )


    file.write(
        "MODEL RESULTS\n"
    )

    file.write(
        "-" * 75
        + "\n\n"
    )


    for _, row in results.iterrows():

        file.write(
            f"Dictionary: {row['Dictionary']}\n"
        )

        file.write(
            f"Outcome: {row['Outcome']}\n"
        )

        file.write(
            f"Sample: {row['Sample']}\n"
        )

        file.write(
            f"Model: {row['Model']}\n"
        )

        file.write(
            f"Variable: {row['Variable']}\n"
        )

        file.write(
            "Coefficient: "
            f"{row['Coefficient_pp']:+.4f}\n"
        )

        file.write(
            "HAC standard error: "
            f"{row['HAC_SE_pp']:.4f}\n"
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
        "-" * 75
        + "\n\n"
    )

    file.write(
        "First compare the full-sample recession model with "
        "the restricted-sample recession model. This reveals "
        "whether changing the historical period alone changes "
        "the estimated relationship.\n\n"
    )

    file.write(
        "Then compare the restricted recession-only model "
        "with the recession + inflation + time-trend model. "
        "This shows whether the recession association changes "
        "after controlling for inflation and long-run temporal "
        "changes.\n\n"
    )

    file.write(
        "Similarly, compare GDP-growth-only results with the "
        "GDP growth + inflation + time-trend specification.\n"
    )


print(
    "\nSAVED:"
)

print(
    summary_path
)


# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 100)

print(
    "INFLATION AND TIME TREND MODEL REVISION COMPLETE"
)

print("=" * 100)