# =========================================================
# WEEK 5 DIAGNOSTIC 3B:
# HETEROSKEDASTICITY, HC3, AND HAC ROBUSTNESS CHECKS
# =========================================================
#
# PURPOSE:
# This script diagnoses whether the baseline recession-sentiment
# regressions are sensitive to violations of ordinary OLS
# standard-error assumptions.
#
# BASELINE MODEL:
#
# Sentiment = Intercept + Recession + Error
#
# DICTIONARIES:
# - Harvard General Inquirer
# - Loughran-McDonald
#
# OUTCOMES:
# - Positive sentiment
# - Negative sentiment
#
# WHY THIS DIAGNOSTIC IS NEEDED:
#
# Ordinary OLS standard errors assume that the residual variance
# is constant across observations.
#
# The baseline regression output also showed very low
# Durbin-Watson statistics for some quarterly sentiment series,
# suggesting that serial correlation may also be present.
#
# Therefore, this script compares:
#
# 1. Conventional OLS standard errors
# 2. HC3 heteroskedasticity-robust standard errors
# 3. HAC / Newey-West standard errors
#
# HAC standard errors are especially useful here because our
# observations are quarterly and may be serially correlated.
#
# DIAGNOSTIC TEST:
#
# We also run the Breusch-Pagan test for heteroskedasticity.
#
# MAIN QUESTION:
#
# Does the statistical interpretation of the recession coefficient
# change when we relax the standard OLS assumptions?
#
# OUTPUTS:
#
# csv outputs/
#   heteroskedasticity_and_robust_standard_error_results.csv
#   breusch_pagan_test_results.csv
#
# diagnostic images/
#   standard_error_comparison.png
#
# robustness tables/
#   robust_standard_error_summary.txt
#
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path

from statsmodels.stats.diagnostic import het_breuschpagan


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
# LOAD QUARTERLY DATA
# ---------------------------------------------------------

harvard = pd.read_csv(
    HARVARD_FILE
)

lm = pd.read_csv(
    LM_FILE
)


# ---------------------------------------------------------
# CREATE NUMERIC RECESSION INDICATOR
# ---------------------------------------------------------

for data in [
    harvard,
    lm
]:

    data["Recession"] = (
        data["Recession_Status"]
        .map(
            {
                "Recession": 1,
                "Not Recession": 0
            }
        )
    )


# ---------------------------------------------------------
# FUNCTION TO RUN ALL ROBUSTNESS CHECKS
# ---------------------------------------------------------

def run_robustness_checks(
    data,
    outcome,
    dictionary_name
):

    # Independent variable.

    X = data[
        [
            "Recession"
        ]
    ]

    X = sm.add_constant(
        X
    )


    # Dependent variable.

    y = data[
        outcome
    ]


    # -----------------------------------------------------
    # BASELINE OLS MODEL
    # -----------------------------------------------------

    ols_model = sm.OLS(
        y,
        X,
        missing="drop"
    ).fit()


    # -----------------------------------------------------
    # HC3 ROBUST STANDARD ERRORS
    # -----------------------------------------------------
    #
    # HC3 adjusts inference for heteroskedasticity.
    # The regression coefficients stay the same.
    # Only the estimated uncertainty changes.
    # -----------------------------------------------------

    hc3_model = ols_model.get_robustcov_results(
        cov_type="HC3"
    )


    # -----------------------------------------------------
    # HAC / NEWEY-WEST STANDARD ERRORS
    # -----------------------------------------------------
    #
    # HAC allows both heteroskedasticity and serial correlation.
    #
    # maxlags=4 allows correlation across roughly one year
    # of quarterly observations.
    # -----------------------------------------------------

    hac_model = ols_model.get_robustcov_results(
        cov_type="HAC",
        maxlags=4
    )


    # -----------------------------------------------------
    # BREUSCH-PAGAN TEST
    # -----------------------------------------------------
    #
    # Null hypothesis:
    # residual variance is constant.
    #
    # Small p-value:
    # evidence of heteroskedasticity.
    # -----------------------------------------------------

    bp_test = het_breuschpagan(
        ols_model.resid,
        ols_model.model.exog
    )


    bp_lm_statistic = bp_test[0]
    bp_lm_pvalue = bp_test[1]
    bp_f_statistic = bp_test[2]
    bp_f_pvalue = bp_test[3]


    # -----------------------------------------------------
    # EXTRACT RECESSION COEFFICIENT
    # -----------------------------------------------------

    recession_index = (
        list(
            ols_model.params.index
        )
        .index(
            "Recession"
        )
    )


    coefficient = (
        ols_model.params[
            "Recession"
        ]
    )


    # Ordinary OLS inference.

    ols_se = (
        ols_model.bse[
            "Recession"
        ]
    )

    ols_p = (
        ols_model.pvalues[
            "Recession"
        ]
    )


    # HC3 inference.

    hc3_se = (
        hc3_model.bse[
            recession_index
        ]
    )

    hc3_p = (
        hc3_model.pvalues[
            recession_index
        ]
    )


    # HAC inference.

    hac_se = (
        hac_model.bse[
            recession_index
        ]
    )

    hac_p = (
        hac_model.pvalues[
            recession_index
        ]
    )


    # -----------------------------------------------------
    # CONFIDENCE INTERVALS
    # -----------------------------------------------------

    ols_ci = (
        ols_model.conf_int()
        .loc[
            "Recession"
        ]
    )


    hc3_ci = (
        hc3_model.conf_int()[
            recession_index
        ]
    )


    hac_ci = (
        hac_model.conf_int()[
            recession_index
        ]
    )


    # -----------------------------------------------------
    # DURBIN-WATSON
    # -----------------------------------------------------

    durbin_watson = (
        sm.stats.stattools.durbin_watson(
            ols_model.resid
        )
    )


    # -----------------------------------------------------
    # RETURN MAIN RESULT
    # -----------------------------------------------------

    main_result = {
        "Dictionary":
            dictionary_name,

        "Outcome":
            outcome,

        "Observations":
            int(
                ols_model.nobs
            ),

        "Recession_Coefficient":
            coefficient,

        "OLS_SE":
            ols_se,

        "OLS_P_Value":
            ols_p,

        "HC3_SE":
            hc3_se,

        "HC3_P_Value":
            hc3_p,

        "HAC_SE":
            hac_se,

        "HAC_P_Value":
            hac_p,

        "OLS_CI_Lower":
            ols_ci.iloc[0],

        "OLS_CI_Upper":
            ols_ci.iloc[1],

        "HC3_CI_Lower":
            hc3_ci[0],

        "HC3_CI_Upper":
            hc3_ci[1],

        "HAC_CI_Lower":
            hac_ci[0],

        "HAC_CI_Upper":
            hac_ci[1],

        "Durbin_Watson":
            durbin_watson,

        "R_Squared":
            ols_model.rsquared
    }


    # -----------------------------------------------------
    # RETURN BREUSCH-PAGAN RESULT
    # -----------------------------------------------------

    bp_result = {
        "Dictionary":
            dictionary_name,

        "Outcome":
            outcome,

        "LM_Statistic":
            bp_lm_statistic,

        "LM_P_Value":
            bp_lm_pvalue,

        "F_Statistic":
            bp_f_statistic,

        "F_P_Value":
            bp_f_pvalue
    }


    return (
        main_result,
        bp_result
    )


# ---------------------------------------------------------
# RUN ALL FOUR MODELS
# ---------------------------------------------------------

robustness_rows = []

bp_rows = []


model_specs = [
    (
        harvard,
        "Pos",
        "Harvard"
    ),

    (
        harvard,
        "Neg",
        "Harvard"
    ),

    (
        lm,
        "Pos",
        "Loughran-McDonald"
    ),

    (
        lm,
        "Neg",
        "Loughran-McDonald"
    )
]


for data, outcome, dictionary_name in model_specs:

    robustness_result, bp_result = (
        run_robustness_checks(
            data,
            outcome,
            dictionary_name
        )
    )

    robustness_rows.append(
        robustness_result
    )

    bp_rows.append(
        bp_result
    )


# ---------------------------------------------------------
# CREATE DATAFRAMES
# ---------------------------------------------------------

robustness_table = pd.DataFrame(
    robustness_rows
)

bp_table = pd.DataFrame(
    bp_rows
)


# ---------------------------------------------------------
# CONVERT COEFFICIENTS AND STANDARD ERRORS
# TO PERCENTAGE POINTS
# ---------------------------------------------------------

percentage_point_columns = [
    "Recession_Coefficient",
    "OLS_SE",
    "HC3_SE",
    "HAC_SE",
    "OLS_CI_Lower",
    "OLS_CI_Upper",
    "HC3_CI_Lower",
    "HC3_CI_Upper",
    "HAC_CI_Lower",
    "HAC_CI_Upper"
]


robustness_table[
    percentage_point_columns
] = (
    robustness_table[
        percentage_point_columns
    ]
    * 100
)


# ---------------------------------------------------------
# RENAME COLUMNS FOR CLARITY
# ---------------------------------------------------------

robustness_table = robustness_table.rename(
    columns={
        "Recession_Coefficient":
            "Recession_Coefficient_pp",

        "OLS_SE":
            "OLS_SE_pp",

        "HC3_SE":
            "HC3_SE_pp",

        "HAC_SE":
            "HAC_SE_pp",

        "OLS_CI_Lower":
            "OLS_CI_Lower_pp",

        "OLS_CI_Upper":
            "OLS_CI_Upper_pp",

        "HC3_CI_Lower":
            "HC3_CI_Lower_pp",

        "HC3_CI_Upper":
            "HC3_CI_Upper_pp",

        "HAC_CI_Lower":
            "HAC_CI_Lower_pp",

        "HAC_CI_Upper":
            "HAC_CI_Upper_pp"
    }
)


# ---------------------------------------------------------
# DETERMINE WHETHER SIGNIFICANCE CHANGES
# ---------------------------------------------------------

robustness_table[
    "OLS_Significant_05"
] = (
    robustness_table[
        "OLS_P_Value"
    ] < 0.05
)


robustness_table[
    "HC3_Significant_05"
] = (
    robustness_table[
        "HC3_P_Value"
    ] < 0.05
)


robustness_table[
    "HAC_Significant_05"
] = (
    robustness_table[
        "HAC_P_Value"
    ] < 0.05
)


# ---------------------------------------------------------
# PRINT MAIN RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 100)

print(
    "OLS VS HC3 VS HAC STANDARD ERROR COMPARISON"
)

print("=" * 100)

print(
    robustness_table.round(
        6
    ).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# PRINT BREUSCH-PAGAN RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 100)

print(
    "BREUSCH-PAGAN HETEROSKEDASTICITY TEST"
)

print("=" * 100)

print(
    bp_table.round(
        6
    ).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# SAVE MAIN RESULTS
# ---------------------------------------------------------

robustness_csv = (
    CSV_OUTPUT_DIR
    / "heteroskedasticity_and_robust_standard_error_results.csv"
)


robustness_table.to_csv(
    robustness_csv,
    index=False
)


# ---------------------------------------------------------
# SAVE BREUSCH-PAGAN RESULTS
# ---------------------------------------------------------

bp_csv = (
    CSV_OUTPUT_DIR
    / "breusch_pagan_test_results.csv"
)


bp_table.to_csv(
    bp_csv,
    index=False
)


print(
    "\nSAVED:"
)

print(
    robustness_csv
)

print(
    bp_csv
)


# =========================================================
# STANDARD ERROR COMPARISON GRAPH
# =========================================================

plot_data = robustness_table.copy()


plot_data[
    "Method"
] = (
    plot_data[
        "Dictionary"
    ]
    + "\n"
    + plot_data[
        "Outcome"
    ]
)


x = list(
    range(
        len(
            plot_data
        )
    )
)


width = 0.25


fig, ax = plt.subplots(
    figsize=(13, 7)
)


ax.bar(
    [
        i - width
        for i in x
    ],
    plot_data[
        "OLS_SE_pp"
    ],
    width=width,
    label="OLS"
)


ax.bar(
    x,
    plot_data[
        "HC3_SE_pp"
    ],
    width=width,
    label="HC3"
)


ax.bar(
    [
        i + width
        for i in x
    ],
    plot_data[
        "HAC_SE_pp"
    ],
    width=width,
    label="HAC / Newey-West"
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    plot_data[
        "Method"
    ]
)


ax.set_title(
    "Comparison of OLS, HC3, and HAC Standard Errors"
)


ax.set_xlabel(
    "Dictionary and Sentiment Outcome"
)


ax.set_ylabel(
    "Standard Error of Recession Coefficient "
    "(Percentage Points)"
)


ax.legend()


plt.tight_layout()


image_path = (
    IMAGE_OUTPUT_DIR
    / "standard_error_comparison.png"
)


plt.savefig(
    image_path,
    dpi=300
)


plt.close()


print(
    image_path
)


# =========================================================
# CREATE HUMAN-READABLE SUMMARY
# =========================================================

summary_path = (
    TABLE_OUTPUT_DIR
    / "robust_standard_error_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEEK 5 ROBUST STANDARD ERROR DIAGNOSTIC\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )


    file.write(
        "PURPOSE\n"
    )

    file.write(
        "-" * 70
        + "\n"
    )

    file.write(
        "Compare ordinary OLS inference with HC3 "
        "heteroskedasticity-robust standard errors and "
        "HAC/Newey-West standard errors.\n\n"
    )


    for _, row in robustness_table.iterrows():

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
            "OLS SE: "
            f"{row['OLS_SE_pp']:.4f} pp\n"
        )

        file.write(
            "OLS p-value: "
            f"{row['OLS_P_Value']:.6f}\n"
        )

        file.write(
            "HC3 SE: "
            f"{row['HC3_SE_pp']:.4f} pp\n"
        )

        file.write(
            "HC3 p-value: "
            f"{row['HC3_P_Value']:.6f}\n"
        )

        file.write(
            "HAC SE: "
            f"{row['HAC_SE_pp']:.4f} pp\n"
        )

        file.write(
            "HAC p-value: "
            f"{row['HAC_P_Value']:.6f}\n"
        )

        file.write(
            "Durbin-Watson: "
            f"{row['Durbin_Watson']:.4f}\n"
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
        "HC3 addresses heteroskedasticity without changing "
        "the regression coefficient.\n\n"
    )

    file.write(
        "HAC/Newey-West addresses both heteroskedasticity "
        "and serial correlation in the quarterly residuals.\n\n"
    )

    file.write(
        "If statistical significance changes across OLS, HC3, "
        "and HAC, then inference is sensitive to the standard-"
        "error assumption.\n\n"
    )

    file.write(
        "If the coefficient remains similar but p-values change, "
        "the substantive estimate is stable while the estimated "
        "uncertainty is not.\n"
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
    "HETEROSKEDASTICITY AND ROBUST STANDARD ERROR "
    "DIAGNOSTIC COMPLETE"
)

print("=" * 100)