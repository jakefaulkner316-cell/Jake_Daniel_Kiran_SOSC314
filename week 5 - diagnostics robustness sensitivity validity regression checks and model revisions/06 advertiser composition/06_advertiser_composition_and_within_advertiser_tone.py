# =========================================================
# WEEK 5 DIAGNOSTIC 06:
# ADVERTISER COMPOSITION vs WITHIN-ADVERTISER TONE
# =========================================================
#
# PURPOSE:
# Respond to the Week 4 feedback: "If recessions change which
# kinds of firms advertise in The Economist, an aggregate shift in
# sentiment could reflect a change in advertisers rather than the
# same advertisers changing tone."
#
# MAIN DIAGNOSTIC QUESTION:
# Is the (small) recession difference in ad sentiment driven by
#   (a) COMPOSITION: a different mix of advertisers in recessions,
#       each with its own typical tone, or
#   (b) WITHIN-ADVERTISER TONE: the same advertisers writing more
#       or less positively when the economy is in recession?
#
# WHY THIS DIAGNOSTIC MATTERS:
# All earlier results compare quarterly AVERAGES. An average moves
# if every advertiser changes tone, but it also moves if cheerful
# advertisers (e.g. airlines, luxury goods) buy less space in a
# recession while sober ones (banks, financial notices) keep
# advertising. Only (b) is evidence that advertising language
# responds to the cycle. The two are separated by comparing each
# advertiser with itself (brand fixed effects).
#
# THREE PARTS:
#
# PART 1 - Does the advertiser mix change in recessions?
#   Quarterly series: number of ads, number of distinct advertisers,
#   share of ads from long-running advertisers, and the share of ads
#   in each broad sector. Each regressed on Recession + decade fixed
#   effects (HAC, 4 lags; same approach as 04c).
#   Sector = transparent keyword rules applied to the brand name
#   (e.g. "bank", "airways", "hotel"). Brands without a keyword are
#   "Other / unclassified". The rules are coarse; Part 2 does not
#   depend on them.
#
# PART 2 - Does the recession effect survive comparing each
#   advertiser with itself? (main test)
#   Ad-level regressions, one ad = one observation:
#     A. Sentiment ~ Recession
#     B. Sentiment ~ Recession | brand FE
#     C. Sentiment ~ Recession | decade FE
#     D. Sentiment ~ Recession | decade FE + sector FE
#     E. Sentiment ~ Recession | decade FE + brand FE
#   Brand FE mean each advertiser is compared only with its own ads
#   in other quarters. If the recession coefficient shrinks from C
#   to E, the recession difference reflects WHO advertises; if it
#   stays, the same advertisers change tone.
#   All five models use the SAME sample: ads from brands with at
#   least 2 ads (single-ad brands cannot be compared with
#   themselves). Standard errors are clustered by year, allowing for
#   correlation among ads within a year and across its quarters.
#
# PART 3 - Shift-share decomposition by sector
#   Splits the raw recession-minus-expansion difference into a
#   "between-sector" part (sector shares change) and a
#   "within-sector" part (sectors change tone), overall and within
#   decades (to remove long-run drift).
#
# OUTCOMES (percentage points, per advertisement):
#   Harvard positive / negative, Loughran-McDonald positive / negative
#   = category words / all words x 100 (all-words variant)
#
# SAMPLE:
#   Quality-filtered corpus: Week 4 flags plus the case-insensitive
#   blocklist fix from script 01b. 1948-2014.
#
# FILES LOADED:
#   week three data cleaning/Archived Data/week_three_ads_with_sentiment.csv
#       (Git LFS, ~268 MB; Harvard counts; canonical row order)
#   week 4 - updated quality filtered analysis/second dict and data differences/
#       csv/economist_loughran_mcdonald_sentiment.csv   (Git LFS, ~241 MB)
#   week 4 - updated quality filtered analysis/Ad_filtering/
#       week_four_quality_flags.csv   (plain text)
#   week three data cleaning/nber_recession_table.csv
#
#   Environment variables SOSC314_ADS_FILE and SOSC314_LM_FILE can
#   point to copies elsewhere (e.g. Google Drive in Colab).
#
# OUTPUTS (in this script's folder):
#   csv outputs/
#       advertiser_sector_classification_rules.csv
#       brand_sector_lookup.csv
#       quarterly_advertiser_composition.csv
#       composition_recession_tests.csv
#       sector_mean_sentiment.csv
#       within_advertiser_recession_regressions.csv
#       sector_shift_share_decomposition.csv
#   diagnostic images/
#       recession_effect_with_and_without_advertiser_fixed_effects.png
#       sector_share_recession_differences.png
#   robustness tables/
#       advertiser_composition_summary.txt
#
# REQUIRES: pandas, numpy, statsmodels, matplotlib, pyfixest
#   (pip install pyfixest)
#
# INTERPRETATION RULE:
# Descriptive associations only. Brand FE remove stable differences
# between advertisers; they do not make the recession effect causal.
# =========================================================


# ---------------------------------------------------------
# IMPORT PACKAGES
# ---------------------------------------------------------

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pyfixest as pf


# ---------------------------------------------------------
# DEFINE FILE LOCATIONS
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WEEK5_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEEK5_DIR.parent

ADS_FILE = Path(os.environ.get(
    "SOSC314_ADS_FILE",
    REPO_ROOT / "week three data cleaning" / "Archived Data" / "week_three_ads_with_sentiment.csv",
))
LM_FILE = Path(os.environ.get(
    "SOSC314_LM_FILE",
    REPO_ROOT / "week 4 - updated quality filtered analysis" / "second dict and data differences"
    / "csv" / "economist_loughran_mcdonald_sentiment.csv",
))
FLAGS_FILE = (REPO_ROOT / "week 4 - updated quality filtered analysis" / "Ad_filtering"
              / "week_four_quality_flags.csv")
RECESSION_FILE = REPO_ROOT / "week three data cleaning" / "nber_recession_table.csv"


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS
# ---------------------------------------------------------

CSV_OUTPUT_DIR = SCRIPT_DIR / "csv outputs"
IMAGE_OUTPUT_DIR = SCRIPT_DIR / "diagnostic images"
TABLE_OUTPUT_DIR = SCRIPT_DIR / "robustness tables"

for folder in [CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

HAC_LAGS = 4
LONG_RUNNING_YEARS = 10      # "long-running advertiser": ads in 10+ distinct years
MIN_ADS_PER_BRAND = 2        # needed to compare a brand with itself

WEEK4_BLOCKLIST = ["Letters are welcome", "Subscription Service", "Offer to readers"]

OUTCOMES = {
    "Harvard_Pos": "Harvard positive",
    "Harvard_Neg": "Harvard negative",
    "LM_Pos": "Loughran-McDonald positive",
    "LM_Neg": "Loughran-McDonald negative",
}

# Words stripped from brand names so that "Barclays Bank, Limited" and
# "Barclays Bank Ltd" count as the same advertiser.
LEGAL_SUFFIXES = {
    "the", "limited", "ltd", "plc", "inc", "incorporated", "co", "company", "corporation",
    "corp", "sa", "ag", "nv", "llc", "gmbh", "spa", "bv", "ab", "as", "and",
}

# Sector rules: checked in this order; first match wins. Each keyword
# is matched as a whole word in the normalised brand name, except
# entries ending in "*" which match as a word prefix.
SECTOR_RULES = [
    ("Banking & finance", [
        "bank", "banks", "banking", "banque", "banco", "bancorp", "banca", "sparkasse", "girozentrale",
        "securities", "capital", "invest*", "fund", "funds", "finance", "financial", "credit", "trust",
        "insurance", "assurance", "reinsurance", "exchange", "brokers", "stockbrokers", "merrill", "morgan",
        "rothschild", "lazard", "warburg", "barclays", "lloyds", "hsbc", "citibank", "citicorp", "citigroup",
        "chase", "goldman", "nomura", "ubs", "credit suisse", "american express", "visa", "mastercard",
        "deutsche", "paribas", "hongkong and shanghai", "standard chartered", "midland", "sanwa", "sumitomo",
        "mitsubishi bank", "fuji bank", "prudential", "allianz", "axa", "zurich", "aviva",
    ]),
    ("Air travel & hotels", [
        "air", "airline*", "airways", "aviation", "boac", "twa", "klm", "lufthansa", "swissair", "sabena",
        "alitalia", "iberia", "qantas", "cathay", "emirates", "pan am", "concorde", "hotel", "hotels",
        "hilton", "sheraton", "intercontinental", "inter continental", "carlyle", "ritz", "resort*",
        "travel", "tourism", "tourist", "cruise*",
    ]),
    ("Autos & transport", [
        "motor", "motors", "car", "cars", "automobile*", "bmw", "mercedes", "benz", "ford", "volvo",
        "jaguar", "rover", "toyota", "nissan", "honda", "audi", "volkswagen", "porsche", "lexus", "saab",
        "peugeot", "renault", "fiat", "rolls royce", "land rover", "shipping", "lines", "railway*",
        "boeing", "airbus", "courier", "dhl", "fedex", "federal express",
    ]),
    ("Technology & electronics", [
        "ibm", "computer*", "philips", "siemens", "hitachi", "fujitsu", "toshiba", "nec", "sony",
        "panasonic", "samsung", "xerox", "hewlett", "packard", "microsoft", "oracle", "intel", "cisco",
        "nokia", "ericsson", "motorola", "olivetti", "telecom*", "telephone", "telegraph",
        "communications", "electronics", "electronic", "systems", "software", "data", "digital",
        "internet", "technology", "technologies",
    ]),
    ("Energy, industry & materials", [
        "oil", "petroleum", "shell", "bp", "esso", "exxon", "mobil", "texaco", "chevron", "gulf",
        "gas", "energy", "power", "electric*", "steel", "chemical*", "ici", "rubber", "mining", "metals",
        "engineering", "industries", "industrial", "machinery", "cement", "paper", "aluminium",
    ]),
    ("Government, trade & development", [
        "government", "ministry", "republic", "kingdom", "embassy", "development", "authority",
        "board", "agency", "council", "commission", "united nations", "international labour", "world bank",
        "export", "trade", "chamber", "investment promotion", "state of", "province",
    ]),
    ("Education, jobs & conferences", [
        "university", "school", "college", "institute", "business school", "mba", "appointment*",
        "recruitment", "vacancies", "conference*", "seminar*", "course*", "education",
    ]),
    ("Media & publishing (incl. The Economist)", [
        "economist", "subscription*", "newspaper", "times", "journal", "magazine", "press", "books",
        "publishing", "publications", "publishers", "review", "letters are welcome", "offer to readers",
        "eiu", "intelligence unit",
    ]),
    ("Luxury & consumer goods", [
        "rolex", "cartier", "omega", "patek", "philippe", "tag heuer", "longines", "jewel*", "watch*",
        "whisky", "whiskey", "scotch", "cognac", "champagne", "wine*", "gin", "vodka", "beer",
        "cigar*", "tobacco", "perfume", "fashion", "burberry", "dunhill", "montblanc", "parker",
    ]),
]


# ---------------------------------------------------------
# CHECK INPUT FILES
# ---------------------------------------------------------

def check_input_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file:\n  {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.startswith("version https://git-lfs"):
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        raise RuntimeError(f"This file is a Git LFS pointer, not the real data:\n  {path}\n"
                           f"Run: git lfs pull --include=\"{rel}\"")


for path in [ADS_FILE, LM_FILE, FLAGS_FILE, RECESSION_FILE]:
    check_input_file(path)

start_time = time.time()


# ---------------------------------------------------------
# LOAD AND ALIGN THE AD-LEVEL DATA
# ---------------------------------------------------------

harvard = pd.read_csv(
    ADS_FILE,
    usecols=["DateOfIssue", "Brand", "Year_Quarter", "Total_Words", "Positiv_Count", "Negativ_Count"],
)
lm = pd.read_csv(
    LM_FILE,
    usecols=lambda c: c in {"DateOfIssue", "Brand", "Total_Words", "Positive_Count", "Negative_Count"},
)
flags = pd.read_csv(FLAGS_FILE, usecols=["row_id", "DateOfIssue", "Brand", "any_flag"])

n = len(flags)
if not (len(harvard) == len(lm) == n):
    raise ValueError(f"Row counts differ: Harvard {len(harvard):,}, LM {len(lm):,}, flags {n:,}.")

# Row order must be identical in all three files (row_id = position).
def clean_text(series):
    return series.fillna("").astype(str).str.strip().str.lower().to_numpy()


for name, frame in [("Harvard", harvard), ("Loughran-McDonald", lm)]:
    same_brand = (clean_text(frame["Brand"]) == clean_text(flags["Brand"])).mean()
    same_date = (pd.to_datetime(frame["DateOfIssue"], errors="coerce").dt.strftime("%Y-%m-%d").values
                 == flags["DateOfIssue"].astype(str).values).mean()
    print(f"{name} file aligned with flags file: brand {100 * same_brand:.2f}%, date {100 * same_date:.2f}%")
    if same_brand < 0.999 or same_date < 0.999:
        raise ValueError(f"The {name} file is not in the same row order as the flags file.")

ads = pd.DataFrame({
    "row_id": flags["row_id"].values,
    "Brand": flags["Brand"].astype("string").values,
    "Year_Quarter": harvard["Year_Quarter"].astype(str).values,
    "H_Words": harvard["Total_Words"].values,
    "H_Pos": harvard["Positiv_Count"].values,
    "H_Neg": harvard["Negativ_Count"].values,
    "L_Words": lm["Total_Words"].values,
    "L_Pos": lm["Positive_Count"].values,
    "L_Neg": lm["Negative_Count"].values,
})
del harvard, lm

# Quality filter: Week 4 flags + case-insensitive blocklist (script 01b).
blocklist = {b.lower() for b in WEEK4_BLOCKLIST}
flag_any = flags["any_flag"].astype(str).str.strip().str.lower() == "true"
flag_block = flags["Brand"].astype("string").str.strip().str.lower().isin(blocklist).fillna(False)
ads["exclude"] = (flag_any | flag_block).values

ads = ads[~ads["exclude"] & (ads["H_Words"] > 0) & (ads["L_Words"] > 0)].copy()

ads["Harvard_Pos"] = 100 * ads["H_Pos"] / ads["H_Words"]
ads["Harvard_Neg"] = 100 * ads["H_Neg"] / ads["H_Words"]
ads["LM_Pos"] = 100 * ads["L_Pos"] / ads["L_Words"]
ads["LM_Neg"] = 100 * ads["L_Neg"] / ads["L_Words"]

recession = pd.read_csv(RECESSION_FILE, usecols=["Year_Quarter", "Recession_Majority"])
recession["Recession"] = (recession["Recession_Majority"].astype(str).str.strip().str.lower() == "true").astype(int)
ads = ads.merge(recession[["Year_Quarter", "Recession"]], on="Year_Quarter", how="inner")

ads["Year"] = ads["Year_Quarter"].str[:4].astype(int)
ads["Decade"] = (ads["Year"] // 10 * 10).astype(str) + "s"

print("\n" + "=" * 90)
print("SAMPLE")
print("=" * 90)
print(f"Quality-filtered ads with words: {len(ads):,} "
      f"({ads['Year_Quarter'].min()} to {ads['Year_Quarter'].max()}, {ads['Year_Quarter'].nunique()} quarters)")


# ---------------------------------------------------------
# NORMALISE BRAND NAMES AND ASSIGN SECTORS
# ---------------------------------------------------------

def normalise_brand(name):
    text = str(name).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [w for w in text.split() if w not in LEGAL_SUFFIXES]
    return " ".join(words) if words else str(name).strip().lower()


def compile_rules(rules):
    compiled = []
    for sector, keywords in rules:
        parts = []
        for k in keywords:
            if k.endswith("*"):
                parts.append(r"\b" + re.escape(k[:-1]) + r"\w*")
            else:
                parts.append(r"\b" + re.escape(k) + r"\b")
        compiled.append((sector, re.compile("|".join(parts))))
    return compiled


COMPILED_RULES = compile_rules(SECTOR_RULES)


def assign_sector(brand_key):
    for sector, pattern in COMPILED_RULES:
        if pattern.search(brand_key):
            return sector
    return "Other / unclassified"


brand_table = pd.DataFrame({"Brand": ads["Brand"].dropna().unique()})
brand_table["Brand_Key"] = brand_table["Brand"].map(normalise_brand)
brand_table["Sector"] = brand_table["Brand_Key"].map(assign_sector)

ads = ads.merge(brand_table, on="Brand", how="left")
ads["Brand_Key"] = ads["Brand_Key"].fillna("(missing brand)")
ads["Sector"] = ads["Sector"].fillna("Other / unclassified")

brand_ads = ads.groupby("Brand_Key").size()
brand_years = ads.groupby("Brand_Key")["Year"].nunique()
ads["Brand_Ads"] = ads["Brand_Key"].map(brand_ads)
ads["Long_Running"] = ads["Brand_Key"].map(brand_years >= LONG_RUNNING_YEARS).astype(int)

sector_order = [s for s, _ in SECTOR_RULES] + ["Other / unclassified"]
sector_counts = ads["Sector"].value_counts().reindex(sector_order).fillna(0).astype(int)

print(f"Raw brand strings: {brand_table['Brand'].nunique():,} -> normalised advertisers: "
      f"{ads['Brand_Key'].nunique():,}")
print(f"Ads from advertisers with {MIN_ADS_PER_BRAND}+ ads: "
      f"{100 * (ads['Brand_Ads'] >= MIN_ADS_PER_BRAND).mean():.1f}%")
print("\nAds by sector (keyword rules on brand name):")
for sector, count in sector_counts.items():
    print(f"  {sector:<42} {count:>7,} ({100 * count / len(ads):5.1f}%)")

rules_out = pd.DataFrame([{"order": i + 1, "sector": s, "keywords": ", ".join(k)}
                          for i, (s, k) in enumerate(SECTOR_RULES)])
rules_out.to_csv(CSV_OUTPUT_DIR / "advertiser_sector_classification_rules.csv", index=False)

lookup = (ads.groupby(["Brand_Key", "Sector"]).size().rename("Ads").reset_index()
          .sort_values("Ads", ascending=False))
lookup.to_csv(CSV_OUTPUT_DIR / "brand_sector_lookup.csv", index=False)


# ---------------------------------------------------------
# PART 1: DOES THE ADVERTISER MIX CHANGE IN RECESSIONS?
# ---------------------------------------------------------

def hac_recession(frame, outcome, with_decades=True):
    data = frame.copy()
    predictors = ["Recession"]
    if with_decades:
        dummies = pd.get_dummies(data["Decade"], prefix="Decade", drop_first=True).astype(int)
        data = pd.concat([data, dummies], axis=1)
        predictors += list(dummies.columns)
    X = sm.add_constant(data[predictors].astype(float))
    res = sm.OLS(data[outcome].astype(float), X).fit().get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
    i = list(X.columns).index("Recession")
    ci = res.conf_int()[i]
    return res.params[i], res.bse[i], res.pvalues[i], ci[0], ci[1]


quarterly = ads.groupby("Year_Quarter").agg(
    Ads=("row_id", "size"),
    Distinct_Advertisers=("Brand_Key", "nunique"),
    Share_Long_Running=("Long_Running", "mean"),
    Recession=("Recession", "first"),
    Year=("Year", "first"),
    Decade=("Decade", "first"),
).reset_index()

sector_shares = pd.crosstab(ads["Year_Quarter"], ads["Sector"], normalize="index").reindex(columns=sector_order, fill_value=0)
sector_shares.columns = [f"Share: {c}" for c in sector_shares.columns]
quarterly = quarterly.merge(sector_shares.reset_index(), on="Year_Quarter", how="left").sort_values("Year_Quarter")
quarterly["Ads_per_Advertiser"] = quarterly["Ads"] / quarterly["Distinct_Advertisers"]
quarterly.round(5).to_csv(CSV_OUTPUT_DIR / "quarterly_advertiser_composition.csv", index=False)

composition_vars = (["Ads", "Distinct_Advertisers", "Ads_per_Advertiser", "Share_Long_Running"]
                    + list(sector_shares.columns))
comp_rows = []
for var in composition_vars:
    mean = quarterly[var].mean()
    for label, dec in [("Recession only", False), ("Recession + decade FE", True)]:
        est, se, p, lo, hi = hac_recession(quarterly, var, dec)
        comp_rows.append({"Measure": var, "Model": label, "Mean": mean, "Recession_diff": est,
                          "HAC_SE": se, "HAC_p_value": p, "CI_low": lo, "CI_high": hi,
                          "Diff_pct_of_mean": 100 * est / mean if mean else np.nan})
composition_tests = pd.DataFrame(comp_rows)
composition_tests.round(5).to_csv(CSV_OUTPUT_DIR / "composition_recession_tests.csv", index=False)

print("\n" + "=" * 90)
print("PART 1: ADVERTISER MIX IN RECESSION QUARTERS (decade FE, HAC 4 lags)")
print("=" * 90)
with pd.option_context("display.width", 200, "display.max_colwidth", 60):
    print(composition_tests[composition_tests["Model"] == "Recession + decade FE"][
        ["Measure", "Mean", "Recession_diff", "Diff_pct_of_mean", "HAC_p_value"]].round(4).to_string(index=False))

# Do sectors differ in tone? (Composition can only matter if they do.)
sector_sentiment = ads.groupby("Sector")[list(OUTCOMES)].mean().reindex(sector_order)
sector_sentiment["Ads"] = sector_counts
sector_sentiment.round(4).to_csv(CSV_OUTPUT_DIR / "sector_mean_sentiment.csv")

print("\nMean sentiment by sector (pp):")
print(sector_sentiment.round(3).to_string())


# ---------------------------------------------------------
# PART 2: WITHIN-ADVERTISER TEST (MAIN)
# ---------------------------------------------------------

reg = ads[ads["Brand_Ads"] >= MIN_ADS_PER_BRAND].copy()

brand_states = reg.groupby("Brand_Key")["Recession"].agg(["min", "max"])
switchers = brand_states[(brand_states["min"] == 0) & (brand_states["max"] == 1)].index
share_switch_ads = reg["Brand_Key"].isin(switchers).mean()

print("\n" + "=" * 90)
print("PART 2: DOES THE RECESSION EFFECT SURVIVE COMPARING EACH ADVERTISER WITH ITSELF?")
print("=" * 90)
print(f"Regression sample (advertisers with {MIN_ADS_PER_BRAND}+ ads): {len(reg):,} ads, "
      f"{reg['Brand_Key'].nunique():,} advertisers")
print(f"Advertisers seen in BOTH recession and non-recession quarters: {len(switchers):,}, "
      f"accounting for {100 * share_switch_ads:.1f}% of ads (these identify the brand-FE estimate)")

MODELS = {
    "A": ("A. Recession only", "{y} ~ Recession"),
    "B": ("B. + advertiser FE", "{y} ~ Recession | Brand_Key"),
    "C": ("C. + decade FE", "{y} ~ Recession | Decade"),
    "D": ("D. + decade FE + sector FE", "{y} ~ Recession | Decade + Sector"),
    "E": ("E. + decade FE + advertiser FE", "{y} ~ Recession | Decade + Brand_Key"),
}

reg_rows = []
for y, y_label in OUTCOMES.items():
    y_mean = reg[y].mean()
    for code, (label, formula) in MODELS.items():
        fit = pf.feols(formula.format(y=y), data=reg, vcov={"CRV1": "Year"})
        t = fit.tidy().loc["Recession"]
        reg_rows.append({
            "Outcome": y_label, "Model_Code": code, "Model": label,
            "Coefficient_pp": t["Estimate"], "SE_clustered_year": t["Std. Error"],
            "p_value": t["Pr(>|t|)"], "CI_low": t["2.5%"], "CI_high": t["97.5%"],
            "Outcome_mean_pp": y_mean, "Effect_pct_of_mean": 100 * t["Estimate"] / y_mean,
            "N_ads": int(fit._N),
        })

regressions = pd.DataFrame(reg_rows)
regressions.round(6).to_csv(CSV_OUTPUT_DIR / "within_advertiser_recession_regressions.csv", index=False)

with pd.option_context("display.width", 200):
    print(regressions[["Outcome", "Model", "Coefficient_pp", "p_value", "Effect_pct_of_mean", "N_ads"]]
          .round(4).to_string(index=False))

# How much of the (decade-adjusted) recession difference is composition?
share_rows = []
for y_label in OUTCOMES.values():
    r = regressions[regressions["Outcome"] == y_label].set_index("Model_Code")
    share_rows.append({
        "Outcome": y_label,
        "C_decade_FE": r.loc["C", "Coefficient_pp"],
        "D_plus_sector_FE": r.loc["D", "Coefficient_pp"],
        "E_plus_advertiser_FE": r.loc["E", "Coefficient_pp"],
        "Change_C_to_E": r.loc["E", "Coefficient_pp"] - r.loc["C", "Coefficient_pp"],
    })
change_table = pd.DataFrame(share_rows)
print("\nChange in the recession coefficient when advertisers are compared with themselves (C -> E):")
print(change_table.round(4).to_string(index=False))


# ---------------------------------------------------------
# PART 3: SHIFT-SHARE DECOMPOSITION BY SECTOR
# ---------------------------------------------------------
# Recession mean - expansion mean
#   = sum_s (w_R - w_E) * (m_R + m_E)/2      [between-sector: mix]
#   + sum_s (w_R + w_E)/2 * (m_R - m_E)      [within-sector: tone]
# with w = sector share of ads, m = sector mean sentiment. Exact.
# Within-decade version: computed separately in each decade that has
# both recession and expansion ads, then averaged with weights equal
# to the number of recession ads in the decade.
# ---------------------------------------------------------

def shift_share(frame, y):
    rec, exp = frame[frame["Recession"] == 1], frame[frame["Recession"] == 0]
    w_r = rec["Sector"].value_counts(normalize=True)
    w_e = exp["Sector"].value_counts(normalize=True)
    m_r = rec.groupby("Sector")[y].mean()
    m_e = exp.groupby("Sector")[y].mean()
    sectors = sorted(set(w_r.index) | set(w_e.index))
    between = within = 0.0
    for s in sectors:
        wr, we = w_r.get(s, 0.0), w_e.get(s, 0.0)
        mr = m_r.get(s, np.nan)
        me = m_e.get(s, np.nan)
        if np.isnan(mr):
            mr = me
        if np.isnan(me):
            me = mr
        between += (wr - we) * (mr + me) / 2
        within += (wr + we) / 2 * (mr - me)
    total = rec[y].mean() - exp[y].mean()
    return total, between, within


ss_rows = []
for y, y_label in OUTCOMES.items():
    total, between, within = shift_share(ads, y)
    ss_rows.append({"Outcome": y_label, "Scope": "All years pooled", "Total_diff_pp": total,
                    "Between_sector_mix_pp": between, "Within_sector_tone_pp": within})
    parts, weights = [], []
    for decade, d in ads.groupby("Decade"):
        if d["Recession"].nunique() < 2:
            continue
        parts.append(shift_share(d, y))
        weights.append(int(d["Recession"].sum()))
    parts = np.array(parts)
    w = np.array(weights) / np.sum(weights)
    ss_rows.append({"Outcome": y_label, "Scope": "Within decades (weighted average)",
                    "Total_diff_pp": float(parts[:, 0] @ w),
                    "Between_sector_mix_pp": float(parts[:, 1] @ w),
                    "Within_sector_tone_pp": float(parts[:, 2] @ w)})

shift = pd.DataFrame(ss_rows)
# Share of the total explained by the mix; left blank when the total is
# too close to zero for a percentage to mean anything.
shift["Mix_share_of_total_pct"] = np.where(
    shift["Total_diff_pp"].abs() >= 0.005,
    100 * shift["Between_sector_mix_pp"] / shift["Total_diff_pp"],
    np.nan,
)
shift.round(6).to_csv(CSV_OUTPUT_DIR / "sector_shift_share_decomposition.csv", index=False)

print("\n" + "=" * 90)
print("PART 3: SHIFT-SHARE DECOMPOSITION BY SECTOR (pp)")
print("=" * 90)
with pd.option_context("display.width", 200):
    print(shift.round(4).to_string(index=False))


# ---------------------------------------------------------
# FIGURE 1: RECESSION EFFECT WITH AND WITHOUT ADVERTISER FE
# ---------------------------------------------------------

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})

MODEL_STYLE = {
    "A": ("#b5b4ae", "o"), "B": ("#7c7b76", "o"),
    "C": ("#2a78d6", "s"), "D": ("#1baf7a", "D"), "E": ("#eb6834", "^"),
}

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharey=True)
for ax, y_label in zip(axes.flatten(), OUTCOMES.values()):
    sub = regressions[regressions["Outcome"] == y_label].reset_index(drop=True)
    for i, r in sub.iterrows():
        yy = len(sub) - 1 - i
        scale = 100 / r["Outcome_mean_pp"]
        colour, marker = MODEL_STYLE[r["Model_Code"]]
        ax.plot([r["CI_low"] * scale, r["CI_high"] * scale], [yy, yy], color=colour, linewidth=2,
                solid_capstyle="round")
        ax.plot(r["Effect_pct_of_mean"], yy, marker=marker, markersize=7, color=colour,
                markeredgecolor="white", linestyle="none")
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.axhline(2.5, color="#b5b4ae", linewidth=0.8, linestyle="--")
    ax.set_title(y_label, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(axis="x", color="#eeede9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

labels = [MODELS[c][0] for c in MODELS]
axes[0, 0].set_yticks(range(len(labels)))
axes[0, 0].set_yticklabels(labels[::-1])
for ax in axes[1]:
    ax.set_xlabel("Recession effect, % of mean (95% CI, SE clustered by year)")
fig.suptitle("Is the recession difference about WHO advertises or HOW they write?\n"
             "Ad-level models; advertiser FE compare each advertiser only with itself",
             x=0.02, ha="left", y=1.02, fontsize=11)
fig.tight_layout()
fig1_path = IMAGE_OUTPUT_DIR / "recession_effect_with_and_without_advertiser_fixed_effects.png"
fig.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# FIGURE 2: SECTOR SHARE DIFFERENCES IN RECESSIONS
# ---------------------------------------------------------

sec = composition_tests[(composition_tests["Model"] == "Recession + decade FE")
                        & composition_tests["Measure"].str.startswith("Share: ")].copy()
sec["Sector"] = sec["Measure"].str.replace("Share: ", "", regex=False)
sec = sec.sort_values("Recession_diff").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(sec) + 1.3))
for i, r in sec.iterrows():
    ax.plot([100 * r["CI_low"], 100 * r["CI_high"]], [i, i], color="#2a78d6", linewidth=2,
            solid_capstyle="round")
    ax.plot(100 * r["Recession_diff"], i, "o", color="#2a78d6", markeredgecolor="white", markersize=7)
ax.axvline(0, color="#52514e", linewidth=0.8)
ax.set_yticks(range(len(sec)))
ax.set_yticklabels([f"{s} ({100 * m:.1f}% of ads)" for s, m in zip(sec["Sector"], sec["Mean"])])
ax.set_xlabel("Change in the sector's share of ads in recession quarters (percentage points)\n"
              "HAC 95% CI, decade fixed effects; sectors from keyword rules on brand names")
ax.set_title("Does the mix of advertisers change in recessions?", loc="left", fontsize=10, color="#0b0b0b")
ax.grid(axis="x", color="#eeede9", linewidth=0.6)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig2_path = IMAGE_OUTPUT_DIR / "sector_share_recession_differences.png"
fig.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# WRITTEN SUMMARY
# ---------------------------------------------------------

L = []
L.append("WEEK 5 DIAGNOSTIC 06: ADVERTISER COMPOSITION vs WITHIN-ADVERTISER TONE")
L.append("=" * 78)
L.append("")
L.append("Question (Week 4 feedback): does the recession difference reflect a change in")
L.append("WHICH firms advertise, or the SAME firms changing tone?")
L.append("")
L.append(f"Quality-filtered ads: {len(ads):,}; normalised advertisers: {ads['Brand_Key'].nunique():,}.")
L.append(f"Regression sample (advertisers with {MIN_ADS_PER_BRAND}+ ads): {len(reg):,} ads, "
         f"{reg['Brand_Key'].nunique():,} advertisers.")
L.append(f"Advertisers present in both recession and other quarters: {len(switchers):,} "
         f"({100 * share_switch_ads:.1f}% of regression-sample ads).")
L.append(f"Sector rules classify {100 * (1 - sector_counts['Other / unclassified'] / len(ads)):.1f}% of ads; "
         "the rest are 'Other / unclassified'.")
L.append("")
L.append("-" * 78)
L.append("PART 1: ADVERTISER MIX IN RECESSION QUARTERS (Recession + decade FE, HAC 4 lags)")
L.append("-" * 78)
for _, r in composition_tests[composition_tests["Model"] == "Recession + decade FE"].iterrows():
    star = "*" if r["HAC_p_value"] < 0.05 else " "
    if r["Measure"].startswith("Share"):
        L.append(f"{r['Measure']:<52} mean {100 * r['Mean']:5.1f}% | recession diff "
                 f"{100 * r['Recession_diff']:+.2f} pp | p={r['HAC_p_value']:.3f}{star}")
    else:
        L.append(f"{r['Measure']:<52} mean {r['Mean']:8.2f} | recession diff "
                 f"{r['Recession_diff']:+.3f} ({r['Diff_pct_of_mean']:+.1f}%) | p={r['HAC_p_value']:.3f}{star}")
n_sector_tests = len(sector_order)
L.append(f"({n_sector_tests} sector shares tested: about {0.05 * n_sector_tests:.1f} would reach p < .05 by chance.)")
L.append("")
L.append("Mean sentiment by sector (pp) - composition can only matter if sectors differ:")
L.append(sector_sentiment.round(3).to_string())
L.append("")
L.append("-" * 78)
L.append("PART 2: RECESSION EFFECT WITH AND WITHOUT ADVERTISER FIXED EFFECTS")
L.append("(ad-level; SE clustered by year; same sample in every model)")
L.append("-" * 78)
for y_label in OUTCOMES.values():
    L.append(y_label)
    for _, r in regressions[regressions["Outcome"] == y_label].iterrows():
        star = "*" if r["p_value"] < 0.05 else " "
        L.append(f"   {r['Model']:<34} {r['Coefficient_pp']:+.4f} pp ({r['Effect_pct_of_mean']:+6.1f}% of mean) "
                 f"p={r['p_value']:.3f}{star}")
    L.append("")
L.append("-" * 78)
L.append("PART 3: SHIFT-SHARE DECOMPOSITION BY SECTOR (pp)")
L.append("-" * 78)
L.append(shift.round(4).to_string(index=False))
L.append("")
L.append("-" * 78)
L.append("HOW TO READ THIS")
L.append("-" * 78)
L.append("- Part 1 asks whether the MIX changes. A sector whose share rises or falls in")
L.append("  recessions is a candidate composition channel, but only if that sector's")
L.append("  typical tone differs from others (see mean sentiment by sector).")
L.append("- Part 2 is the direct test. Model C compares recession and other quarters")
L.append("  within decades. Model E additionally compares each advertiser only with")
L.append("  itself. If C and E are similar, composition is not what drives the")
L.append("  recession difference; if E moves toward zero, it is.")
L.append("- Model D (sector FE) shows how much of any composition effect is broad")
L.append("  industry mix, as opposed to individual firms.")
L.append("- Part 3 gives the same idea as an accounting identity at sector level: the")
L.append("  'between-sector' term is what the difference would be if every sector kept")
L.append("  its tone and only the sector mix changed.")
L.append("- Ad-level estimates weight each ad equally, so they need not equal the")
L.append("  quarterly averages in scripts 03a-04d, which weight each quarter equally.")
L.append("- Sector rules are keyword-based and coarse; Part 2's advertiser FE do not")
L.append("  depend on them. Brand names in the source data are sometimes wrong (see 01b).")
L.append("- All results are descriptive associations, not causal effects.")

summary_path = TABLE_OUTPUT_DIR / "advertiser_composition_summary.txt"
summary_path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\n" + "=" * 90)
print(f"ADVERTISER COMPOSITION DIAGNOSTIC COMPLETE ({(time.time() - start_time) / 60:.1f} minutes)")
print("=" * 90)
for p in sorted(CSV_OUTPUT_DIR.glob("*.csv")) + [fig1_path, fig2_path, summary_path]:
    print(f"  {p.relative_to(REPO_ROOT)}")
