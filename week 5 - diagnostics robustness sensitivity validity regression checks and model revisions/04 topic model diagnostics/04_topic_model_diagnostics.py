# 04 - WEEK 5 DIAGNOSTIC (TOPIC MODEL):
# IS AN LDA TOPIC MODEL STABLE ENOUGH TO ADD INFORMATION BEYOND THE DICTIONARY ANALYSIS?


# PURPOSE:
# Fit a small, bounded set of LDA topic models to the quality-
# filtered advertisement text and check whether the topics are
# stable across random seeds. If they are, use them to describe
# which kinds of advertising are more or less common in recession
# quarters. If they are not, report that and keep the dictionary
# approach as the main measurement.
#
# This is a DIAGNOSTIC and a complementary text representation.
# It does not replace the Harvard / Loughran-McDonald analysis.
#
# MAIN DIAGNOSTIC QUESTIONS:
# 1. Stability: if LDA is re-run with a different random seed, do
#    the same topics come back? (A topic that only appears under
#    one seed is an artefact of the algorithm, not a pattern in the
#    text.)
# 2. Number of topics: are topics more stable with k = 10 or 20?
# 3. Only for stable topics: does topic prevalence differ between
#    recession and non-recession quarters, once long-run drift is
#    controlled with decade fixed effects?
#
# WHY THIS DIAGNOSTIC MATTERS:
# The dictionary measures show a small recession gap in positive
# language that mostly disappears once long-run drift is
# controlled (script 04c). One possible reason is COMPOSITION:
# the mix of advertisers (finance, travel, luxury, recruitment...)
# may shift over the cycle, and each category has its own typical
# tone. Topics give a rough, data-driven picture of that mix.
# LDA results are known to vary with the random seed and the number
# of topics, so stability must be checked before any topic is
# interpreted.
#
# DESIGN (kept deliberately small):
#   Corpus   : ads that pass the quality filter (Week 4 flags plus
#              the case-insensitive blocklist fix from script 01b)
#   Sample   : 20,000 ads drawn at random (seed 314) for fitting;
#              a separate 2,000 held out for perplexity
#   Text     : lower-case words of 3+ letters, English stopwords and
#              a few publication words removed; words in fewer than
#              20 ads or more than 50% of ads dropped; 5,000-word
#              vocabulary
#   Models   : k = 10 and k = 20 topics, random seeds 1 and 2
#              (4 fits)
#   Stability: topics from seed 1 and seed 2 are paired one-to-one
#              (Hungarian matching on cosine similarity of the
#              topic-word distributions). Reported per topic:
#              cosine similarity and top-10-word overlap. Reported
#              per ad: share of ads whose dominant topic is the same
#              under both seeds after pairing.
#   Stable   : a topic counts as stable if its matched cosine
#              similarity is at least 0.80 (a conventional, not
#              universal, cut-off; the full distribution is saved).
#   Recession: for the more stable k, quarterly mean topic share
#              (268 quarters) regressed on Recession, and on
#              Recession + decade fixed effects, HAC 4 lags.
#              Only topics that are stable are interpreted.
#
# FILES LOADED:
#   week three data cleaning/Archived Data/
#       week_three_ads_with_sentiment.csv   (Git LFS, ~268 MB; the
#                                            canonical row order)
#   week 4 - updated quality filtered analysis/Ad_filtering/
#       week_four_quality_flags.csv          (plain text)
#   week three data cleaning/nber_recession_table.csv
#
#   If the ads file is kept elsewhere (e.g. Google Drive in Colab),
#   set the environment variable SOSC314_ADS_FILE to its path.
#
# OUTPUTS (in this script's folder):
#   csv outputs/
#       topic_top_words_all_runs.csv
#       topic_stability_matched_pairs.csv
#       topic_stability_summary.csv
#       topic_prevalence_by_quarter.csv
#       topic_recession_prevalence_tests.csv
#   diagnostic images/
#       topic_stability_across_seeds.png
#       topic_prevalence_recession_effects.png
#   robustness tables/
#       topic_model_diagnostic_summary.txt
#
# RUNTIME: about 5-15 minutes in Colab (4 LDA fits).
#
# INTERPRETATION RULE:
# Topics are labelled by their top words only; labels are the
# reader's interpretation, not something the model knows. All
# recession comparisons are descriptive associations.

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


# DEFINE FILE LOCATIONS

SCRIPT_DIR = Path(__file__).resolve().parent
WEEK5_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEEK5_DIR.parent

ADS_FILE = Path(os.environ.get(
    "SOSC314_ADS_FILE",
    REPO_ROOT / "week three data cleaning" / "Archived Data" / "week_three_ads_with_sentiment.csv",
))

FLAGS_FILE = (
    REPO_ROOT
    / "week 4 - updated quality filtered analysis"
    / "Ad_filtering"
    / "week_four_quality_flags.csv"
)

RECESSION_FILE = REPO_ROOT / "week three data cleaning" / "nber_recession_table.csv"


# ---------------------------------------------------------
# DEFINE OUTPUT FOLDERS

CSV_OUTPUT_DIR = SCRIPT_DIR / "csv outputs"
IMAGE_OUTPUT_DIR = SCRIPT_DIR / "diagnostic images"
TABLE_OUTPUT_DIR = SCRIPT_DIR / "robustness tables"

for folder in [CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TABLE_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS

SAMPLE_SEED = 314
FIT_SAMPLE_SIZE = int(os.environ.get("SOSC314_FIT_SAMPLE", 20000))
HOLDOUT_SIZE = int(os.environ.get("SOSC314_HOLDOUT", 2000))

TOPIC_COUNTS = [10, 20]
LDA_SEEDS = [1, 2]
LDA_MAX_ITER = 20

MIN_DOC_FREQ = int(os.environ.get("SOSC314_MIN_DF", 20))
MAX_DOC_SHARE = 0.50
VOCAB_SIZE = 5000
TOP_N_WORDS = 10

STABLE_COSINE = 0.80
HAC_LAGS = 4

# Week 4 blocklist (from week4_initial_flags.py), matched
# case-insensitively as in script 01b.
WEEK4_BLOCKLIST = ["Letters are welcome", "Subscription Service", "Offer to readers"]

# Words that say "this is The Economist" rather than anything about
# the advertisement, plus common OCR / web fragments.
EXTRA_STOP_WORDS = {
    "economist", "www", "com", "http", "https", "html", "org", "net",
    "tel", "fax", "ltd", "inc", "plc", "llc", "co", "per", "cent",
}


# ---------------------------------------------------------
# CHECK INPUT FILES

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
        raise RuntimeError(
            f"This file is a Git LFS pointer, not the real data:\n  {path}\n"
            f"Run: git lfs pull --include=\"{rel}\""
        )


for path in [ADS_FILE, FLAGS_FILE, RECESSION_FILE]:
    check_input_file(path)


# LOAD ADS, FLAGS AND RECESSION DATES

start_time = time.time()

ads = pd.read_csv(
    ADS_FILE,
    usecols=["DateOfIssue", "OCR_GoogleVision_original", "Year_Quarter"],
    dtype={"OCR_GoogleVision_original": "string"},
)
# row_id = position in the canonical file. Never sort before this.
ads = ads.reset_index(drop=True)
ads["row_id"] = ads.index

flags = pd.read_csv(FLAGS_FILE, usecols=["row_id", "DateOfIssue", "Brand", "any_flag"])
flags["any_flag"] = flags["any_flag"].astype(str).str.strip().str.lower() == "true"
brand_clean = flags["Brand"].astype("string").str.strip().str.lower()
flags["blocklist_ci"] = brand_clean.isin([b.lower() for b in WEEK4_BLOCKLIST]).fillna(False).astype(bool)
flags["exclude"] = flags["any_flag"] | flags["blocklist_ci"]

if len(ads) != len(flags):
    raise ValueError(f"Ads file has {len(ads):,} rows but flags file has {len(flags):,}; "
                     "they must describe the same records in the same order.")

date_mismatch = (ads["DateOfIssue"].astype(str).values != flags["DateOfIssue"].astype(str).values).mean()
if date_mismatch > 0:
    raise ValueError(f"{100 * date_mismatch:.2f}% of rows have different dates in the ads and "
                     "flags files; the row order does not match.")

ads = ads.merge(flags[["row_id", "exclude"]], on="row_id", how="left")

recession = pd.read_csv(RECESSION_FILE, usecols=["Year_Quarter", "Recession_Majority"])
recession["Recession"] = (
    recession["Recession_Majority"].astype(str).str.strip().str.lower() == "true"
).astype(int)

print("\n" + "=" * 90)
print("CORPUS")
print("=" * 90)
print(f"All ads:                  {len(ads):,}")
print(f"Excluded by quality flags: {int(ads['exclude'].sum()):,}")

corpus = ads[~ads["exclude"]].copy()
corpus["text"] = corpus["OCR_GoogleVision_original"].fillna("")
corpus = corpus[corpus["text"].str.len() > 0]
print(f"Quality-filtered ads:      {len(corpus):,}")


# DRAW THE FITTING SAMPLE AND HOLD-OUT

n_needed = min(len(corpus), FIT_SAMPLE_SIZE + HOLDOUT_SIZE)
drawn = corpus.sample(n=n_needed, random_state=SAMPLE_SEED)
fit_docs = drawn.iloc[:min(FIT_SAMPLE_SIZE, len(drawn) - HOLDOUT_SIZE)].copy()
holdout_docs = drawn.iloc[len(fit_docs):].copy()

print(f"Fitting sample: {len(fit_docs):,} ads | held-out sample: {len(holdout_docs):,} ads")
print(f"Quarters represented in fitting sample: {fit_docs['Year_Quarter'].nunique()}")


# BAG-OF-WORDS REPRESENTATION

stop_words = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS)

vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    stop_words=stop_words,
    min_df=MIN_DOC_FREQ,
    max_df=MAX_DOC_SHARE,
    max_features=VOCAB_SIZE,
)

X_fit = vectorizer.fit_transform(fit_docs["text"])
X_holdout = vectorizer.transform(holdout_docs["text"]) if len(holdout_docs) else None
vocab = np.array(vectorizer.get_feature_names_out())

empty_docs = int((X_fit.sum(axis=1) == 0).sum())
print(f"Vocabulary: {len(vocab):,} words | ads with no remaining words: {empty_docs:,}")


# FIT THE MODELS


models = {}
doc_topics = {}
top_word_rows = []
fit_rows = []

for k in TOPIC_COUNTS:
    for seed in LDA_SEEDS:
        t0 = time.time()
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method="online",
            max_iter=LDA_MAX_ITER,
            random_state=seed,
            n_jobs=-1,
        )
        theta = lda.fit_transform(X_fit)
        models[(k, seed)] = lda
        doc_topics[(k, seed)] = theta

        perplexity = lda.perplexity(X_holdout) if X_holdout is not None and X_holdout.shape[0] else np.nan
        fit_rows.append({"k": k, "seed": seed, "held_out_perplexity": perplexity,
                         "fit_seconds": time.time() - t0})
        print(f"Fitted k={k:>2}, seed={seed}: held-out perplexity {perplexity:,.0f} "
              f"({time.time() - t0:.0f}s)")

        topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        prevalence = theta.mean(axis=0)
        for t in range(k):
            top = np.argsort(topic_word[t])[::-1][:TOP_N_WORDS]
            top_word_rows.append({
                "k": k, "seed": seed, "topic": t,
                "mean_share_in_sample": prevalence[t],
                "top_words": ", ".join(vocab[top]),
            })

top_words = pd.DataFrame(top_word_rows)
fit_summary = pd.DataFrame(fit_rows)


# STABILITY ACROSS SEEDS
# Pair each topic from seed 1 with exactly one topic from seed 2 so
# that total cosine similarity is as high as possible (Hungarian
# algorithm). check how similar each pair is.

def normalised_topics(lda):
    return lda.components_ / lda.components_.sum(axis=1, keepdims=True)


def cosine_matrix(a, b):
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_n @ b_n.T


pair_rows = []
stability_rows = []
alignment = {}

seed_a, seed_b = LDA_SEEDS

for k in TOPIC_COUNTS:
    phi_a = normalised_topics(models[(k, seed_a)])
    phi_b = normalised_topics(models[(k, seed_b)])
    sim = cosine_matrix(phi_a, phi_b)

    rows, cols = linear_sum_assignment(-sim)
    alignment[k] = dict(zip(rows, cols))

    for a, b in zip(rows, cols):
        top_a = set(vocab[np.argsort(phi_a[a])[::-1][:TOP_N_WORDS]])
        top_b = set(vocab[np.argsort(phi_b[b])[::-1][:TOP_N_WORDS]])
        pair_rows.append({
            "k": k,
            f"topic_seed{seed_a}": a,
            f"topic_seed{seed_b}": b,
            "cosine_similarity": sim[a, b],
            "top10_overlap_words": len(top_a & top_b),
            "top10_jaccard": len(top_a & top_b) / len(top_a | top_b),
            "stable": sim[a, b] >= STABLE_COSINE,
            f"top_words_seed{seed_a}": ", ".join(vocab[np.argsort(phi_a[a])[::-1][:TOP_N_WORDS]]),
            f"top_words_seed{seed_b}": ", ".join(vocab[np.argsort(phi_b[b])[::-1][:TOP_N_WORDS]]),
        })

    # Document-level agreement: same dominant topic under both seeds
    # after pairing.
    dom_a = doc_topics[(k, seed_a)].argmax(axis=1)
    dom_b = doc_topics[(k, seed_b)].argmax(axis=1)
    inverse = {b: a for a, b in alignment[k].items()}
    dom_b_mapped = np.array([inverse[t] for t in dom_b])
    doc_agreement = float(np.mean(dom_a == dom_b_mapped))

    k_pairs = [r for r in pair_rows if r["k"] == k]
    cos_values = np.array([r["cosine_similarity"] for r in k_pairs])
    stability_rows.append({
        "k": k,
        "mean_matched_cosine": cos_values.mean(),
        "median_matched_cosine": np.median(cos_values),
        "min_matched_cosine": cos_values.min(),
        "stable_topics": int((cos_values >= STABLE_COSINE).sum()),
        "share_stable_topics": float((cos_values >= STABLE_COSINE).mean()),
        "mean_top10_overlap_words": np.mean([r["top10_overlap_words"] for r in k_pairs]),
        "dominant_topic_agreement_across_seeds": doc_agreement,
        "held_out_perplexity_seed1": fit_summary.query("k == @k and seed == @seed_a")["held_out_perplexity"].iloc[0],
        "held_out_perplexity_seed2": fit_summary.query("k == @k and seed == @seed_b")["held_out_perplexity"].iloc[0],
    })

pairs = pd.DataFrame(pair_rows)
stability = pd.DataFrame(stability_rows)

print("\n" + "=" * 90)
print("STABILITY ACROSS RANDOM SEEDS")
print("=" * 90)
with pd.option_context("display.width", 200):
    print(stability.round(3).to_string(index=False))

# Choose the k whose topics are more stable (share of stable topics,
# then mean cosine) for the substantive comparison.
best = stability.sort_values(["share_stable_topics", "mean_matched_cosine"], ascending=False).iloc[0]
PRIMARY_K = int(best["k"])
print(f"\nMore stable specification: k = {PRIMARY_K}")


# TOPIC PREVALENCE BY QUARTER (primary k, seed 1)

theta = doc_topics[(PRIMARY_K, seed_a)]
topic_cols = [f"topic_{t}" for t in range(PRIMARY_K)]

shares = pd.DataFrame(theta, columns=topic_cols, index=fit_docs.index)
shares["Year_Quarter"] = fit_docs["Year_Quarter"].values

quarterly = shares.groupby("Year_Quarter")[topic_cols].mean()
quarterly["Ads_in_sample"] = shares.groupby("Year_Quarter").size()
quarterly = quarterly.reset_index().merge(recession[["Year_Quarter", "Recession"]], on="Year_Quarter", how="inner")
quarterly = quarterly.sort_values("Year_Quarter").reset_index(drop=True)
quarterly["Year"] = quarterly["Year_Quarter"].str[:4].astype(int)
quarterly["Decade"] = (quarterly["Year"] // 10 * 10).astype(str) + "s"

print(f"\nQuarters with sampled ads: {len(quarterly)} | ads per quarter: "
      f"median {quarterly['Ads_in_sample'].median():.0f}, min {quarterly['Ads_in_sample'].min()}")


# RECESSION vs NON-RECESSION TOPIC PREVALENCE

def fit_hac(data, outcome, predictors):
    X = sm.add_constant(data[predictors].astype(float))
    y = data[outcome].astype(float)
    ols_model = sm.OLS(y, X).fit()
    return ols_model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS), list(X.columns)


decade_dummies = pd.get_dummies(quarterly["Decade"], prefix="Decade", drop_first=True).astype(int)
decade_columns = list(decade_dummies.columns)
model_data = pd.concat([quarterly, decade_dummies], axis=1)

primary_pairs = pairs[pairs["k"] == PRIMARY_K].set_index(f"topic_seed{seed_a}")
primary_top = top_words[(top_words["k"] == PRIMARY_K) & (top_words["seed"] == seed_a)].set_index("topic")

test_rows = []
for t, col in enumerate(topic_cols):
    mean_share = model_data[col].mean()
    for model_name, predictors in [("Recession only", ["Recession"]),
                                   ("Recession + decade FE", ["Recession"] + decade_columns)]:
        res, names = fit_hac(model_data, col, predictors)
        i = names.index("Recession")
        ci = res.conf_int()[i]
        test_rows.append({
            "topic": t,
            "top_words": primary_top.loc[t, "top_words"],
            "stable_across_seeds": bool(primary_pairs.loc[t, "stable"]),
            "matched_cosine": primary_pairs.loc[t, "cosine_similarity"],
            "mean_topic_share": mean_share,
            "model": model_name,
            "recession_coef_share_points": res.params[i],
            "HAC_SE": res.bse[i],
            "HAC_p_value": res.pvalues[i],
            "CI_low": ci[0],
            "CI_high": ci[1],
            "effect_pct_of_mean": 100 * res.params[i] / mean_share,
        })

tests = pd.DataFrame(test_rows)

n_tests = int((tests["model"] == "Recession + decade FE").sum())
print("\n" + "=" * 90)
print(f"RECESSION vs NON-RECESSION TOPIC PREVALENCE (k = {PRIMARY_K}, HAC 4 lags)")
print("=" * 90)
with pd.option_context("display.width", 220, "display.max_colwidth", 60):
    print(tests[tests["model"] == "Recession + decade FE"][[
        "topic", "stable_across_seeds", "mean_topic_share", "recession_coef_share_points",
        "HAC_p_value", "effect_pct_of_mean", "top_words",
    ]].round(4).to_string(index=False))
print(f"\n{n_tests} topics tested: at p < .05, about {0.05 * n_tests:.1f} would be 'significant' "
      "by chance alone even if no topic differed.")


# SAVE CSV OUTPUTS

paths = {
    "top_words": CSV_OUTPUT_DIR / "topic_top_words_all_runs.csv",
    "pairs": CSV_OUTPUT_DIR / "topic_stability_matched_pairs.csv",
    "stability": CSV_OUTPUT_DIR / "topic_stability_summary.csv",
    "quarterly": CSV_OUTPUT_DIR / "topic_prevalence_by_quarter.csv",
    "tests": CSV_OUTPUT_DIR / "topic_recession_prevalence_tests.csv",
}

top_words.round(5).to_csv(paths["top_words"], index=False)
pairs.round(4).to_csv(paths["pairs"], index=False)
stability.round(4).to_csv(paths["stability"], index=False)
quarterly.drop(columns=["Decade"]).round(5).to_csv(paths["quarterly"], index=False)
tests.round(5).to_csv(paths["tests"], index=False)


# FIGURE 1: STABILITY ACROSS SEEDS

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})

K_STYLE = {10: {"color": "#2a78d6", "marker": "o"}, 20: {"color": "#eb6834", "marker": "s"}}

fig, ax = plt.subplots(figsize=(8, 4))
for k in TOPIC_COUNTS:
    values = np.sort(pairs.loc[pairs["k"] == k, "cosine_similarity"].values)[::-1]
    rank_share = (np.arange(len(values)) + 0.5) / len(values)
    row = stability[stability["k"] == k].iloc[0]
    ax.plot(rank_share, values, color=K_STYLE[k]["color"], marker=K_STYLE[k]["marker"],
            markersize=5, linewidth=2, markeredgecolor="white",
            label=f"k = {k}: {int(row['stable_topics'])}/{k} stable, "
                  f"{100 * row['dominant_topic_agreement_across_seeds']:.0f}% of ads keep their topic")
ax.axhline(STABLE_COSINE, color="#52514e", linewidth=0.8, linestyle="--")
ax.text(1.0, STABLE_COSINE + 0.01, f"stability cut-off ({STABLE_COSINE:.2f})",
        ha="right", va="bottom", fontsize=8, color="#52514e")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Topics, ranked from most to least stable (share of all topics)")
ax.set_ylabel("Cosine similarity with matched topic\nfrom the other random seed")
ax.set_title("Do the same topics come back with a different random seed?",
             loc="left", fontsize=10, color="#0b0b0b")
ax.grid(axis="y", color="#eeede9", linewidth=0.6)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=8.5)
fig.tight_layout()
stability_fig_path = IMAGE_OUTPUT_DIR / "topic_stability_across_seeds.png"
fig.savefig(stability_fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# FIGURE 2: RECESSION EFFECT ON TOPIC PREVALENCE
# Recession + decade FE, primary k. Effect as % of the topic's mean
# share. Unstable topics drawn hollow and greyed.

fe = tests[tests["model"] == "Recession + decade FE"].copy()
fe = fe.sort_values("effect_pct_of_mean").reset_index(drop=True)
fe["label"] = [f"T{t}: " + ", ".join(w.split(", ")[:4]) for t, w in zip(fe["topic"], fe["top_words"])]

fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(fe) + 1.4))
for y, r in fe.iterrows():
    scale = 100 / r["mean_topic_share"]
    colour = "#2a78d6" if r["stable_across_seeds"] else "#b5b4ae"
    ax.plot([r["CI_low"] * scale, r["CI_high"] * scale], [y, y], color=colour, linewidth=2,
            solid_capstyle="round")
    ax.plot(r["effect_pct_of_mean"], y, marker="o", markersize=6, linestyle="none",
            color=colour if r["stable_across_seeds"] else "white",
            markeredgecolor=colour, markeredgewidth=1.5)
ax.axvline(0, color="#52514e", linewidth=0.8)
ax.set_yticks(range(len(fe)))
ax.set_yticklabels(fe["label"], fontsize=8)
ax.set_xlabel("Recession difference in topic share, % of the topic's mean (HAC 95% CI, decade FE)")
ax.set_title(f"Topic prevalence in recession vs other quarters (k = {PRIMARY_K})\n"
             f"Filled blue = stable across seeds; hollow grey = unstable, do not interpret",
             loc="left", fontsize=10, color="#0b0b0b")
ax.grid(axis="x", color="#eeede9", linewidth=0.6)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
prevalence_fig_path = IMAGE_OUTPUT_DIR / "topic_prevalence_recession_effects.png"
fig.savefig(prevalence_fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# WRITTEN SUMMARY

lines = []
lines.append("WEEK 5 DIAGNOSTIC 04: LDA TOPIC MODEL STABILITY")
lines.append("=" * 78)
lines.append("")
lines.append(f"Quality-filtered corpus: {len(corpus):,} ads "
             f"(excluded {int(ads['exclude'].sum()):,} flagged records).")
lines.append(f"Fitting sample: {len(fit_docs):,} random ads (seed {SAMPLE_SEED}); "
             f"held-out sample: {len(holdout_docs):,}.")
lines.append(f"Vocabulary: {len(vocab):,} words (3+ letters, stopwords removed, "
             f"min {MIN_DOC_FREQ} ads, max {int(100 * MAX_DOC_SHARE)}% of ads).")
lines.append(f"Models: k = {TOPIC_COUNTS}, seeds = {LDA_SEEDS}, online LDA, {LDA_MAX_ITER} passes.")
lines.append(f"Stable topic: matched cosine similarity >= {STABLE_COSINE:.2f}.")
lines.append("")
lines.append("-" * 78)
lines.append("STABILITY ACROSS RANDOM SEEDS")
lines.append("-" * 78)
for _, r in stability.iterrows():
    lines.append(f"k = {int(r['k'])}: {int(r['stable_topics'])}/{int(r['k'])} topics stable; "
                 f"mean matched cosine {r['mean_matched_cosine']:.3f} (min {r['min_matched_cosine']:.3f}); "
                 f"mean top-10 overlap {r['mean_top10_overlap_words']:.1f} words; "
                 f"{100 * r['dominant_topic_agreement_across_seeds']:.1f}% of ads keep the same dominant topic; "
                 f"held-out perplexity {r['held_out_perplexity_seed1']:,.0f} / {r['held_out_perplexity_seed2']:,.0f}")
lines.append(f"More stable specification used below: k = {PRIMARY_K}.")
lines.append("")
lines.append("-" * 78)
lines.append(f"TOPICS (k = {PRIMARY_K}, seed {seed_a}) WITH STABILITY")
lines.append("-" * 78)
for t in range(PRIMARY_K):
    p = primary_pairs.loc[t]
    lines.append(f"T{t:<2} share {primary_top.loc[t, 'mean_share_in_sample']:.3f} | cosine "
                 f"{p['cosine_similarity']:.2f} {'STABLE' if p['stable'] else 'unstable'} | "
                 f"{primary_top.loc[t, 'top_words']}")
lines.append("")
lines.append("-" * 78)
lines.append("RECESSION vs OTHER QUARTERS (quarterly mean topic share, HAC 4 lags)")
lines.append("-" * 78)
for _, r in tests.sort_values(["topic", "model"]).iterrows():
    star = "*" if r["HAC_p_value"] < 0.05 else " "
    lines.append(f"T{int(r['topic']):<2} {r['model']:<22} {r['recession_coef_share_points']:+.4f} "
                 f"({r['effect_pct_of_mean']:+6.1f}% of mean) p={r['HAC_p_value']:.3f}{star} "
                 f"{'[stable]' if r['stable_across_seeds'] else '[unstable - do not interpret]'}")
lines.append(f"\n{n_tests} topics tested per model: about {0.05 * n_tests:.1f} would reach p < .05 "
             "by chance alone.")
lines.append("")
lines.append("-" * 78)
lines.append("HOW TO READ THIS")
lines.append("-" * 78)
lines.append("- Cosine similarity compares a topic's word distribution under two random")
lines.append("  seeds: 1 = identical, 0 = no words in common. Topics below the cut-off")
lines.append("  depend on the seed and should not be interpreted.")
lines.append("- Dominant-topic agreement shows how often an individual ad is assigned")
lines.append("  to the same topic under both seeds (after pairing the topics).")
lines.append("- Lower held-out perplexity means a better statistical fit, but more")
lines.append("  topics usually fit better while being less stable and harder to read.")
lines.append("- Recession differences are computed on 268 quarterly averages from a")
lines.append("  20,000-ad sample, so each quarter rests on relatively few ads.")
lines.append("- With many topics tested, a few p < .05 results are expected by chance;")
lines.append("  only stable topics with clear, sizeable effects deserve attention.")
lines.append("- Topic labels come from reading the top words and are interpretations.")
lines.append("- All results are descriptive associations, not causal effects.")

summary_path = TABLE_OUTPUT_DIR / "topic_model_diagnostic_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")

for path in list(paths.values()) + [stability_fig_path, prevalence_fig_path, summary_path]:
    print(f"  {path.relative_to(REPO_ROOT)}")
