# Fightin' Words: Recession vs. Expansion Advertising Vocabulary
# SOSC 314
#
# Purpose: Identify vocabulary associated with recessions and
# expansions, accounting for historical language drift.
#
# Inputs: Economist master dataset, quality flags, NBER recession dates.
# Outputs: Word scores, permutation thresholds, and figures.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent

MASTER = ROOT / "Initial file imports/The Economist Historical Advertisements – Master Dataset.csv"
FLAGS = ROOT / "week 4 - updated quality filtered analysis/Ad_filtering/week_four_quality_flags.csv"
RECESSIONS = ROOT / "week three data cleaning/nber_recession_table.csv"

# Load the original advertisements.
ads = pd.read_csv(MASTER, low_memory=False)

ads = ads[["DateOfIssue", "Brand", "Brand is generic (e.g. 'Notices')", "OCR_GoogleVision_original"]].copy()

ads["DateOfIssue"] = pd.to_datetime(ads["DateOfIssue"], errors="coerce")
ads["Year"] = ads["DateOfIssue"].dt.year
ads["Quarter"] = ads["DateOfIssue"].dt.quarter
ads["Year_Quarter"] = ads["DateOfIssue"].dt.to_period("Q").astype(str)

# Recreate the original corpus without sorting.
ads = ads[(ads["Year"] >= 1948) & (ads["Brand is generic (e.g. 'Notices')"] == False)].copy()
ads = ads.reset_index(drop=True)
ads["row_id_0based"] = ads.index

# Load quality flags.
flags = pd.read_csv(FLAGS, low_memory=False)
flags = flags.rename(columns={"row_id": "row_id_0based"})

ads = ads.merge(flags[["row_id_0based", "any_flag"]], on="row_id_0based", how="left", validate="one_to_one")

# Load recession quarters.
recessions = pd.read_csv(RECESSIONS, low_memory=False)
ads = ads.merge(recessions[["Year_Quarter", "Recession_Majority"]], on="Year_Quarter", how="left", validate="many_to_one")

# Keep the original analysis period and valid texts.
ads = ads[(ads["Year"] <= 2014) & ads["OCR_GoogleVision_original"].notna()].copy()

print("Original records with text:", len(ads))
print("Flagged records:", ads["any_flag"].sum())
print("Missing quality flags:", ads["any_flag"].isna().sum())
print("Missing recession classifications:", ads["Recession_Majority"].isna().sum())

# Starting comparison: remove records flagged by the available v1 filter.
clean_ads = ads[ads["any_flag"] == False].copy()

print("Advertisements available for Fightin' Words:", len(clean_ads))
print("Recession advertisements:", clean_ads["Recession_Majority"].sum())
print("Expansion advertisements:", (clean_ads["Recession_Majority"] == False).sum())

# ---------------------------------------------------------
# FIGHTIN' WORDS: CACHED DECADE WORD COUNTS
# ---------------------------------------------------------

from collections import Counter
import pickle

clean_ads["Decade"] = (clean_ads["Year"] // 10) * 10

csv_dir = OUTPUT / "csv outputs"
csv_dir.mkdir(parents=True, exist_ok=True)

cache_file = OUTPUT / "word_counts_v1.pkl"

def tokenize(text):
    return re.findall(r"[a-z]{3,}", str(text).lower())

if cache_file.exists():
    print("\nLoading cached word counts...")
    with open(cache_file, "rb") as f:
        word_counts, total_words = pickle.load(f)
else:
    print("\nCalculating word counts...")
    word_counts = {}
    total_words = {}

    for (decade, recession), group in clean_ads.groupby(["Decade", "Recession_Majority"]):
        counts = Counter()

        for text in group["OCR_GoogleVision_original"]:
            counts.update(tokenize(text))

        word_counts[(decade, recession)] = counts
        total_words[(decade, recession)] = sum(counts.values())

    with open(cache_file, "wb") as f:
        pickle.dump((word_counts, total_words), f)

    print("Word counts cached.")
# ---------------------------------------------------------
# FIGHTIN' WORDS: WEIGHTED LOG-ODDS WITH DIRICHLET PRIOR
# ---------------------------------------------------------

from scipy.stats import norm

# Build a shared vocabulary and informative prior.
overall_counts = Counter()
for counts in word_counts.values():
    overall_counts.update(counts)

# Restrict testing to words occurring at least 50 times.
vocabulary = [word for word, count in overall_counts.items() if count >= 50]

# Allocate prior mass proportional to overall vocabulary frequencies.
prior_strength = 1000
total_corpus_words = sum(overall_counts.values())
prior = {word: prior_strength * overall_counts[word] / total_corpus_words for word in vocabulary}

results = []

for word in vocabulary:
    decade_effects = []
    decade_variances = []

    for decade in sorted(clean_ads["Decade"].unique()):
        rec_counts = word_counts.get((decade, True))
        exp_counts = word_counts.get((decade, False))

        # Only compare decades containing both recession and expansion ads.
        if rec_counts is None or exp_counts is None:
            continue

        y_rec = rec_counts[word]
        y_exp = exp_counts[word]
        n_rec = total_words[(decade, True)]
        n_exp = total_words[(decade, False)]
        alpha = prior[word]

        # Smoothed log-odds within each decade.
        odds_rec = np.log((y_rec + alpha) / (n_rec + prior_strength - y_rec - alpha))
        odds_exp = np.log((y_exp + alpha) / (n_exp + prior_strength - y_exp - alpha))

        delta = odds_rec - odds_exp
        variance = 1 / (y_rec + alpha) + 1 / (y_exp + alpha)

        decade_effects.append(delta)
        decade_variances.append(variance)

    if not decade_effects:
        continue

    # Inverse-variance weighted combination across decades.
    weights = 1 / np.array(decade_variances)
    combined_delta = np.sum(weights * decade_effects) / np.sum(weights)
    combined_se = np.sqrt(1 / np.sum(weights))
    z_score = combined_delta / combined_se

    results.append({
        "word": word,
        "total_count": overall_counts[word],
        "log_odds": combined_delta,
        "z_score": z_score,
        "p_value_naive": 2 * norm.sf(abs(z_score)),
        "decades_compared": len(decade_effects)
    })

scores = pd.DataFrame(results).sort_values("z_score", ascending=False)

scores.to_csv(csv_dir / "fighting_words_scores_v1.csv", index=False)

print("\nTOP 20 RECESSION-ASSOCIATED WORDS:")
print(scores.head(20)[["word", "total_count", "z_score"]].to_string(index=False))

print("\nTOP 20 EXPANSION-ASSOCIATED WORDS:")
print(scores.tail(20)[["word", "total_count", "z_score"]].to_string(index=False))

print("\nSaved Fightin' Words scores.")

# ---------------------------------------------------------
# FIGHTIN' WORDS: CACHE QUARTER-LEVEL WORD COUNTS
# ---------------------------------------------------------

quarter_cache = OUTPUT / "quarter_word_counts_v1.pkl"

if quarter_cache.exists():
    print("\nLoading cached quarterly word counts...")
    with open(quarter_cache, "rb") as f:
        quarter_counts = pickle.load(f)

else:
    print("\nCalculating quarterly word counts...")
    quarter_counts = {}

    for quarter, group in clean_ads.groupby("Year_Quarter"):
        counts = Counter()

        for text in group["OCR_GoogleVision_original"]:
            counts.update(tokenize(text))

        quarter_counts[quarter] = counts

    with open(quarter_cache, "wb") as f:
        pickle.dump(quarter_counts, f)

    print("Quarterly word counts cached.")

print("Total quarters:", len(quarter_counts))
print("Total quarterly word occurrences:", sum(sum(c.values()) for c in quarter_counts.values()))

# ---------------------------------------------------------
# FIGHTIN' WORDS: PREPARE QUARTER-LEVEL PERMUTATIONS
# ---------------------------------------------------------


# Build quarter metadata from the existing dataset.
quarter_info = clean_ads.groupby("Year_Quarter", sort=False).agg(
    Year=("Year", "first"),
    Recession=("Recession_Majority", "first")
).reset_index()

quarter_info["Decade"] = (quarter_info["Year"] // 10) * 10

# Use the vocabulary from our existing Fightin' Words analysis.
vocabulary = sorted(vocabulary)
word_index = {word: i for i, word in enumerate(vocabulary)}

# Convert cached quarterly counts into a compact matrix.
quarter_matrix = np.zeros((len(quarter_info), len(vocabulary)), dtype=np.int32)

for i, quarter in enumerate(quarter_info["Year_Quarter"]):
    for word, count in quarter_counts[quarter].items():
        if word in word_index:
            quarter_matrix[i, word_index[word]] = count

# Retain total word counts, including words outside the tested vocabulary.
quarter_totals = np.array([
    sum(quarter_counts[q].values()) for q in quarter_info["Year_Quarter"]
], dtype=np.int64)

print("\nQuarter matrix created.")
print("Shape:", quarter_matrix.shape)
print("Total words:", quarter_totals.sum())

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

keep_indices = [i for i, word in enumerate(vocabulary) if word not in stop_words]

vocabulary = [vocabulary[i] for i in keep_indices]
quarter_matrix = quarter_matrix[:, keep_indices]
prior_array = np.array([prior[w] for w in vocabulary])

print("\nVocabulary after removing stopwords:", len(vocabulary))

# Save quarter metadata for reproducibility.
quarter_info.to_csv(csv_dir / "fighting_words_quarter_metadata.csv", index=False)

# ---------------------------------------------------------
# FIGHTIN' WORDS: QUARTER-LEVEL PERMUTATION TEST
# ---------------------------------------------------------

# Use the same weighted log-odds statistic for actual and shuffled data.
decades = quarter_info["Decade"].to_numpy()
actual_labels = quarter_info["Recession"].astype(bool).to_numpy()
unique_decades = np.unique(decades)

prior_array = np.array([prior[w] for w in vocabulary])
prior_strength = prior_array.sum()

def calculate_z_scores(labels):
    effects = []
    precisions = []

    for decade in unique_decades:
        rec = (decades == decade) & labels
        exp = (decades == decade) & ~labels

        if not rec.any() or not exp.any():
            continue

        rec_counts = quarter_matrix[rec].sum(axis=0).astype(float)
        exp_counts = quarter_matrix[exp].sum(axis=0).astype(float)

        rec_total = quarter_totals[rec].sum()
        exp_total = quarter_totals[exp].sum()

        rec_num = rec_counts + prior_array
        exp_num = exp_counts + prior_array

        rec_den = rec_total + prior_strength - rec_num
        exp_den = exp_total + prior_strength - exp_num

        delta = np.log(rec_num / rec_den) - np.log(exp_num / exp_den)
        variance = 1 / rec_num + 1 / exp_num

        precision = 1 / variance
        effects.append(delta * precision)
        precisions.append(precision)

    total_precision = np.sum(precisions, axis=0)
    combined_delta = np.sum(effects, axis=0) / total_precision

    return combined_delta * np.sqrt(total_precision)


# Calculate observed statistics.
observed_z = calculate_z_scores(actual_labels)

# Shuffle recession labels across quarters within each decade.
rng = np.random.default_rng(42)
n_permutations = 500
max_z_null = np.empty(n_permutations)
max_words_null = []

print("\nRunning quarter-level permutations...")

for b in range(n_permutations):
    shuffled = actual_labels.copy()

    for decade in unique_decades:
        idx = np.where(decades == decade)[0]
        shuffled[idx] = rng.permutation(shuffled[idx])

    shuffled_z = calculate_z_scores(shuffled)
    max_idx = np.argmax(np.abs(shuffled_z))
    max_z_null[b] = abs(shuffled_z[max_idx])

    max_words_null.append({
        "permutation": b + 1,
        "word": vocabulary[max_idx],
        "z_score": shuffled_z[max_idx],
        "max_abs_z": max_z_null[b]
    })

    if (b + 1) % 50 == 0:
        print(f"Completed {b + 1}/{n_permutations} permutations")

permutation_diagnostics = pd.DataFrame(max_words_null)
permutation_diagnostics.to_csv(csv_dir / "fighting_words_permutation_max_words_v1.csv", index=False)

print("\nMOST EXTREME PERMUTATION RESULTS:")
print(permutation_diagnostics.nlargest(15, "max_abs_z").to_string(index=False))
# Family-wise 5% significance threshold.
threshold = np.quantile(max_z_null, 0.95)

permutation_results = pd.DataFrame({
    "word": vocabulary,
    "z_score": observed_z,
    "abs_z": np.abs(observed_z),
    "significant_fwer_05": np.abs(observed_z) > threshold
}).sort_values("abs_z", ascending=False)

permutation_results.to_csv(
    csv_dir / "fighting_words_permutation_results_v1.csv",
    index=False
)

threshold_results = pd.DataFrame({
    "permutations": [n_permutations],
    "threshold_95": [threshold],
    "significant_words": [permutation_results["significant_fwer_05"].sum()]
})

threshold_results.to_csv(
    csv_dir / "fighting_words_permutation_threshold_v1.csv",
    index=False
)

print("\nPermutation threshold:", round(threshold, 3))
print("Words exceeding threshold:", permutation_results["significant_fwer_05"].sum())

print("\nTOP WORDS AFTER PERMUTATION CORRECTION:")
print(permutation_results.head(20).to_string(index=False))

# ---------------------------------------------------------
# FIGHTIN' WORDS: VISUALIZE PERMUTATION RESULTS
# ---------------------------------------------------------

image_dir = OUTPUT / "diagnostic images"
image_dir.mkdir(parents=True, exist_ok=True)

top_rec = permutation_results.nlargest(10, "z_score")
top_exp = permutation_results.nsmallest(10, "z_score")
plot_data = pd.concat([top_exp, top_rec]).sort_values("z_score")

fig, ax = plt.subplots(figsize=(11, 8))

colors = ["#D55E00" if z > 0 else "#0072B2" for z in plot_data["z_score"]]

ax.barh(plot_data["word"], plot_data["z_score"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(threshold, color="gray", linestyle="--", label="Permutation threshold")
ax.axvline(-threshold, color="gray", linestyle="--")

ax.set_xlabel("Decade-adjusted weighted log-odds z-score")
ax.set_title("Advertising Vocabulary During Recessions vs. Expansions")
ax.legend()

plt.tight_layout()
plt.savefig(image_dir / "fighting_words_top_words_v1.png", dpi=300)
plt.close()

# Diagnostic: distribution of permutation maxima.
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(max_z_null, bins=30, edgecolor="white")
ax.axvline(threshold, color="red", linestyle="--", label=f"95% threshold = {threshold:.2f}")
ax.axvline(np.max(np.abs(observed_z)), color="black", linestyle="--", label="Observed maximum")

ax.set_xlabel("Maximum absolute z-score")
ax.set_ylabel("Number of permutations")
ax.set_title("Quarter-Level Permutation Null Distribution")
ax.legend()

plt.tight_layout()
plt.savefig(image_dir / "fighting_words_permutation_diagnostic_v1.png", dpi=300)
plt.close()

print("\nFigures saved successfully.")

# ---------------------------------------------------------
# DIAGNOSTIC: IDENTIFY QUARTERS WITH CONCENTRATED VOCABULARY
# ---------------------------------------------------------

quarter_diagnostics = quarter_info.copy()

quarter_diagnostics["total_words"] = quarter_totals
quarter_diagnostics["largest_word_count"] = quarter_matrix.max(axis=1)
quarter_diagnostics["largest_word_share"] = quarter_diagnostics["largest_word_count"] / quarter_diagnostics["total_words"]

print("\nQUARTERS WITH THE MOST CONCENTRATED VOCABULARY:")

print(
    quarter_diagnostics.nlargest(15, "largest_word_share")[
        ["Year_Quarter", "Decade", "Recession", "total_words", "largest_word_count", "largest_word_share"]
    ].to_string(index=False)
)

quarter_diagnostics.to_csv(
    csv_dir / "fighting_words_quarter_diagnostics_v1.csv",
    index=False
)

quarter_diagnostics["top_word"] = [
    max(quarter_counts[q], key=quarter_counts[q].get)
    for q in quarter_info["Year_Quarter"]
]

print("\nTOP WORDS IN HIGH-CONCENTRATION QUARTERS:")

print(
    quarter_diagnostics.nlargest(15, "largest_word_share")[
        ["Year_Quarter", "Recession", "top_word", "largest_word_share"]
    ].to_string(index=False)
)