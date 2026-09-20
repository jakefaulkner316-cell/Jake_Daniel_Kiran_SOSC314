#  WEEK 4 CORPUS QUALITY FLAGS

# Our corpus is supposed to have ads onlly, but some of them are not.
# Some  are letters, double pages, and other formats that disrupt our analysis.
# Some are also way too short, in a manner that we cannot grade their sentiment.
#
#Here, we are JUST LABELING, NOT DELETING. In this way, we can preserve the 
# version control and run each analysis separately, verifying the impacts of filterning or not.
#
# Flagged records that are not single advertisements:
#   1. Brands that are not advertisers (letters page, house promos)
#   2. Multi-page supplements merged into one row
#   3. Records too short for a stable sentiment ratio
#
# Again: WE FLAG, WE DO NOT DELETE. Output is one row per advertisement with
# boolean columns. Filtering happens later, at analysis time, so every
# filter stays available as a sensitivity check.

import os
import subprocess
import sys

import pandas as pd


# Brands that are not advertisers. Each needs a reason: we have to
# defend these entries individually in the report and the oral check.
NON_ADVERTISERS = {
    "Letters are welcome":  "letters-to-the-editor page",
    "Subscription Service": "Economist house promotion",
    "Offer to readers":     "editorial text captured as an advertisement",
}

# A multi-page record repeats the page header. One advertisement cannot contain it twice
HEADER_PATTERN = r"THE\s+ECONOMIST\s+[A-Z]+\s+\d{1,2},?\s+\d{4}"
ADVERT_LABEL_PATTERN = r"\bAdvertisement\b"

# Placeholder (prb just keep 10 tokens indeed)
MIN_TOKENS = 10

# clone the repo and fetch one file for setup


REPO_URL = "https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git"
REPO_DIR = "/content/Jake_Daniel_Kiran_SOSC314"
DATA_REL = "week three data cleaning/Archived Data/week_three_ads_with_sentiment.csv"
DATA_ABS = os.path.join(REPO_DIR, DATA_REL)
OUT_DIR = "/content/week_four_outputs"


def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print("STDERR:", r.stderr.strip())
    return r.returncode


def is_pointer(path):
    """An unresolved Git LFS file is a 3-line text stub, not real data."""
    try:
        with open(path) as f:
            return f.readline().startswith("version https://git-lfs")
    except (FileNotFoundError, UnicodeDecodeError):
        return True


run("apt-get -qq install -y git-lfs")
run("git lfs install --skip-repo")

# GIT_LFS_SKIP_SMUDGE=1 clones pointers only. Without it, the clone
# drags down about 2.9 GB of files we do not need.
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Cloning repository...")
    run(f"GIT_LFS_SKIP_SMUDGE=1 git clone {REPO_URL} {REPO_DIR}")
else:
    print("Repository already present. Pulling latest changes.")
    run("git pull", cwd=REPO_DIR)

if is_pointer(DATA_ABS):
    print("\nDownloading the data file (about 255 MB). A few minutes.")
    run(f'git lfs pull --include="{DATA_REL}"', cwd=REPO_DIR)
else:
    print("\nData file already downloaded.")

if is_pointer(DATA_ABS):
    sys.exit("Download failed. The file is still a Git LFS pointer.")

os.makedirs(OUT_DIR, exist_ok=True)
print("Setup complete.\n")


# STEP 1 -  LOAD

ads = pd.read_csv(
    DATA_ABS,
    usecols=["DateOfIssue", "Brand", "OCR_GoogleVision_original", "Total_Words"],
    dtype={"Brand": "string", "OCR_GoogleVision_original": "string"},
)

print("Rows loaded:", len(ads))

# row_id is the join key for the whole group. This file is the cleaned
# advertisements with sentiment columns attached, written with
# index=False, so its row order is canonical. Never sort these rows.
ads = ads.reset_index(drop=True)
ads["row_id"] = ads.index

ads["Date"] = pd.to_datetime(ads["DateOfIssue"], errors="coerce")
ads["Decade"] = (ads["Date"].dt.year // 10) * 10

text = ads["OCR_GoogleVision_original"].fillna("")

# Total_Words comes from week_three_sentiment.py, so too_short refers to
# exactly the denominator used in the sentiment ratios. Recomputing it
# here would risk a silent mismatch.
ads["token_count"] = ads["Total_Words"]

# STEP 2: WORD COUNT DISTRIBUTION - use this to set MIN_TOKENS 

print("\nWORD COUNT DISTRIBUTION")
print(ads["token_count"].describe())

print("\nPercentiles:")
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
    print(f"  p{q*100:>5.1f}: {ads['token_count'].quantile(q):>8.0f}")

print("\nHow many ads fall below each candidate minimum:")
for cut in [1, 5, 10, 15, 20, 30]:
    n = int((ads["token_count"] < cut).sum())
    print(f"  < {cut:3d} words: {n:6d} ads ({100*n/len(ads):5.2f}%)")

# STEP 3: THE THREE FLAGS

ads["is_non_advertiser"] = ads["Brand"].isin(NON_ADVERTISERS.keys())

ads["header_count"] = text.str.count(HEADER_PATTERN)
ads["advert_label_count"] = text.str.count(ADVERT_LABEL_PATTERN)
ads["is_multipage"] = ads["header_count"] >= 2

ads["too_short"] = ads["token_count"] < MIN_TOKENS

print("\nFLAG is_non_advertiser:", int(ads["is_non_advertiser"].sum()), "ads")
for brand, reason in NON_ADVERTISERS.items():
    n = int((ads["Brand"] == brand).sum())
    print(f"  {brand!r}: {n} ads  ({reason})")

print("\nFLAG is_multipage:", int(ads["is_multipage"].sum()), "ads")
print("Page-header counts found:")
print(ads["header_count"].value_counts().sort_index().head(8))

print("\nLongest multi-page records:")
cols = ["Brand", "DateOfIssue", "token_count", "header_count", "advert_label_count"]
print(ads[ads["is_multipage"]].nlargest(10, "token_count")[cols].to_string(index=False))

print(f"\nFLAG too_short (< {MIN_TOKENS} words):", int(ads["too_short"].sum()), "ads")

# STEP 4: CANDIDATES FOR THE BLOCKLIST
#
# Brands appearing often among very long records are likely supplements
# or editorial content. This flags nothing: it is a shortlist for you to
# review by hand and add to NON_ADVERTISERS above.

long_records = ads[ads["token_count"] > 1000]

candidates = (
    long_records["Brand"].value_counts().head(40)
    .rename_axis("Brand").reset_index(name="Long_Records")
)
candidates["Already_Blocked"] = candidates["Brand"].isin(NON_ADVERTISERS.keys())
candidates["Total_Records"] = candidates["Brand"].map(ads["Brand"].value_counts())

print("\nBLOCKLIST CANDIDATES (brands common among 1000+ word records)")
print(candidates.head(25).to_string(index=False))

candidates.to_csv(f"{OUT_DIR}/blocklist_candidates.csv", index=False)

# STEP 5: IS THE FILTERING BIASED ACROSS TIME?

#
# This decides whether a flag can be applied quietly. Record length
# correlates with decade in our corpus, peaking in the 1970s and 80s,
# and sentiment varies strongly by decade. A flag that removes far more
# from one era than another interacts with our main result.

FLAGS = ["is_non_advertiser", "is_multipage", "too_short"]
ads["any_flag"] = ads[FLAGS].any(axis=1)

by_decade = (
    ads.groupby("Decade")
    .agg(
        Ads=("row_id", "size"),
        Pct_non_advertiser=("is_non_advertiser", lambda s: 100 * s.mean()),
        Pct_multipage=("is_multipage", lambda s: 100 * s.mean()),
        Pct_too_short=("too_short", lambda s: 100 * s.mean()),
        Pct_any_flag=("any_flag", lambda s: 100 * s.mean()),
    )
    .round(3)
    .reset_index()
)

print("\nREMOVAL RATE BY DECADE (percent of that decade's ads)")
print(by_decade.to_string(index=False))

by_decade.to_csv(f"{OUT_DIR}/flag_removal_by_decade.csv", index=False)

spread = by_decade["Pct_any_flag"].max() - by_decade["Pct_any_flag"].min()
print(f"\nSpread between highest and lowest decade: {spread:.2f} percentage points")
if spread > 3:
    print("=> Uneven across decades. Report as a sensitivity check,")
    print("   do not apply silently.")
else:
    print("=> Reasonably even across decades.")

# STEP 6: SAVE
#
# DateOfIssue and Brand travel with the flags so anyone merging on
# row_id can verify the join lined up. Kiran's OCR and notice flags get
# added to this same file later, on the same key and row order. Follow up on this to see how it goes. ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

OUT_COLS = [
    "row_id", "DateOfIssue", "Brand", "token_count",
    "header_count", "advert_label_count",
    "is_non_advertiser", "is_multipage", "too_short", "any_flag",
]

ads[OUT_COLS].to_csv(f"{OUT_DIR}/week_four_quality_flags.csv", index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total advertisements:        {len(ads):>7,}")
for f in FLAGS:
    n = int(ads[f].sum())
    print(f"  {f:<22} {n:>7,}  ({100*n/len(ads):5.2f}%)")
n_any = int(ads["any_flag"].sum())
print(f"  {'flagged by any rule':<22} {n_any:>7,}  ({100*n_any/len(ads):5.2f}%)")
print(f"Remaining if all applied:    {len(ads)-n_any:>7,}")

# STEP 7: DOWNLOAD

import shutil

shutil.make_archive("/content/week_four_outputs", "zip", OUT_DIR)
print("\nFiles ready:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)

try:
    from google.colab import files
    files.download("/content/week_four_outputs.zip")
except Exception as e:
    print("\nAutomatic download blocked. Use the folder icon in the left")
    print("sidebar and download /content/week_four_outputs.zip by hand.")
