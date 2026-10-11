"""
PURPOSE
    Draw about 150 records, stratified by decade (1948 to 2014), for hand
    labelling as advertisement vs non-advertisement. Kiran's first 200 labels
    all come from 2012 to 2014, so they cannot tell us how the filters work in
    earlier decades.

QUESTION
    How well do our ad / non-ad filters work across the whole period, and can a
    text classifier do better? (This script only draws the sample)

WHY IT MATTERS
    The Week 5 feedback asked us to tighten the ad vs non-ad filter so the null
    result is more convincing. That needs labels from every decade.

DESIGN
    Within each decade, records fall into three non-overlapping strata:
      cue   : text contains a phrase typical of non-ads (see CUE_PATTERNS)
      long  : not cue, and among the longest 15% of records in that decade
      other : everything else
    Each stratum is sampled separately (seed 314). Every sampled record gets a
    weight = stratum population / stratum sample size, so corpus-level precision
    and recall can be estimated without bias later.
    Kiran's 200 labelled records are excluded.

FILES LOADED
    week three data cleaning/Archived Data/week_three_ads_with_sentiment.csv (Git LFS)
        or the path in env var SOSC314_ADS_FILE (Daniel's Drive copy)
    week 5 - .../01 ad filter robustness/hand labelled sample/kiran_200_hand_labelled_ads.csv

OUTPUTS (in this script's folder)
    labelling sheets/labelling_sheet_main.xlsx           -> Daniel labels all rows
    labelling sheets/labelling_sheet_second_coder.xlsx   -> Daniel labels 30 rows
    csv outputs/labelling_sample_key.csv                 -> stratum, weight, decade (NOT for labellers)
    csv outputs/labelling_sample_text.csv                -> same rows as the main sheet, plain CSV
    robustness tables/labelling_sample_summary.txt       -> counts and built-in checks

JOIN RULE
    row_id_0based = 0-based row position in week_three_ads_with_sentiment.csv.
    That file is stored newest first.

RUN
    python "week 7 - final adjustments/01 ad filter tightening/01_draw_labelling_sample.py"
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings
SEED = 314
EXPECTED_ROWS = int(os.environ.get("SOSC314_EXPECTED_ROWS", 172219))
TEXT_PREVIEW_CHARS = 5000          # Excel and Sheets cells cap near 32k / 50k chars
LONG_QUANTILE = 0.85               # top 15% by word count within decade = "long"
N_SECOND_CODER = 30

# Rows per decade and stratum: (cue, long, other). 1940s = 1948-49 only.
ALLOCATION = {
    1940: (3, 2, 5),
    1950: (5, 5, 10),
    1960: (5, 5, 10),
    1970: (5, 5, 10),
    1980: (5, 5, 10),
    1990: (5, 5, 10),
    2000: (5, 5, 10),
    2010: (5, 5, 10),
}

# Phrases typical of non-ads. Case-insensitive. Used only to oversample.
CUE_PATTERNS = [
    r"\bthe economist\b",
    r"subscri",                                  # subscribe, subscription
    r"offer to readers",
    r"letters are welcome",
    r"intelligence unit",
    r"matter of record",
    r"notice is hereby given",
    r"annual general meeting",
    r"balance sheet",
    r"prospectus",
    r"\bbranches\b",
    r"interim (?:results|report|dividend)",
    r"profit (?:before|after) tax",
]

TEXT_COL_CANDIDATES = ["OCR_GoogleVision_original"]
DATE_COL_CANDIDATES = ["Date", "date", "issue_date", "Issue_Date", "DATE", "pub_date", "publication_date"]
BRAND_COL_CANDIDATES = ["Brand", "brand"]

CATEGORY_CODES = ["A1", "A2", "A3", "N1", "N2", "N3", "N4", "N5"]

# ----------------------------------------------------------------------------
# Paths
HERE = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "week three data cleaning").is_dir():
            return p
    sys.exit("Could not find the repo root (a folder containing 'week three data cleaning'). "
             "Run this script from inside the cloned repo.")


ROOT = find_repo_root(HERE)
ADS_FILE = Path(os.environ.get(
    "SOSC314_ADS_FILE",
    ROOT / "week three data cleaning" / "Archived Data" / "week_three_ads_with_sentiment.csv"))
WEEK5 = ROOT / "week 5 - diagnostics robustness sensitivity validity regression checks and model revisions"
KIRAN_FILE = Path(os.environ.get(
    "SOSC314_KIRAN_FILE",
    WEEK5 / "01 ad filter robustness" / "hand labelled sample" / "kiran_200_hand_labelled_ads.csv"))

OUT_SHEETS = HERE / "labelling sheets"
OUT_CSV = HERE / "csv outputs"
OUT_TABLES = HERE / "robustness tables"
for d in (OUT_SHEETS, OUT_CSV, OUT_TABLES):
    d.mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def pick(columns, candidates, contains=None, required=True, what="column"):
    for c in candidates:
        if c in columns:
            return c
    if contains:
        hits = [c for c in columns if contains in c.lower()]
        if hits:
            return hits[0]
    if required:
        sys.exit(f"Could not find the {what}. Columns in the file:\n{list(columns)}\n"
                 f"Add the right name to the candidates list at the top of the script.")
    return None


# ----------------------------------------------------------------------------
# 1. Load the corpus (only the columns we need), keep file order
if not ADS_FILE.exists():
    sys.exit(f"Ads file not found: {ADS_FILE}\n"
             "If it is a Git LFS pointer, run `git lfs pull`, or set SOSC314_ADS_FILE to your Drive copy.")

header = pd.read_csv(ADS_FILE, nrows=0).columns
TEXT_COL = pick(header, TEXT_COL_CANDIDATES, contains="ocr", what="OCR text column")
DATE_COL = pick(header, DATE_COL_CANDIDATES, contains="date", what="date column")
BRAND_COL = pick(header, BRAND_COL_CANDIDATES, required=False)
use = [c for c in (TEXT_COL, DATE_COL, BRAND_COL) if c]

ads = pd.read_csv(ADS_FILE, usecols=use, low_memory=False)
ads = ads[use]                              # keep a fixed column order
ads.insert(0, "row_id_0based", np.arange(len(ads)))   # positional index, never sorted

log("=" * 70)
log("Task a: second labelling sample")
log("=" * 70)
log(f"Ads file: {ADS_FILE}")
log(f"Columns used: text={TEXT_COL!r}, date={DATE_COL!r}, brand={BRAND_COL!r}")
log(f"Rows loaded: {len(ads):,}")

ads["_date"] = pd.to_datetime(ads[DATE_COL], errors="coerce")
ads["year"] = ads["_date"].dt.year
ads["decade"] = (ads["year"] // 10 * 10).astype("Int64")
ads["_text"] = ads[TEXT_COL].fillna("").astype(str)
ads["n_words"] = ads["_text"].str.split().str.len()

# ----------------------------------------------------------------------------
# 2. Built-in checks on the corpus
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{'  (' + detail + ')' if detail else ''}")


check("Row count matches the canonical corpus", len(ads) == EXPECTED_ROWS,
      f"{len(ads):,} vs expected {EXPECTED_ROWS:,}")
check("Every row has a parseable date", ads["_date"].notna().all(),
      f"{ads['_date'].isna().sum()} missing")
check("File is stored newest first (row 0 is the latest date)",
      ads["_date"].iloc[0] == ads["_date"].max(),
      f"row 0 = {ads['_date'].iloc[0].date() if pd.notna(ads['_date'].iloc[0]) else 'NA'}")
check("Years fall within 1948-2014",
      ads["year"].min() >= 1948 and ads["year"].max() <= 2014,
      f"{ads['year'].min()} to {ads['year'].max()}")

# ----------------------------------------------------------------------------
# 3. Exclude Kiran's 200 labelled records
excluded = set()
if KIRAN_FILE.exists():
    kiran = pd.read_csv(KIRAN_FILE)
    if "row_id" not in kiran.columns:
        sys.exit(f"Kiran's file has no 'row_id' column. Columns: {list(kiran.columns)}")
    excluded = set((kiran["row_id"].astype(int) - 1).tolist())   # her row_id starts at 1
    check("Kiran's file has 200 unique records", len(excluded) == 200, f"{len(excluded)} unique")
    check("All of Kiran's records fall in 2012-2014",
          ads.loc[ads["row_id_0based"].isin(excluded), "year"].between(2012, 2014).all())
else:
    log(f"WARNING: Kiran's file not found at {KIRAN_FILE}. Nothing excluded.")
    check("Kiran's file found", False, str(KIRAN_FILE))

pool = ads[~ads["row_id_0based"].isin(excluded)].copy()
log(f"Sampling pool after exclusion: {len(pool):,}")

# ----------------------------------------------------------------------------
# 4. Strata: cue / long / other, within decade
cue_regex = re.compile("|".join(CUE_PATTERNS), flags=re.IGNORECASE)
pool["is_cue"] = pool["_text"].str.contains(cue_regex)
long_cut = pool.groupby("decade")["n_words"].transform(lambda s: s.quantile(LONG_QUANTILE))
pool["is_long"] = pool["n_words"] >= long_cut
pool["sample_stratum"] = np.select(
    [pool["is_cue"], pool["is_long"]], ["cue", "long"], default="other")

# ----------------------------------------------------------------------------
# 5. Draw
rng = np.random.default_rng(SEED)
parts = []
log("")
log(f"{'decade':>6} {'stratum':>7} {'population':>11} {'drawn':>6} {'weight':>9}")
for decade, (n_cue, n_long, n_other) in ALLOCATION.items():
    for stratum, n_want in (("cue", n_cue), ("long", n_long), ("other", n_other)):
        cell = pool[(pool["decade"] == decade) & (pool["sample_stratum"] == stratum)]
        n = min(n_want, len(cell))
        if n < n_want:
            log(f"NOTE: {decade}s {stratum} has only {len(cell)} records; drawing {n}.")
        if n == 0:
            continue
        idx = rng.choice(cell.index.to_numpy(), size=n, replace=False)
        drawn = cell.loc[idx].copy()
        drawn["stratum_population"] = len(cell)
        drawn["weight"] = len(cell) / n
        parts.append(drawn)
        log(f"{decade:>6} {stratum:>7} {len(cell):>11,} {n:>6} {len(cell) / n:>9.1f}")

sample = pd.concat(parts)
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle order for labelling
sample.insert(0, "label_id", [f"L{i + 1:03d}" for i in range(len(sample))])

# Second coder: 30 rows, spread across the sample
second_ids = set(rng.choice(sample["label_id"].to_numpy(), size=min(N_SECOND_CODER, len(sample)), replace=False))
sample["double_label"] = sample["label_id"].isin(second_ids).astype(int)

# ----------------------------------------------------------------------------
# 6. Checks on the sample
log("")
target = sum(sum(v) for v in ALLOCATION.values())
check("Sample size equals the allocation", len(sample) == target, f"{len(sample)} of {target}")
check("No duplicate records in the sample", sample["row_id_0based"].is_unique)
check("No overlap with Kiran's 200", not sample["row_id_0based"].isin(excluded).any())
check("Every decade is represented",
      set(sample["decade"].dropna().astype(int)) == set(ALLOCATION))
check("Weights add up to the pool size",
      abs(sample["weight"].sum() - len(pool[pool["decade"].isin(list(ALLOCATION))])) < 1e-6,
      f"{sample['weight'].sum():,.0f} vs {len(pool):,}")
# Positional join check: re-read only the sampled rows' text straight from the file
recheck = pd.read_csv(ADS_FILE, usecols=[TEXT_COL], low_memory=False)[TEXT_COL].fillna("").astype(str)
same = (recheck.iloc[sample["row_id_0based"]].to_numpy() == sample["_text"].to_numpy()).all()
check("Text at row_id_0based matches a fresh read of the file", same)
check("Second-coder subset has the right size", sample["double_label"].sum() == N_SECOND_CODER)

log("")
log("Sample by decade and stratum:")
log(sample.pivot_table(index="decade", columns="sample_stratum", values="label_id",
                       aggfunc="count", fill_value=0).to_string())
log(f"\nTexts longer than {TEXT_PREVIEW_CHARS:,} characters (truncated in the sheet): "
    f"{(sample['_text'].str.len() > TEXT_PREVIEW_CHARS).sum()}")

# ----------------------------------------------------------------------------
# 7. Write outputs
sample["issue_date"] = sample["_date"].dt.strftime("%Y-%m-%d")
sample["brand"] = sample[BRAND_COL] if BRAND_COL else ""
sample["chars_total"] = sample["_text"].str.len()
sample["text_truncated"] = (sample["chars_total"] > TEXT_PREVIEW_CHARS).astype(int)
sample["ocr_text"] = sample["_text"].str.slice(0, TEXT_PREVIEW_CHARS)

key_cols = ["label_id", "row_id_0based", "issue_date", "year", "decade", "sample_stratum",
            "stratum_population", "weight", "n_words", "chars_total", "double_label"]
sample[key_cols].to_csv(OUT_CSV / "labelling_sample_key.csv", index=False)

sheet_cols = ["label_id", "row_id_0based", "issue_date", "brand", "n_words",
              "text_truncated", "ocr_text"]
fill_cols = ["category", "is_ad", "confidence", "notes"]
main = sample[sheet_cols].copy()
for c in fill_cols:
    main[c] = ""
main.to_csv(OUT_CSV / "labelling_sample_text.csv", index=False, encoding="utf-8-sig")

second = sample.loc[sample["double_label"] == 1, sheet_cols].copy()
second = second.sample(frac=1, random_state=SEED + 1)          # different order from the main sheet
for c in fill_cols:
    second[c] = ""


def write_sheet(df, path, coder):
    """Excel sheet with dropdowns, wrapped text and an auto-filled is_ad column."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="labels", index=False)
        pd.DataFrame({
            "code": CATEGORY_CODES,
            "is_ad": [1, 1, 1, 0, 0, 0, 0, 0],
            "meaning": ["Persuasive ad", "Recruitment ad", "Other outside promotion",
                        "Editorial", "Economist house content", "Financial or legal notice",
                        "Mixed or multi-item", "Unreadable OCR"],
        }).to_excel(xw, sheet_name="codes", index=False)
        ws = xw.sheets["labels"]
        cols = {name: i + 1 for i, name in enumerate(df.columns)}
        letter = lambda name: ws.cell(row=1, column=cols[name]).column_letter
        n = len(df) + 1

        widths = {"label_id": 9, "row_id_0based": 13, "issue_date": 12, "brand": 18, "n_words": 9,
                  "text_truncated": 9, "ocr_text": 90, "category": 11, "is_ad": 7,
                  "confidence": 11, "notes": 35}
        for name, w in widths.items():
            ws.column_dimensions[letter(name)].width = w
        for row in ws.iter_rows(min_row=2, max_row=n):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in ws[1]:
            cell.font = Font(bold=True)
        to_fill = PatternFill("solid", fgColor="FFF4D6")
        for name in ("category", "confidence", "notes"):
            for r in range(1, n + 1):
                ws[f"{letter(name)}{r}"].fill = to_fill

        dv_cat = DataValidation(type="list", formula1='"' + ",".join(CATEGORY_CODES) + '"', allow_blank=True)
        dv_conf = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
        ws.add_data_validation(dv_cat)
        ws.add_data_validation(dv_conf)
        dv_cat.add(f"{letter('category')}2:{letter('category')}{n}")
        dv_conf.add(f"{letter('confidence')}2:{letter('confidence')}{n}")
        cat = letter("category")
        for r in range(2, n + 1):
            ws[f"{letter('is_ad')}{r}"] = f'=IF({cat}{r}="","",IF(LEFT({cat}{r},1)="A",1,0))'
        ws.freeze_panes = "B2"
        ws.sheet_properties.tabColor = "C2410C"
        ws.title = f"labels_{coder}"


write_sheet(main, OUT_SHEETS / "labelling_sheet_main.xlsx", "main")
write_sheet(second, OUT_SHEETS / "labelling_sheet_second_coder.xlsx", "second")

log("")
log("Outputs:")
for p in [OUT_SHEETS / "labelling_sheet_main.xlsx", OUT_SHEETS / "labelling_sheet_second_coder.xlsx",
          OUT_CSV / "labelling_sample_key.csv", OUT_CSV / "labelling_sample_text.csv"]:
    log(f"  {p.relative_to(ROOT)}")

n_fail = sum(not ok for _, ok, _ in checks)
log("")
log(f"Built-in checks: {len(checks) - n_fail} passed, {n_fail} failed")
(OUT_TABLES / "labelling_sample_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
if n_fail:
    sys.exit("Some checks failed. Read the FAIL lines above before using the sample.")
