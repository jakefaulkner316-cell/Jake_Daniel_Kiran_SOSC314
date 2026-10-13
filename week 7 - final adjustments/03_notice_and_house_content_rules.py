"""
03_notice_and_house_content_rules.py
SOSC 314 final phase, task c.

PURPOSE
    Add filter version v3: v2 plus text rules that catch the non-advertising
    records the brand blocklist misses:
      tombstone            "this announcement appears as a matter of record only", securities notices
      legal_notice         "notice is hereby given", general meetings, court-ordered publications
      financial_statement  balance sheets, interim results, profit statements
      economist_house      The Economist promoting itself (Intelligence Unit, conferences,
                           subscriptions, sister magazines, Economist.com listings)
      editorial            section contents, economic data tables
    Bank "cards" (name + branches, no selling message) are NOT flagged: the group has not
    settled guide rule 3 yet.

QUESTION
    How much better does v3 catch non-ads than v1 and v2, and does it remove real ads?

WHY IT MATTERS
    The Week 5 feedback: a tighter ad vs non-ad filter makes the null result more
    convincing, because we can be surer we are measuring advertising language.

HOW THE RULES ARE JUDGED (honest split)
    Development set: the 150 decade-stratified records (final_labels_150.csv). The rules
        were written while reading these, so their score here is optimistic.
    Held-out test set: Kiran's 200 labels (2012-2014). Never used to write the rules.
    Corpus: hits by rule and decade, plus a random sample of flagged records per rule
        to read by hand (precision audit).
    Built-in check: v1 and v2 must reproduce 01b on Kiran's labels
        (recall 2/37, kappa 0.085; recall 13/37, kappa 0.47).

FILES LOADED
    week 7 - final adjustments/model runner/csv outputs/filter_versions.csv   (from task b)
    week 7 - final adjustments/labelling sample/csv outputs/final_labels_150.csv
    week 7 - final adjustments/labelling sample/csv outputs/labelling_sample_key.csv
    week 5 - .../01 ad filter robustness/csv outputs/hand_label_vs_flag_merged_records.csv
    week_three_ads_with_sentiment.csv (OCR text; Drive copy)

OUTPUTS (week 7 - final adjustments/notice rules/)
    csv outputs/rule_hits_v3.csv                 row_id + one column per rule
    csv outputs/filter_validation_v1_v2_v3.csv   metrics per version and label set
    csv outputs/rule_hits_by_decade.csv
    csv outputs/precision_audit_sample.csv       30 flagged records per rule to read
    robustness tables/notice_rules_summary.txt
    and flag_v3 added to model runner/csv outputs/filter_versions.csv

RUN
    Paste into one Colab cell (mount Drive first), or run as a file.
    Then re-run 02_filter_versions_and_model_runner.py: it picks up flag_v3 by itself.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 314
AUDIT_N = 30
EXPECTED_ROWS = int(os.environ.get("SOSC314_EXPECTED_ROWS", 172219))
REPO_URL = "https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git"
WEEK7 = "week 7 - final adjustments"
WEEK5 = "week 5 - diagnostics robustness sensitivity validity regression checks and model revisions"
DRIVE_DATA = Path("/content/drive/MyDrive/SOSC314_data")

RULES = {
    "tombstone": [
        r"matter of record",
        r"(?:securities|notes|shares|bonds) (?:have|having) been sold",
        r"have not been registered under the (?:united states )?securities act",
        r"issued in compliance with the requirements",
        r"not an invitation to any person to (?:subscribe|purchase)",
        r"admitted to the official list",
    ],
    "legal_notice": [
        r"notice is hereby given",
        r"(?:annual|extraordinary) general meeting",
        r"judicial publication",
        r"published by order of",
    ],
    "financial_statement": [
        r"balance sheet as (?:at|of)",
        r"consolidated balance sheet",
        r"profit and loss account",
        r"interim results",
        r"interim report \d{4}",
        r"highlights of the year",
        r"three months'? review",
        r"profit attributable to (?:share|stock)holders",
    ],
    "economist_house": [
        r"economist intelligence unit",
        r"economist conferences",
        r"economist\.com/(?:node|debates)",
        r"intelligent life",
        r"economist booklets",
        r"offer to readers",
        r"letters are welcome",
        r"subscription service",
        r"economistsubscriptions",
        r"subscribe to the economist",
        r"the economist plaza",
        r"nico colchester",
    ],
    "editorial": [
        r"also in this section",
        r"% change on year ago",
        r"economics focus",
    ],
}

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ----------------------------------------------------------------------------
# Locations (works pasted into a cell or run as a file)
# ----------------------------------------------------------------------------
def find_repo_root():
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    if os.environ.get("SOSC314_REPO"):
        starts.append(Path(os.environ["SOSC314_REPO"]))
    starts += [Path.cwd(), Path("/content/repo"), Path("/content/Jake_Daniel_Kiran_SOSC314")]
    for s in starts:
        for p in [s, *s.parents]:
            if (p / "week three data cleaning").is_dir():
                return p
    if IN_COLAB:
        target = Path("/content/Jake_Daniel_Kiran_SOSC314")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True,
                       env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        return target
    sys.exit("Could not find the repo. Set SOSC314_REPO to the path of your clone.")


def refresh_repo(root):
    if not IN_COLAB or not (root / ".git").exists():
        return
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    r = subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "--quiet"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin"], env=env)
        subprocess.run(["git", "-C", str(root), "reset", "--hard", "--quiet", "origin/HEAD"],
                       env=env, capture_output=True)


def is_lfs_pointer(p):
    return p.exists() and p.stat().st_size < 2000 and \
        p.read_text(errors="ignore").startswith("version https://git-lfs")


def locate(name, *candidates):
    """First real copy of a file: given candidates, then anywhere in the repo or Drive data folder."""
    for p in candidates:
        if p and p.exists() and not is_lfs_pointer(p):
            return p
    for root in (ROOT, DRIVE_DATA):
        if root.exists():
            hits = [p for p in root.rglob(name) if not is_lfs_pointer(p)]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    return None


ROOT = find_repo_root()
refresh_repo(ROOT)
if IN_COLAB and not Path("/content/drive/MyDrive").exists():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"Note: could not mount Drive here ({e}). Mount it in a cell first.")

OUT = ROOT / WEEK7 / "notice rules"
OUT_CSV, OUT_TAB = OUT / "csv outputs", OUT / "robustness tables"
for d in (OUT_CSV, OUT_TAB):
    d.mkdir(parents=True, exist_ok=True)

log_lines, checks = [], []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def check(name, ok, detail="", soft=False):
    status = "PASS" if ok else ("WARN" if soft else "FAIL")
    checks.append((name, status))
    log(f"[{status}] {name}{'  (' + detail + ')' if detail else ''}")


VERSIONS_FILE = locate("filter_versions.csv",
                       ROOT / WEEK7 / "model runner" / "csv outputs" / "filter_versions.csv",
                       DRIVE_DATA / "model runner" / "csv outputs" / "filter_versions.csv")
LABELS_150 = locate("final_labels_150.csv",
                    ROOT / WEEK7 / "labelling sample" / "csv outputs" / "final_labels_150.csv")
SAMPLE_KEY = locate("labelling_sample_key.csv",
                    ROOT / WEEK7 / "labelling sample" / "csv outputs" / "labelling_sample_key.csv")
KIRAN_MERGED = locate("hand_label_vs_flag_merged_records.csv",
                      ROOT / WEEK5 / "01 ad filter robustness" / "csv outputs" / "hand_label_vs_flag_merged_records.csv")
ADS_FILE = locate("week_three_ads_with_sentiment.csv",
                  Path(os.environ["SOSC314_ADS_FILE"]) if os.environ.get("SOSC314_ADS_FILE") else None,
                  DRIVE_DATA / "week_three_ads_with_sentiment.csv")
missing = {n: p for n, p in {"filter_versions.csv (run task b first)": VERSIONS_FILE,
                             "final_labels_150.csv": LABELS_150, "labelling_sample_key.csv": SAMPLE_KEY,
                             "hand_label_vs_flag_merged_records.csv (01b output)": KIRAN_MERGED,
                             "week_three_ads_with_sentiment.csv": ADS_FILE}.items() if p is None}
if missing:
    sys.exit("Missing inputs:\n  " + "\n  ".join(missing) + "\nCommit them to the repo or put them in Drive's SOSC314_data.")

log("=" * 72)
log("Task c: notice and house-content rules (filter v3)")
log("=" * 72)
for n, p in [("filter versions", VERSIONS_FILE), ("labels (150)", LABELS_150), ("sample key", SAMPLE_KEY),
             ("Kiran 01b merged", KIRAN_MERGED), ("OCR text", ADS_FILE)]:
    log(f"{n:>17}: {p}")

# ----------------------------------------------------------------------------
# 1. Apply the rules to every record
# ----------------------------------------------------------------------------
versions = pd.read_csv(VERSIONS_FILE)
for c in [c for c in versions.columns if c.startswith("flag_")]:
    versions[c] = versions[c].astype(str).str.lower().eq("true")
check("filter_versions.csv has the canonical row count", len(versions) == EXPECTED_ROWS, f"{len(versions):,}")
check("filter_versions.csv has v1 and v2", {"flag_v1", "flag_v2"} <= set(versions.columns))

header = pd.read_csv(ADS_FILE, nrows=0).columns
date_col = next(c for c in header if "date" in c.lower())
ads = pd.read_csv(ADS_FILE, usecols=["OCR_GoogleVision_original", date_col], low_memory=False)
check("OCR file has the canonical row count", len(ads) == len(versions), f"{len(ads):,}")
text = (ads["OCR_GoogleVision_original"].fillna("").astype(str).str.lower()
        .str.replace(r"-\s+", "", regex=True)          # re-join words hyphenated across lines
        .str.replace(r"\s+", " ", regex=True))
decade = pd.to_datetime(ads[date_col], errors="coerce").dt.year // 10 * 10

hits = pd.DataFrame({"row_id": versions["row_id"]})
for name, pats in RULES.items():
    hits[name] = text.str.contains("|".join(f"(?:{p})" for p in pats), regex=True)
hits["any_rule"] = hits[list(RULES)].any(axis=1)
hits.to_csv(OUT_CSV / "rule_hits_v3.csv", index=False)

versions["flag_v3"] = versions["flag_v2"] | hits["any_rule"]
versions.to_csv(VERSIONS_FILE, index=False)
log(f"\nflag_v3 added to {VERSIONS_FILE}")

log(f"\nCorpus counts ({len(versions):,} records):")
log(f"  v1 flags {versions['flag_v1'].sum():,} | v2 flags {versions['flag_v2'].sum():,} | "
    f"v3 flags {versions['flag_v3'].sum():,} ({100 * versions['flag_v3'].mean():.2f}%)")
for name in RULES:
    new = hits[name] & ~versions["flag_v2"]
    log(f"  {name:20} hits {hits[name].sum():6,}  (new beyond v2: {new.sum():6,})")
by_dec = hits[list(RULES) + ["any_rule"]].groupby(decade).sum()
by_dec["records"] = decade.value_counts().sort_index()
by_dec["any_rule_pct"] = (100 * by_dec["any_rule"] / by_dec["records"]).round(2)
by_dec.index.name = "decade"
by_dec.to_csv(OUT_CSV / "rule_hits_by_decade.csv")
log("\nRule hits by decade:\n" + by_dec.to_string())

# ----------------------------------------------------------------------------
# 2. Validation on labelled records
# ----------------------------------------------------------------------------
def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def evaluate(df, label_set, weight=None):
    rows = []
    non_ad = df["non_ad"].to_numpy(bool)
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy(float)
    for v in ["flag_v1", "flag_v2", "flag_v3"]:
        f = df[v].to_numpy(bool)
        tp, fp = int((f & non_ad).sum()), int((f & ~non_ad).sum())
        fn, tn = int((~f & non_ad).sum()), int((~f & ~non_ad).sum())
        lo, hi = wilson(tp, tp + fn)
        row = dict(label_set=label_set, version=v.replace("flag_", ""), n=len(df), non_ads=int(non_ad.sum()),
                   caught=tp, missed=fn, real_ads_flagged=fp, real_ads_kept=tn,
                   recall=tp / max(tp + fn, 1), recall_ci_low=lo, recall_ci_high=hi,
                   precision=tp / (tp + fp) if tp + fp else np.nan, kappa=kappa(f, non_ad))
        if weight is not None:   # corpus-level estimates using the sampling weights
            row["recall_weighted"] = (w * (f & non_ad)).sum() / (w * non_ad).sum()
            row["precision_weighted"] = (w * (f & non_ad)).sum() / (w * f).sum() if (w * f).sum() else np.nan
            row["nonad_share_weighted"] = (w * non_ad).sum() / w.sum()
        rows.append(row)
    return rows


# Development set: 150 stratified records
lab = pd.read_csv(LABELS_150).merge(pd.read_csv(SAMPLE_KEY)[["label_id", "weight", "sample_stratum"]], on="label_id")
lab["non_ad"] = lab["is_ad_final"] == 0
lab = lab.merge(versions, left_on="row_id_0based", right_on="row_id", how="left")
check("All 150 labelled records join to the corpus", lab["flag_v3"].notna().all() and len(lab) == 150)

# Held-out test set: Kiran's 200 via the 01b merged file
km = pd.read_csv(KIRAN_MERGED)
lab_col = "Is_Non_Ad" if "Is_Non_Ad" in km.columns else next(c for c in km.columns if "non" in c.lower() and "ad" in c.lower())
km["non_ad"] = km[lab_col].astype(str).str.lower().isin(["true", "1", "yes"])
km = km[["row_id", "non_ad"]].merge(versions, on="row_id", how="left")
check("Kiran's labels join to the corpus", km["flag_v3"].notna().all(), f"{len(km)} records, {km['non_ad'].sum()} non-ads")

res = pd.DataFrame(evaluate(km, "Kiran 200 (held-out, 2012-2014)") +
                   evaluate(lab, "150 stratified (development)", weight="weight"))
res.to_csv(OUT_CSV / "filter_validation_v1_v2_v3.csv", index=False)

r = res.set_index(["label_set", "version"])
k1, k2 = r.loc[("Kiran 200 (held-out, 2012-2014)", "v1")], r.loc[("Kiran 200 (held-out, 2012-2014)", "v2")]
check("v1 reproduces 01b on Kiran's labels (2/37, kappa 0.085)",
      k1.caught == 2 and k1.non_ads == 37 and abs(k1.kappa - 0.085) < 0.005, f"{int(k1.caught)}/{int(k1.non_ads)}, kappa {k1.kappa:.3f}")
check("v2 reproduces 01b on Kiran's labels (13/37, kappa 0.47)",
      k2.caught == 13 and abs(k2.kappa - 0.47) < 0.01, f"{int(k2.caught)}/{int(k2.non_ads)}, kappa {k2.kappa:.3f}", soft=True)

log("\nFilter validation (non-ad = positive class)")
show = res[["label_set", "version", "caught", "non_ads", "real_ads_flagged", "recall", "precision", "kappa",
            "recall_weighted", "precision_weighted"]].copy()
for c in ["recall", "precision", "kappa", "recall_weighted", "precision_weighted"]:
    show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
log(show.to_string(index=False))
log("recall_weighted / precision_weighted: corpus-level estimates from the 150 using sampling weights.")
log("The 150 were used to write the rules, so their scores are optimistic; Kiran's 200 are the fair test.")

# What v3 still misses / wrongly removes, on both label sets
log("\nStill missed by v3 (development set):")
for _, x in lab[lab["non_ad"] & ~lab["flag_v3"]].iterrows():
    log(f"  {x.label_id} {x.category_final}")
wrong = pd.concat([lab.loc[~lab["non_ad"] & lab["flag_v3"], ["row_id"]], km.loc[~km["non_ad"] & km["flag_v3"], ["row_id"]]])
log(f"Real ads removed by v3 across both label sets: {len(wrong)}")

# ----------------------------------------------------------------------------
# 3. Precision audit: flagged records to read by hand
# ----------------------------------------------------------------------------
labelled_ids = set(lab["row_id"]) | set(km["row_id"])
rng = np.random.default_rng(SEED)
aud = []
for name in RULES:
    pool = hits.index[hits[name] & ~hits["row_id"].isin(labelled_ids)]
    take = rng.choice(pool, size=min(AUDIT_N, len(pool)), replace=False) if len(pool) else []
    for i in take:
        m = re.search("|".join(f"(?:{p})" for p in RULES[name]), text.iat[i])
        start = max(0, m.start() - 150) if m else 0
        aud.append(dict(rule=name, row_id=int(hits.at[i, "row_id"]), decade=decade.iat[i],
                        matched=m.group(0) if m else "", already_in_v2=bool(versions.at[i, "flag_v2"]),
                        excerpt=text.iat[i][start:start + 600], is_non_ad_yes_no=""))
pd.DataFrame(aud).to_csv(OUT_CSV / "precision_audit_sample.csv", index=False, encoding="utf-8-sig")
log(f"\nPrecision audit: {len(aud)} flagged records (up to {AUDIT_N} per rule) in precision_audit_sample.csv")

n_fail = sum(s == "FAIL" for _, s in checks)
n_warn = sum(s == "WARN" for _, s in checks)
log(f"\nBuilt-in checks: {len(checks) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failed")
log("Next: re-run 02_filter_versions_and_model_runner.py to get the models under v3.")
(OUT_TAB / "notice_rules_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")

if IN_COLAB and Path("/content/drive/MyDrive").exists():
    import shutil
    shutil.copytree(OUT, DRIVE_DATA / "notice rules", dirs_exist_ok=True)
    dst = DRIVE_DATA / "model runner" / "csv outputs"
    dst.mkdir(parents=True, exist_ok=True)
    if VERSIONS_FILE.resolve() != (dst / "filter_versions.csv").resolve():
        shutil.copy(VERSIONS_FILE, dst / "filter_versions.csv")
    print(f"\nCopies saved to Drive: {DRIVE_DATA / 'notice rules'} and the updated filter_versions.csv")
