"""
02_filter_versions_and_model_runner.py
SOSC 314 final phase, task b.

PURPOSE
    1. Build the ad / non-ad filter versions in one file:
         flag_v1 = the Week 4 quality flags (any_flag), case-sensitive brand blocklist
         flag_v2 = the same flags with the brand blocklist matched case-insensitively
       Later filters (task c rules, the classifier) are added as new flag_* columns
       to the same file, and this script runs every one of them automatically.
    2. Re-run the key recession models once per filter version, so we can see
       whether a cleaner corpus changes the answer.

QUESTION
    Does removing more non-advertising records (v2 instead of v1) change the
    recession estimates? (Week 5 script 01 showed v1 barely mattered.)

WHY IT MATTERS
    The Week 5 feedback asked us to tighten the ad vs non-ad filter so the null
    result is more convincing. This script is the engine every tighter filter
    goes through.

MODELS (all four outcomes: Harvard / Loughran-McDonald x positive / negative)
    Quarterly (268 quarters, HAC / Newey-West, 4 lags), macro and pooled series:
        raw:     y ~ recession
        trend:   y ~ recession + linear trend
        decade:  y ~ recession + decade effects
    Ad level (errors clustered by year), advertisers seen in both periods:
        decade:            y ~ recession | decade
        decade+advertiser: y ~ recession | decade + Brand
    Every effect is also shown as % of the series' own mean.

HOW SENTIMENT COLUMNS ARE FOUND
    The script does not guess column names. It tries every positive / negative
    column in the two large files and keeps the one that exactly reproduces the
    Week 4 quarterly files (week4_aggregations/quarterly_*.csv). If nothing
    reproduces them, it stops and prints what it tried.

FILES LOADED
    week 4 - updated quality filtered analysis/Ad_filtering/week_four_quality_flags.csv
    week 4 - updated quality filtered analysis/week4_aggregations/quarterly_{harvard,loughran}_{macro,pooled}.csv
    week three data cleaning/nber_recession_table.csv
    week_three_ads_with_sentiment.csv              (Drive copy, or repo if LFS pulled)
    economist_loughran_mcdonald_sentiment.csv      (Drive copy, or repo if LFS pulled)

OUTPUTS (week 7 - final adjustments/model runner/)
    csv outputs/filter_versions.csv                 row_id + one flag_* column per version
    csv outputs/model_results_by_filter_version.csv every estimate, all versions
    robustness tables/model_runner_summary.txt      readable summary + built-in checks
    diagnostic images/lm_positive_by_filter_version.png

RUN (either way works)
    a) Paste the whole script into one Colab cell and run it.
    b) python "week 7 - final adjustments/02_filter_versions_and_model_runner.py"
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
HAC_LAGS = 4
EXPECTED_ROWS = int(os.environ.get("SOSC314_EXPECTED_ROWS", 172219))
REPO_URL = "https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git"
WEEK7_FOLDER = "week 7 - final adjustments"
DRIVE_DATA = Path("/content/drive/MyDrive/SOSC314_data")

# Week 5 numbers this script should reproduce (soft checks: WARN, not stop)
REFERENCE = {
    "flagged_v1": 3303, "flagged_v2": 3704,
    # LM positive, macro, majority definition, % of mean (script 04c)
    ("quarterly", "raw"): -6.0, ("quarterly", "trend"): -1.2, ("quarterly", "decade"): -1.7,
    # LM positive, ad level (script 06)
    ("ad", "decade"): -2.6, ("ad", "decade+advertiser"): -3.4,
    "ad_n": 156557, "ad_switchers": 3991,
}

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


ensure("statsmodels")
ensure("pyfixest")
import statsmodels.api as sm   # noqa: E402
import pyfixest as pf          # noqa: E402
import matplotlib              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------
# Repo, Drive and file locations (works pasted into a cell or run as a file)
# ----------------------------------------------------------------------------
def is_repo_root(p):
    return (p / "week three data cleaning").is_dir()


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
            if is_repo_root(p):
                return p
    if IN_COLAB:
        target = Path("/content/Jake_Daniel_Kiran_SOSC314")
        print(f"Repo not found. Cloning it to {target} (large LFS files skipped)...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True,
                       env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        return target
    sys.exit("Could not find the repo. Set SOSC314_REPO to the path of your clone.")


def refresh_repo(root):
    if not IN_COLAB or not (root / ".git").exists():
        return
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    r = subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "--quiet"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin"], env=env)
        subprocess.run(["git", "-C", str(root), "reset", "--hard", "--quiet", "origin/HEAD"],
                       env=env, capture_output=True)
    print("Repo matches GitHub.")


def mount_drive():
    if not IN_COLAB or Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"Note: could not mount Drive here ({e}). Mount it in a cell first.")


def is_lfs_pointer(p):
    return p.exists() and p.stat().st_size < 2000 and \
        p.read_text(errors="ignore").startswith("version https://git-lfs")


def first_real(paths):
    return next((p for p in paths if p.exists() and not is_lfs_pointer(p)), None)


ROOT = find_repo_root()
refresh_repo(ROOT)
mount_drive()

W4 = ROOT / "week 4 - updated quality filtered analysis"
FLAGS_FILE = W4 / "Ad_filtering" / "week_four_quality_flags.csv"
AGG_DIR = W4 / "week4_aggregations"
NBER_FILE = ROOT / "week three data cleaning" / "nber_recession_table.csv"
ADS_CANDIDATES = [Path(os.environ["SOSC314_ADS_FILE"])] if os.environ.get("SOSC314_ADS_FILE") else []
ADS_CANDIDATES += [DRIVE_DATA / "week_three_ads_with_sentiment.csv",
                   ROOT / "week three data cleaning" / "Archived Data" / "week_three_ads_with_sentiment.csv"]
LM_CANDIDATES = [Path(os.environ["SOSC314_LM_FILE"])] if os.environ.get("SOSC314_LM_FILE") else []
LM_CANDIDATES += [DRIVE_DATA / "economist_loughran_mcdonald_sentiment.csv",
                  W4 / "second dict and data differences" / "csv" / "economist_loughran_mcdonald_sentiment.csv"]

OUT = ROOT / WEEK7_FOLDER / "model runner"
OUT_CSV, OUT_TAB, OUT_IMG = OUT / "csv outputs", OUT / "robustness tables", OUT / "diagnostic images"
for d in (OUT_CSV, OUT_TAB, OUT_IMG):
    d.mkdir(parents=True, exist_ok=True)

log_lines, checks = [], []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def check(name, ok, detail="", soft=False):
    status = "PASS" if ok else ("WARN" if soft else "FAIL")
    checks.append((name, status))
    log(f"[{status}] {name}{'  (' + detail + ')' if detail else ''}")


for p in [FLAGS_FILE, NBER_FILE, *[AGG_DIR / f"quarterly_{d}_{a}.csv" for d in ("harvard", "loughran")
                                   for a in ("macro", "pooled")]]:
    if not p.exists():
        sys.exit(f"Missing input: {p}")

ADS_FILE, LM_FILE = first_real(ADS_CANDIDATES), first_real(LM_CANDIDATES)
for name, f, cands in (("ads (Harvard)", ADS_FILE, ADS_CANDIDATES), ("Loughran-McDonald", LM_FILE, LM_CANDIDATES)):
    if f is None:
        sys.exit(f"Could not find the {name} file. Tried:\n" + "\n".join(f"  {p}" for p in cands) +
                 "\nMount Drive, or set SOSC314_ADS_FILE / SOSC314_LM_FILE to its full path.")

log("=" * 72)
log("Task b: filter versions and model runner")
log("=" * 72)
log(f"Harvard file: {ADS_FILE}")
log(f"LM file:      {LM_FILE}")

# ----------------------------------------------------------------------------
# 1. Flags and filter versions
# ----------------------------------------------------------------------------
flags = pd.read_csv(FLAGS_FILE)
for c in ["is_non_advertiser", "is_multipage", "too_short", "any_flag"]:
    flags[c] = flags[c].astype(str).str.strip().str.lower() == "true"
check("Flags file has the canonical row count", len(flags) == EXPECTED_ROWS, f"{len(flags):,}")
check("Flags row_id is the 0-based row position", (flags["row_id"].to_numpy() == np.arange(len(flags))).all())

brand_norm = flags["Brand"].fillna("").astype(str).str.strip()
blocked = sorted(set(brand_norm[flags["is_non_advertiser"]]))
blocked_lower = {b.lower() for b in blocked}
nonadv_v2 = brand_norm.str.lower().isin(blocked_lower)

versions = pd.DataFrame({"row_id": flags["row_id"]})
versions["flag_v1"] = flags["any_flag"]
versions["flag_v2"] = nonadv_v2 | flags["is_multipage"] | flags["too_short"]
check("v1 flags the Week 4 count", versions["flag_v1"].sum() == REFERENCE["flagged_v1"],
      f"{versions['flag_v1'].sum():,} vs {REFERENCE['flagged_v1']:,}")
check("v2 flags the Week 5 01b count", versions["flag_v2"].sum() == REFERENCE["flagged_v2"],
      f"{versions['flag_v2'].sum():,} vs {REFERENCE['flagged_v2']:,}", soft=True)
check("v2 contains every v1 record", (versions["flag_v2"] | ~versions["flag_v1"]).all())
log(f"Brand blocklist: {len(blocked)} brands. Extra records caught by case-insensitive matching: "
    f"{int((nonadv_v2 & ~flags['is_non_advertiser']).sum()):,}")

# Keep any later versions (task c) already saved in the file, then save
vfile = OUT_CSV / "filter_versions.csv"
for src in (vfile, DRIVE_DATA / "model runner" / "csv outputs" / "filter_versions.csv"):
    if not src.exists():
        continue
    old = pd.read_csv(src)
    for c in old.columns:
        if c.startswith("flag_") and c not in versions.columns and len(old) == len(versions):
            versions[c] = old[c].astype(str).str.lower().eq("true")
            log(f"Kept later filter version from {src.parent.parent.name}/.../{src.name}: {c} "
                f"({int(versions[c].sum()):,} records)")
versions.to_csv(vfile, index=False)
VERSION_COLS = ["none"] + [c for c in versions.columns if c.startswith("flag_")]

# ----------------------------------------------------------------------------
# 2. Quarters, recession and decade
# ----------------------------------------------------------------------------
date = pd.to_datetime(flags["DateOfIssue"], errors="coerce")
quarter = date.dt.year.astype(str) + "Q" + date.dt.quarter.astype(str)
nber = pd.read_csv(NBER_FILE)
for c in ["Recession_Majority", "Recession_Any"]:
    nber[c] = nber[c].astype(str).str.strip().str.lower() == "true"
check("NBER table has 268 quarters and 42 majority recession quarters",
      len(nber) == 268 and nber["Recession_Majority"].sum() == 42,
      f"{len(nber)} quarters, {nber['Recession_Majority'].sum()} recession")
qinfo = nber.set_index("Year_Quarter")[["Recession_Majority", "Recession_Any"]]
qinfo["t"] = np.arange(len(qinfo))
qinfo["decade"] = qinfo.index.str[:4].astype(int) // 10 * 10

base = pd.DataFrame({"row_id": flags["row_id"], "q": quarter, "year": date.dt.year,
                     "Brand": brand_norm.replace("", np.nan), "tokens": flags["token_count"].astype(float)})
base = base.join(qinfo, on="q")
check("Every record falls in an NBER quarter", base["Recession_Majority"].notna().all())


# Advertiser key: reuse script 06's own normalise_brand() so both scripts agree exactly
def load_function_from(path, func_name):
    """Pull one function, and the module-level constants and imports it needs, out of a
    script without running the script."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigns = {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    assigns[t.id] = n
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    needed, stack, order = set(), [func_name], []
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        node = defs.get(name) or assigns.get(name)
        if node is None:
            continue
        needed.add(name)
        order.append(node)
        stack += [x.id for x in ast.walk(node) if isinstance(x, ast.Name)]
    order.sort(key=lambda n: n.lineno)
    module = ast.Module(body=imports + order, type_ignores=[])
    ns = {}
    exec(compile(module, str(path), "exec"), ns)
    return ns[func_name], ns


SCRIPT_06 = (ROOT / "week 5 - diagnostics robustness sensitivity validity regression checks and model revisions"
             / "06 advertiser composition" / "06_advertiser_composition_and_within_advertiser_tone.py")
MIN_ADS_PER_BRAND = 2
try:
    import re as _re
    normalise_brand, _ = load_function_from(SCRIPT_06, "normalise_brand")
    _m = _re.search(r"^MIN_ADS_PER_BRAND\s*=\s*(\d+)", SCRIPT_06.read_text(errors="ignore"), _re.M)
    MIN_ADS_PER_BRAND = int(_m.group(1)) if _m else 2
    keys = {b: normalise_brand(b) for b in base["Brand"].dropna().unique()}
    base["Brand_Key"] = base["Brand"].map(keys).fillna("(missing brand)")
    log(f"Advertiser key: normalise_brand() from script 06; {len(keys):,} brand strings -> "
        f"{base['Brand_Key'].nunique():,} advertisers")
except Exception as e:
    base["Brand_Key"] = base["Brand"].str.lower().fillna("(missing brand)")
    check("Script 06 brand normalisation loaded", False, f"{type(e).__name__}: {e}; using lower-cased brand", soft=True)

# ----------------------------------------------------------------------------
# 3. Find the sentiment columns by reproducing the Week 4 quarterly files
# ----------------------------------------------------------------------------
def numeric_header(path):
    return list(pd.read_csv(path, nrows=0).columns)


def discover(path, dict_name):
    """Return {'Pos': Series, 'Neg': Series} of per-ad ratios plus pooled num/den,
    using the columns that reproduce week4_aggregations/quarterly_<dict>_*.csv."""
    cols = numeric_header(path)
    cand = [c for c in cols if ("pos" in c.lower() or "neg" in c.lower())
            and "position" not in c.lower()]
    dens = [c for c in cols if any(k in c.lower() for k in ("token", "word", "total", "length", "len"))]
    use = list(dict.fromkeys(cand + dens + [c for c in cols if c.lower() == "row_id"]))
    if not cand:
        sys.exit(f"No positive/negative columns in {path.name}. Columns: {cols}")
    df = pd.read_csv(path, usecols=use, low_memory=False)
    if "row_id" in df.columns:
        df = df.set_index("row_id").reindex(base["row_id"]).reset_index(drop=True)
    if len(df) != len(base):
        sys.exit(f"{path.name} has {len(df):,} rows, expected {len(base):,}.")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    macro = pd.read_csv(AGG_DIR / f"quarterly_{dict_name}_macro.csv").set_index("Year_Quarter")
    pooled = pd.read_csv(AGG_DIR / f"quarterly_{dict_name}_pooled.csv").set_index("Year_Quarter")

    # Which population did Week 4 aggregate? Match the Ads counts.
    pops = {"all records": pd.Series(True, index=base.index), "v1-filtered": ~versions["flag_v1"]}
    pop_name = min(pops, key=lambda k: (base.loc[pops[k]].groupby("q").size()
                                        .reindex(macro.index).fillna(0) - macro["Ads"]).abs().sum())
    keep = pops[pop_name]
    n_err = (base.loc[keep].groupby("q").size().reindex(macro.index).fillna(0) - macro["Ads"]).abs().sum()

    den_options = {"tokens(flags file)": base["tokens"]}
    den_options.update({d: df[d] for d in dens if d in df})
    out, report = {}, []
    for s in ("Pos", "Neg"):
        best = None
        for c in [c for c in cand if s.lower() in c.lower()]:
            for dname, dser in [(None, None), *den_options.items()]:
                r = df[c] if dname is None else df[c] / dser.replace(0, np.nan)
                if r.dropna().gt(1.0).mean() > 0.01:      # not a share
                    continue
                q_mean = r[keep].groupby(base.loc[keep, "q"]).mean().reindex(macro.index)
                err = float((q_mean - macro[s]).abs().max())
                if best is None or err < best[0]:
                    best = (err, c, dname, r)
        if best is None:
            sys.exit(f"No usable {s} column in {path.name}. Candidates: {cand}")
        err, c, dname, r = best
        # pooled: words in category / words, summed over the quarter
        pbest = None
        for pdn, pds in den_options.items():
            num = r * pds
            pq = (num[keep].groupby(base.loc[keep, "q"]).sum() /
                  pds[keep & r.notna()].groupby(base.loc[keep & r.notna(), "q"]).sum()).reindex(pooled.index)
            perr = float((pq - pooled[s]).abs().max())
            if pbest is None or perr < pbest[0]:
                pbest = (perr, pdn, num, pds)
        # Ad-level models follow script 06: category count / Total_Words
        tw = df["Total_Words"].replace(0, np.nan) if "Total_Words" in df.columns else None
        ad_ratio = df[c] / tw if (tw is not None and dname is not None) else r
        out[s] = {"ratio": r, "num": pbest[2], "den": pbest[3], "ad_ratio": ad_ratio}
        report.append((s, c, dname or "(already a share)", err, pbest[1], pbest[0]))
    log(f"\n{dict_name.title()} columns (population Week 4 used: {pop_name}, Ads-count mismatch {n_err:.0f}):")
    for s, c, dname, err, pdn, perr in report:
        log(f"  {s}: column '{c}' / {dname}  -> macro max error {err:.2e}; pooled denominator {pdn}, max error {perr:.2e}")
        check(f"{dict_name.title()} {s} reproduces Week 4 macro series", err < 1e-6, f"{err:.1e}", soft=err < 1e-4)
        check(f"{dict_name.title()} {s} reproduces Week 4 pooled series", perr < 1e-6, f"{perr:.1e}", soft=perr < 1e-4)
    return out, pop_name


harvard, pop_h = discover(ADS_FILE, "harvard")
lm, pop_l = discover(LM_FILE, "loughran")
DICTS = {"Harvard": harvard, "LM": lm}
hard_fail = [n for n, s in checks if s == "FAIL"]
if hard_fail:
    log("\nStopping: the inputs do not reproduce the Week 4 files, so any estimate would be unreliable.")
    (OUT_TAB / "model_runner_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    sys.exit("Failed checks: " + "; ".join(hard_fail))

# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
rows = []


def add(version, level, d, agg, s, rec, spec, coef, se, p, mean_y, n):
    pct = 100 * coef / mean_y
    rows.append(dict(version=version, level=level, dictionary=d, aggregation=agg, outcome=s,
                     recession_def=rec, spec=spec, coef=coef, se=se, p=p, mean_y=mean_y,
                     pct_of_mean=pct, ci_low_pct=100 * (coef - 1.96 * se) / mean_y,
                     ci_high_pct=100 * (coef + 1.96 * se) / mean_y, n=n))


def quarterly_models(y, version, d, agg, s):
    q = qinfo.join(y.rename("y"), how="left")
    for rec in ("Recession_Majority", "Recession_Any"):
        X_base = q[[rec]].astype(float)
        specs = {"raw": X_base,
                 "trend": X_base.join(q["t"]),
                 "decade": X_base.join(pd.get_dummies(q["decade"], prefix="d", drop_first=True).astype(float))}
        for spec, X in specs.items():
            data = sm.add_constant(X).join(q["y"]).dropna()        # quarters with no kept ads drop out
            Xd = data.drop(columns="y")
            Xd = Xd.loc[:, (Xd != 0).any() | (Xd.columns == "const")]  # a decade emptied by a filter
            fit = sm.OLS(data["y"], Xd).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            add(version, "quarterly", d, agg, s, rec.replace("Recession_", "").lower(), spec,
                fit.params[rec], fit.bse[rec], fit.pvalues[rec], q["y"].mean(), int(fit.nobs))


def ad_models(ratio, keep, version, d, s):
    """Same sample rule as Week 5 script 06: normalised advertiser (Brand_Key) with
    at least MIN_ADS_PER_BRAND kept ads; missing brands form one '(missing brand)' key."""
    df = base.loc[keep, ["year", "decade", "Brand_Key", "Recession_Majority"]].copy()
    df["y"] = ratio[keep]
    df = df.dropna(subset=["y"])
    df["rec"] = df["Recession_Majority"].astype(float)
    n_ads = df.groupby("Brand_Key")["y"].transform("size")
    df = df[n_ads >= MIN_ADS_PER_BRAND]
    states = df.groupby("Brand_Key")["rec"].agg(["min", "max"])
    switchers = int(((states["min"] == 0) & (states["max"] == 1)).sum())
    for spec, fml in (("decade", "y ~ rec | decade"), ("decade+advertiser", "y ~ rec | decade + Brand_Key")):
        fit = pf.feols(fml, data=df, vcov={"CRV1": "year"})
        t = fit.tidy().loc["rec"]
        add(version, "ad", d, "per ad", s, "majority", spec, t["Estimate"], t["Std. Error"],
            t["Pr(>|t|)"], df["y"].mean(), len(df))
    return len(df), df["Brand_Key"].nunique(), switchers


ad_samples = {}
log("\nRunning models...")
for v in VERSION_COLS:
    keep = pd.Series(True, index=base.index) if v == "none" else ~versions[v]
    vname = v.replace("flag_", "")
    for d, parts in DICTS.items():
        for s in ("Pos", "Neg"):
            r, num, den = parts[s]["ratio"], parts[s]["num"], parts[s]["den"]
            ok = keep & r.notna()
            macro_y = r[ok].groupby(base.loc[ok, "q"]).mean()
            pooled_y = num[ok].groupby(base.loc[ok, "q"]).sum() / den[ok].groupby(base.loc[ok, "q"]).sum()
            quarterly_models(macro_y, vname, d, "macro", s)
            quarterly_models(pooled_y, vname, d, "pooled", s)
            n_ads, n_brands, n_switch = ad_models(parts[s]["ad_ratio"], keep, vname, d, s)
    ad_samples[vname] = (n_ads, n_switch)
    log(f"  {vname:>5}: kept {int(keep.sum()):,} records; ad-level sample {n_ads:,} ads, "
        f"{n_brands:,} advertisers, {n_switch:,} seen in both recession and expansion quarters")

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV / "model_results_by_filter_version.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Checks against Week 5 and the comparison table
# ----------------------------------------------------------------------------
week4_pop = "none" if pop_l == "all records" else "v1"
log(f"\nWeek 5 reference values were computed on: {pop_l} (version '{week4_pop}' here)")
sel = res[(res.dictionary == "LM") & (res.outcome == "Pos") & (res.recession_def == "majority")]
for spec in ("raw", "trend", "decade"):
    got = sel[(sel.version == week4_pop) & (sel.level == "quarterly") & (sel.aggregation == "macro") &
              (sel.spec == spec)]["pct_of_mean"].iloc[0]
    ref = REFERENCE[("quarterly", spec)]
    check(f"LM positive quarterly {spec} matches 04c", abs(got - ref) < 0.2, f"{got:.2f}% vs {ref}%", soft=True)
ref_v = min(ad_samples, key=lambda v: abs(ad_samples[v][0] - REFERENCE["ad_n"]))
check("An ad-level sample matches script 06 exactly (ads)", ad_samples[ref_v][0] == REFERENCE["ad_n"],
      f"{ad_samples[ref_v][0]:,} ads in version '{ref_v}' vs {REFERENCE['ad_n']:,}", soft=True)
check("Same version matches 06's advertisers seen in both periods", ad_samples[ref_v][1] == REFERENCE["ad_switchers"],
      f"{ad_samples[ref_v][1]:,} vs {REFERENCE['ad_switchers']:,}", soft=True)
log(f"Script 06 used the population of version '{ref_v}'.")
for spec in ("decade", "decade+advertiser"):
    got = sel[(sel.version == ref_v) & (sel.level == "ad") & (sel.spec == spec)]["pct_of_mean"].iloc[0]
    ref = REFERENCE[("ad", spec)]
    check(f"LM positive ad-level {spec} matches 06", abs(got - ref) < 0.1,
          f"{got:.2f}% (version {ref_v}) vs {ref}%", soft=True)

log("\nLM positive, majority definition: % of mean (p-value) by filter version")
show = sel[(sel.aggregation.isin(["macro", "per ad"]))].copy()
show["cell"] = show.apply(lambda r: f"{r.pct_of_mean:+.2f} (p={r.p:.3f})", axis=1)
show["model"] = show.level + " | " + show.spec
log(show.pivot(index="model", columns="version", values="cell")
    .reindex(["quarterly | raw", "quarterly | trend", "quarterly | decade",
              "ad | decade", "ad | decade+advertiser"]).to_string())

log("\nAll four outcomes, quarterly decade-effects model (macro), % of mean:")
t2 = res[(res.level == "quarterly") & (res.spec == "decade") & (res.aggregation == "macro") &
         (res.recession_def == "majority")]
t2 = t2.assign(series=t2.dictionary + " " + t2.outcome,
               cell=t2.apply(lambda r: f"{r.pct_of_mean:+.2f} (p={r.p:.3f})", axis=1))
log(t2.pivot(index="series", columns="version", values="cell").to_string())

sig = res[(res.p < 0.05)]
log(f"\nEstimates with p < .05: {len(sig)} of {len(res)} "
    f"(about {0.05 * len(res):.0f} expected by chance if there were no effects at all).")

# ----------------------------------------------------------------------------
# 6. Figure: LM positive by filter version
# ----------------------------------------------------------------------------
order = ["quarterly | raw", "quarterly | trend", "quarterly | decade", "ad | decade", "ad | decade+advertiser"]
vers = [v.replace("flag_", "") for v in VERSION_COLS]
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["#9AA3B2", "#1F4E8C", "#C2410C", "#2E7D32", "#6A1B9A"]
for i, v in enumerate(vers):
    sub = show[show.version == v].set_index("model").reindex(order)
    y = np.arange(len(order)) + (i - (len(vers) - 1) / 2) * 0.18
    ax.errorbar(sub["pct_of_mean"], y, xerr=[sub["pct_of_mean"] - sub["ci_low_pct"],
                                             sub["ci_high_pct"] - sub["pct_of_mean"]],
                fmt="o", ms=5, capsize=2, color=colors[i % len(colors)], label=f"filter {v}")
ax.axvline(0, color="#555", lw=1)
ax.set_yticks(np.arange(len(order)), order)
ax.invert_yaxis()
ax.set_xlabel("Recession effect on Loughran-McDonald positive, % of series mean (95% CI)")
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_IMG / "lm_positive_by_filter_version.png", dpi=200)
plt.close(fig)

n_fail = sum(s == "FAIL" for _, s in checks)
n_warn = sum(s == "WARN" for _, s in checks)
log(f"\nBuilt-in checks: {len(checks) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} failed")
log("Outputs:")
for p in (vfile, OUT_CSV / "model_results_by_filter_version.csv",
          OUT_TAB / "model_runner_summary.txt", OUT_IMG / "lm_positive_by_filter_version.png"):
    log(f"  {p.relative_to(ROOT)}")
(OUT_TAB / "model_runner_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")

if IN_COLAB and Path("/content/drive/MyDrive").exists():
    import shutil
    shutil.copytree(OUT, DRIVE_DATA / "model runner", dirs_exist_ok=True)
    print(f"\nCopy of all outputs saved to Drive: {DRIVE_DATA / 'model runner'}")
