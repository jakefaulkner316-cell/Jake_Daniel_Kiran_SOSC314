# Advertising Sentiment and the Business Cycle: Evidence from *The Economist* Advertisements

SOSC 314 Computational Social Science, Duke Kunshan University (Prof. Markus Neumann), Fall 2026
Jake Faulkner, Daniel Monteiro and Kiran Perla

**Read the report:** [to be added
**Report notebook:** [to be adeed]

## Project summary

We ask whether the sentiment of advertising language in *The Economist* differs between U.S. recessions and expansions. The corpus is 172,219 branded ads from 1948 to 2014 (268 quarters), scored with the Harvard General Inquirer and Loughran-McDonald dictionaries and compared across NBER recession dates.

A raw comparison shows slightly less positive language in recession quarters. That gap does not survive better measurement. Long-run drift in ad language is 4 to 40 times larger than the recession gap, and quarterly models with time controls and HAC errors find no reliable recession effect. A validated ad vs non-ad filter (recall 65%, kappa 0.72 on held-out labels) changes no conclusion. Two leads remain for future work: lower Loughran-McDonald positive tone within advertisers, and a shift from technology to industrial topics in recessions. All results are descriptive, not causal.

## Repository structure

The repository is organised by project stage. Each stage folder holds its scripts and their outputs (`csv outputs/`, `diagnostic images/`, `robustness tables/`).

| Folder | Contents |
|---|---|
| `Initial file imports/` | Raw inputs: the Master Dataset (Git LFS), Harvard General Inquirer spreadsheet, FRED and World Bank data |
| `Week 1: Early Statistical Analysis/`, `early statistical analysis/` | First exploration of the corpus and economic data |
| `week three data cleaning/` | Corpus construction, Harvard scoring, NBER recession table (`nber_recession_table.csv`) |
| `week 4 - updated quality filtered analysis/` | Loughran-McDonald scoring, Week 4 quality flags, quarterly aggregations |
| `week 5 - diagnostics robustness .../` | Filter validation, preprocessing sensitivity, regression diagnostics, time controls, topic stability, advertiser composition |
| `week 7 - final adjustments/` | Labelling guide and sample, filter versions v1 to v3, model runner (all 224 estimates), notice and house-content rules |
| `topic analysis/` | LDA topic-share regressions and the structural topic model (R) |
| `fighting words/` | Fightin' words with a permutation threshold |
| `final_report/` | Report notebook, build script, figures |
| `docs/` | Rendered report served by GitHub Pages (`index.html`) |
| `METHODOLOGY.md` | Running record of data-treatment decisions |

## Getting the data

1. **Ads.** Download "The Economist Historical Advertisements: Master Dataset" (Kluge, Gehrmann and Stahl 2023) from OpenBIGData: https://openbigdata.org/resource/the-economist-historical-advertisements-master-dataset/. Save it as `Initial file imports/The Economist Historical Advertisements – Master Dataset.csv`.
2. **Dictionaries.** Harvard General Inquirer (`Initial file imports/inquireraugmented.xls`, from https://inquirer.sites.fas.harvard.edu/spreadsheet_guide.htm) and the Loughran-McDonald Master Dictionary (https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
3. **Large derived files.** The ad-level sentiment files (about 250 MB each) are stored with Git LFS. Clone without them and pull only what you need, because the group's LFS bandwidth is limited:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/jakefaulkner316-cell/Jake_Daniel_Kiran_SOSC314.git
   cd Jake_Daniel_Kiran_SOSC314
   git lfs pull --include="week three data cleaning/Archived Data/week_three_ads_with_sentiment.csv"
   ```
   The week 5 and week 7 scripts also accept copies of the two large files through the environment variables `SOSC314_ADS_FILE` and `SOSC314_LM_FILE` (for example, copies on Google Drive in Colab).

The report notebook itself needs none of the large files. It reads only small committed outputs.

## Environment

- Python 3.10 or later: `pip install -r requirements.txt`
- R 4.2 or later for the structural topic model: `install.packages(c("stm", "quanteda", "readr", "dplyr", "lmtest", "sandwich"))`
- Most scripts were run in Google Colab. Mount Google Drive in its own cell first, then run each script as a file (`python script.py`) or paste it into one cell. On Colab, install Git LFS with `apt-get install git-lfs` before any `git lfs` command.

## Reproducing the results (run order)

Run the stages in order. Each script's header lists its inputs and outputs, and most check themselves against earlier numbers.

| Step | Script(s) | Produces |
|---|---|---|
| 1. Corpus and Harvard scores | In `week three data cleaning/week_3_scripts/` (run from `week three data cleaning/`): `week_three_cleaning.py`, `week_three_sentiment.py`, `week_three_sentiment_stopwords.py` | `week_three_ads_with_sentiment.csv` and quarterly series. Its row order (newest first) defines `row_id`, the join key for every later step; never sort it |
| 2. Recession dates | `week3_recession_dates.py` | `nber_recession_table.csv` (42 majority and 48 any-month recession quarters) |
| 3. Loughran-McDonald scores | `week 4 .../second dict and data differences.py` | `economist_loughran_mcdonald_sentiment.csv` |
| 4. Week 4 flags and aggregation | `Ad_filtering/week4_initial_flags.py`, then `week4_aggregations/Aggregations_and_Sensitivity.py` | `week_four_quality_flags.csv`, `quarterly_{harvard,loughran}_{macro,pooled}.csv` |
| 5. Week 5 diagnostics | In `week 5 .../`: `01`, `01b`, `02`, `03a`, `03b`, `04`, `04b`, `04c`, `04d`, `04_topic_model_diagnostics.py`, `06` | Filter validation, diagnostics, time-control and advertiser models |
| 6. Labelling sample | `week 7 .../01_draw_labelling_sample.py` (labels in `labelling sheets/`, final labels in `csv outputs/final_labels_150.csv`) | 150-record decade-stratified sample |
| 7. Filters and models | `02_filter_versions_and_model_runner.py`, then `03_notice_and_house_content_rules.py`, then `02` again | `filter_versions.csv` (v1 to v3), `model_results_by_filter_version.csv`, `notice rules/` outputs |
| 8. Topics and words | `topic analysis/05_topic_share_regression.py`; `Rscript "topic analysis/structural topic model/06_structural_topic_model.R"` and `07_stm_robustness.R` (from the repo root); `fighting words/05_fighting_words_analysis.py` | Topic and word results |
| 9. Report | `bash final_report/build_html.sh` | Executes the notebook and writes `docs/index.html` |

**GitHub Pages:** Settings, then Pages, then "Deploy from a branch", branch `main`, folder `/docs`.

## Use of generative AI

We used Claude (Anthropic) and ChatGPT (OpenAI) to help write and debug code, to draft and edit text, and to label records. Claude labelled the 150-record development sample and read the precision-audit sample using our written labelling guide, and a group member second-coded 30 of the development records. Research questions, design decisions and interpretation are the group's own. The report's acknowledgments section gives details.
