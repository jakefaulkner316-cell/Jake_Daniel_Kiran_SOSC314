**Data Treatment:**

   - **Recession dating.** We date recessions using the NBER business cycle chronology rather than the common two-consecutive-negative-quarters rule. The NBER committee weighs employment, real income, industrial production and sales, not GDP alone, and dates turning points by month. The two definitions disagree: the two-quarter rule misses the 2001 and 1960–61 recessions entirely. Converting monthly dates to quarters takes two decisions. The NBER treats the peak as the last month of expansion, so our recessions start the month after the peak and run through the trough. (FRED's monthly recession indicator instead counts the peak month as recessionary, so the two sources can disagree at the edges of a recession.) And since a quarter can hold zero to three recession months, we call a quarter recessionary when at least two of its months fall within a contraction. This gives 42 recession quarters in 11 episodes; an any-month rule gives 48, and the 6 quarters where they differ serve as a robustness check. The verdict is the same under both rules.

   - **Bretton Woods split.** The collapse of the Bretton Woods system in 1973 changed how strongly business cycles moved together across countries. We tested a pre/post-1973 interaction (Week 5). It shows no difference, but only 17 recession quarters fall before 1973, so the test has little power.

   - **Correlated quarters and unequal precision.** Quarterly sentiment is not independent across adjacent quarters (for example 2004Q1, 2004Q2 and 2004Q3), because campaigns run for months and the same advertisers buy space repeatedly. The Durbin-Watson statistic is about 0.19 (2 means no correlation). Separately, the number of advertisements per quarter ranges from 141 to 1,124, so quarters built from few advertisements are measured less precisely. Both problems affect the standard errors rather than the coefficients. After comparing OLS, HC3 and HAC errors (Week 5), all quarterly models use Newey-West (HAC) errors with 4 lags, and all ad-level models cluster errors by year.

   - **Generic notices.** We exclude entries flagged with _Brand is Generic_, containing notices not attributed to any brand or promoter. This leaves 172,219 branded ads from 1948 to 2014 (268 quarters). Every derived file keeps these records in the same order (newest first, never sorted); `row_id`, the 0-based row position in `week_three_ads_with_sentiment.csv`, is the join key between all later files.

   - **Vocabulary choices.** "_All words_" (counting every token), "_No Stopwords_" (removing English stopwords) and "_Dictionary Only_" (counting only the words that appear in the dictionary). The denominator falls sharply: from All words to Harvard-only, mean advertisement length drops from 189.6 to 16.8 words. Shares are therefore inflated for purely arithmetic reasons. The mean positive share rises from 2.27% to 23.87% of words for Harvard and from 1.24% to 34.48% for Loughran-McDonald, about 11 and 28 times. The main analysis uses all words.

   - **Scale rule.** Because each variant divides by a different denominator, raw differences are not comparable across variants. Every effect is reported as a percentage of that series' own mean level.

   - **Aggregation.** Quarterly values are formed by averaging the per-advertisement ratios, so a 10-word advertisement counts as much as an 8,062-word one. We compare this against pooling, where category words and total words are summed across the quarter before dividing once. The first asks how positive the typical advertisement is; the second asks how positive advertising language is overall. The two give the same significance verdict in 40 of 40 comparisons (Week 5).

   - **Long-run drift.** Mean Harvard positive sentiment varies from 1.77% of words in the 1970s to 2.93% in 2010–2014, a range roughly forty times the raw recession-expansion difference (about 13 times for Harvard and 4 times for Loughran-McDonald when measured by the standard deviation of decade means). The drift is U-shaped, with its low around 1970. Recession quarters are unevenly distributed against it, with nine in the 1970s and none after 2009. We therefore control for time with a linear trend or decade effects in every model beyond the raw comparison.

**Ad vs Non-Ad Filtering (Weeks 4 to 7):**

   - The corpus contains text that is not advertising: bond announcements, legal notices, company accounts, editorial pages and The Economist's own promotions. A dictionary score measures advertising language only if the record is an ad, so we built and validated a filter in three versions:
     - **v1 (Week 4):** a brand blocklist plus flags for multi-page and very short records. Removes 3,303 records (1.9%).
     - **v2:** the same blocklist matched case-insensitively, fixing a bug where "Subscription Service" matched but "Subscription service" did not. Removes 3,704 records (2.2%).
     - **v3 (Week 7):** v2 plus regular-expression rules in five groups: bond tombstones ("appears as a matter of record only"), legal notices ("notice is hereby given"), financial statements ("consolidated balance sheet"), Economist house content (conferences, Intelligence Unit, subscriptions) and editorial pages. Removes 12,396 records (7.2%). Bank "cards" (a name and branch list with no sales message) are deliberately not flagged, since phrases cannot separate them from real bank ads.

   - **Validation.** The rules were written while reading a 150-record development set, so its scores are optimistic. The fair test is Kiran's 200 hand labels (2012 to 2014; 163 ads, 37 non-ads), never used to write the rules:

     | Filter | Non-ads caught | Real ads removed | Cohen's kappa |
     |---|---|---|---|
     | v1 | 2 of 37 | 0 of 163 | 0.09 |
     | v2 | 13 of 37 | 0 of 163 | 0.47 |
     | v3 | 24 of 37 | 2 of 163 | 0.72 |

     Because the test set covers only 2012 to 2014, a precision audit read 150 randomly flagged records (30 per rule) from every decade: 148 were non-ads (99%).

   - **Where notices sit.** Notices make up 10.7% of 1970s records, the decade where sentiment bottoms out. Whether part of the 1970s low is a notice effect is untested.

   - **Dictionary genre.** Loughran-McDonald was built for company filings, so it scores words like "loss" and "liabilities" in notices and balance sheets. Removing notices weakens the LM negative recession signal (+5.0%, p = .057 under v2, to +3.8%, p = .135 under v3).

**Hand Labelling (Week 7):**

   - A written guide (`week 7 - final adjustments/labelling sheets/`) asks one question: is this a paid message from an outside organisation trying to persuade readers? Ad codes: A1 persuasive ad, A2 recruitment, A3 other outside promotion. Non-ad codes: N1 editorial, N2 Economist house content, N3 financial or legal notice, N4 mixed, N5 unreadable OCR.

   - **Sample.** 150 records, stratified by decade into three strata (records with a non-ad phrase, the longest 15%, and the rest), seed 314, excluding Kiran's 200. Each record carries a weight (stratum population / sample size).

   - **Process.** An AI model (Claude) labelled all 150 with the guide. A group member second-coded 30 blind. First-pass agreement on ad vs non-ad was weak (kappa 0.16), mostly over ads written as prose. The 15 disagreements were adjudicated starting from AI-proposed resolutions, and two rules were added to the guide. This was not a fully independent reliability test. Final labels: 126 ads, 24 non-ads.

**Models and Inference:**

   - **Specification ladder.** (1) raw quarterly comparison; (2) plus a linear trend; (3) plus decade effects; (4) ad level with decade effects; (5) plus advertiser fixed effects, comparing each firm only with its own ads.

   - **Advertiser key.** Brand names are normalised (23,419 brand strings become 23,078 advertisers). Ad-level models keep advertisers with two or more ads and use total words as the denominator.

   - **Full reporting.** Every combination of 4 outcomes, 2 aggregations, 2 recession definitions and 4 filter versions is estimated: 224 estimates, all saved in `model_results_by_filter_version.csv`. Six have p < .05, fewer than the 11 expected by chance. No specification was selected for significance.

   - **Advertiser mix.** Keyword rules assign a sector to 47% of ads. Sector mix explains about 5% of the Loughran-McDonald positive gap within decades.

   - **Macro controls (Week 5).** Models with GDP growth and inflation were also compared. Inflation is annual World Bank data assigned to every quarter of the year.

**Topics and Words (Week 7):**

   - **Structural topic model.** 10 topics on a 20,000-ad random sample (seed 314), with recession and decade as prevalence covariates. Each topic's quarterly share is regressed on recession with decade effects and HAC errors, then Holm-corrected for 10 tests. Topic names are read from top words and have not been checked against the top ads.

   - **LDA.** The same test on LDA topics, as a cross-check; topic stability was checked across random seeds.

   - **Fightin' words.** Recession vs expansion vocabulary is compared within decades. The significance threshold comes from 500 shuffles of recession labels across quarters within each decade, keeping the largest |z| from each shuffle. This guards against single advertisers whose repeated campaign words inflate naive z-scores.

**Data Feasibility:**

   - The master dataset is a 670 MB CSV, large enough to require chunked reading or column selection but manageable on a standard machine. Filtering to post-1948 branded advertisements reduces it substantially.

   - Large derived files (about 250 MB each) are stored with Git LFS. Small output tables are kept as plain text through `.gitattributes` exceptions, so they render on GitHub and the final report notebook runs without LFS.

   - Using NBER dates for a publication with international readership raises questions about external validity. Two considerations mitigate this. All four global recessions identified since 1950 (1975, 1982, 1991 and 2009) fall within NBER-dated U.S. recessions (Kose, Sugawara and Terrones, 2020), so the U.S. dating captures the major worldwide downturns. Business cycles were also less synchronised before 1973 than after (Bordo and Helbling, 2010), which motivated the pre/post-1973 comparison described above.
