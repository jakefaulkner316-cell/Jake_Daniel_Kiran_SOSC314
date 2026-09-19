**Data Treatment:**
    
   - We date recessions using the NBER business cycle chronology rather than the common two-consecutive-negative-quarters rule. The NBER committee weighs employment, real income, industrial production and sales, not GDP alone, and dates turning points by month. The two definitions disagree: the two-quarter rule misses the 2001 and 1960–61 recessions entirely. Converting monthly dates to quarters takes two decisions. The NBER treats the peak as the last month of expansion, so our recessions start the month after the peak and run through the trough. And since a quarter can hold zero to three recession months, we call a quarter recessionary when at least two of its months fall within a contraction. This gives 42 recession quarters; an any-month rule gives 48, and the 6 quarters where they differ serve as a robustness check.
   
   - The collapse of the Bretton Woods system in 1973 changed how strongly business cycles moved together across countries. We therefore plan to compare the pre-1973 and post-1973 periods separately, to check whether our results depend on the degree of international synchronisation. (Week 5)
   
   - Quarterly sentiment is not independent across adjacent quarters (for example 2004Q1, 2004Q2 and 2004Q3), because campaigns run for months and the same advertisers buy space repeatedly. This violates the OLS assumption of independent errors. Separately, the number of advertisements per quarter ranges from 141 to 1,124, so quarters built from few advertisements are measured less precisely than quarters built from many. This violates the assumption of constant error variance. Both problems affect the standard errors rather than the coefficients, so any eventual test will need standard errors robust to autocorrelation and heteroskedasticity.

   - NBER and FRED-GDP flag recessions differently on quarter basis. For NBER, when real GDP peaks in a month, just the next month is counted as recessionary. For FRED, the peak month is already considered recessionary. Further than that, within a quarter, 0 to 3 months can be recessionary, instead of a homogenous and solely label. These different approaches account for flagging recessionary periods that happened out of the two-consecutive-negative-quarters rule.
     
   - We exclude entries flagged with _Brand is Generic_, containing notices not attributed to any brand or promoter.
     
   - Vocabulary choices: "_All words_" (counting every token), "_No Stopwords_" (removing English stopwords with empty meaning), and "_Dictionary Only_" (counting just the words that intersect between the advertisement and the dictionary). The average length of ads fall substantially; e.g., from All words to Harvard-only, the mean advertisment length drops from 189.6 to 16.8. When calculating our _"positiv_ratio"_ metric (positive-bearing words/total words), these numbers fit in as denominators; hence, the final indicator can be scaled multiple times solely for arithmetic reasons.

   - Aggregation: Quarterly values are formed by averaging the per-advertisement ratios, so a 10-word advertisement counts as much as an 8,062-word one. We compare this against pooling, where category words and total words are summed across the quarter before dividing once. The first asks how positive the typical advertisement is; the second asks how positive advertising language is overall.

   - Long-run drift: Mean positive sentiment varies from 0.0177 in the 1970s to 0.0293 in 2010–2014, a range roughly forty times the raw recession-expansion difference. Recession quarters are unevenly distributed against this trend, with nine in the 1970s and none after 2009. Any test of the business cycle hypothesis must therefore control for this long-run drift.

**Data Feasibility:** 

   - The master dataset is a 670 MB CSV, large enough to require chunked reading or column selection but manageable on a standard machine. Filtering to post-1948 branded advertisements reduces it substantially.
    
   - Using NBER dates for a publication with international readership raises questions about external validity. Two considerations mitigate this. All four global recessions identified since 1950 (1975, 1982, 1991 and 2009) fall within NBER-dated U.S. recessions (Kose, Sugawara and Terrones, 2020), so the U.S. dating captures the major worldwide downturns. Business cycles were also less synchronised before 1973 than after (Bordo and Helbling, 2010), which motivates the pre and post-1973 comparison described above.
