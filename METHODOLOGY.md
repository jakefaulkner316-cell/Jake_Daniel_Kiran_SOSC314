**Data Treatment:**
    
   - The dataset contains individual advertisements with publication dates, advertiser/brand information, and OCR-generated advertisement text. The text was extracted from scanned Economist pages using Google Vision OCR. We use advertisements published from 1948 onward. This comes after the postwar reconversion period, when the U.S. economy had returned to peacetime conditions and NBER recession dates reflect ordinary business cycles rather than wartime mobilisation and demobilisation. Modern quarterly national accounts also begin around this time, which gives us a consistent economic record to match against the advertisement text. This yields 172,219 branded advertisements across 268 quarters, 1948Q1 to 2014Q4.
   
   - The collapse of the Bretton Woods system in 1973 changed how strongly business cycles moved together across countries. We therefore plan to compare the pre-1973 and post-1973 periods separately, to check whether our results depend on the degree of international synchronisation. (Week 4, tentatively)
   
   - Quarterly sentiment is not independent across adjacent quarters (for example 2004Q1, 2004Q2 and 2004Q3), because campaigns run for months and the same advertisers buy space repeatedly. This violates the OLS assumption of independent errors. Separately, the number of advertisements per quarter ranges from 141 to 1,124, so quarters built from few advertisements are measured less precisely than quarters built from many. This violates the assumption of constant error variance. Both problems affect the standard errors rather than the coefficients, so any eventual test will need standard errors robust to autocorrelation and heteroskedasticity.

   - NBER and FRED-GDP flag recessions differently on quarter basis. For NBER, when real GDP peaks in a month, just the next month is counted as recessionary. For FRED, the peak month is already considered recessionary. Further than that, within a quarter, 0 to 3 months can be recessionary, instead of a homogenous and solely label. These different approaches account for flagging recessionary periods that happened out of the two-consecutive-negative-quarters rule.

**Data Feasibility:** 

   - The master dataset is a 670 MB CSV, large enough to require chunked reading or column selection but manageable on a standard machine. Filtering to post-1948 branded advertisements reduces it substantially.
    
   - We acknowledge that using NBER definitions within the economy of the United States can raise questions about external validity, but we clarify that all US global recessions since 1950 matched NBER recessions, capturing the main downturns in a global context (Gose, Sugawara & Torrones, 2010). 
