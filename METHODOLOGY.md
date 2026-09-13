**Data Treatment:**
    
   - The dataset contains individual advertisements with publication dates, advertiser/brand information, and OCR-generated advertisement text. The text was extracted from scanned Economist pages using Google Vision OCR. We use advertisements published from 1948 onward. This comes after the postwar reconversion period, when the U.S. economy had returned to peacetime conditions and NBER recession dates reflect ordinary business cycles rather than wartime mobilisation and demobilisation. Modern quarterly national accounts also begin around this time, which gives us a consistent economic record to match against the advertisement text. This yields 172,219 branded advertisements across 268 quarters, 1948Q1 to 2014Q4.
   
   - Filters as per period of ruling exchange rate and fiat currency, such as the post-Bretton-Woods period (1973), logically have interference in the transmissibility of a cycle; hence, this is an analysis we shall execute.
   
   - Advertisement and sentiment are not independent across sequential quarters (1.g. 2004Q2, 2004Q2-2004Q4), and the quantity of advertisements range from a few hundreds to some thousands. The error term is hence incompatible with OLS homoskedasticity assumptions, and we need robust errors to have an adequate analysis.

   - NBER and FRED-GDP flag recessions differently on quarter basis. For NBER, when real GDP peaks in a month, just the next month is counted as recessionary. For FRED, the peak month is already considered recessionary. Further than that, within a quarter, 0 to 3 months can be recessionary, instead of a homogenous and solely label. These different approaches account for flagging recessionary periods that happened out of the two-consecutive-negative-quarters rule.

**Data Feasibility:** 

   - The Master Dataset is relatively small at about 670 MB and is available as a CSV, making it very feasible to process. It will become smaller after filtering by date and possibly advertisement type. OCR quality must still be considered because older advertisements or things like unusual fonts may lead to errors. 
    
   - We acknowledge that using NBER definitions within the economy of the United States can raise questions about external validity, but we clarify that all US global recessions since 1950 matched NBER recessions, capturing the main downturns in a global context (Gose, Sugawara & Torrones, 2010). 
