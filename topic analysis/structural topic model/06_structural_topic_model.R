
# ============================================================
# TASK F: STRUCTURAL TOPIC MODEL
# Recessions and advertising topics, 1948-2014
# ============================================================

.libPaths(c(path.expand("~/R/library"), .libPaths()))

library(stm)
library(quanteda)
library(readr)

set.seed(314)

# ------------------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# ------------------------------------------------------------

OUT <- "topic analysis/structural topic model"
CSV_OUT <- file.path(OUT, "csv outputs")
IMG_OUT <- file.path(OUT, "diagnostic images")

dir.create(CSV_OUT, recursive=TRUE, showWarnings=FALSE)
dir.create(IMG_OUT, recursive=TRUE, showWarnings=FALSE)

FLAGS_FILE <- "week 4 - updated quality filtered analysis/Ad_filtering/week_four_quality_flags.csv"
RECESSION_FILE <- "week three data cleaning/nber_recession_table.csv"

K <- 10
SAMPLE_SIZE <- 20000
SAMPLE_SEED <- 314

# Find the canonical advertisement CSV by its required columns.
# If there are several copies, choose the largest file and report it.

files <- list.files(".", pattern="\\.csv$", recursive=TRUE,
                    full.names=TRUE)

files <- files[!grepl("^\\./topic analysis/", files)]

is_ads_file <- function(path) {
  first <- tryCatch(readLines(path, n=1, warn=FALSE),
                    error=function(e) "")
  if (length(first) == 0) return(FALSE)
  all(c("DateOfIssue", "OCR_GoogleVision_original", "Year_Quarter") %in%
        strsplit(first[1], ",", fixed=TRUE)[[1]])
}

candidates <- files[vapply(files, is_ads_file, logical(1))]

if (length(candidates) == 0) {
  stop("Could not locate the original advertisement CSV.")
}

ADS_FILE <- candidates[which.max(file.info(candidates)$size)]

cat("Advertisement file:", ADS_FILE, "\n")

if (length(candidates) > 1) {
  cat("Note: Multiple candidate files found. Using largest.\n")
}

# ------------------------------------------------------------
# 2. LOAD ORIGINAL DATA
# ------------------------------------------------------------

cat("\nLoading advertisements...\n")

ads <- read_csv(
  ADS_FILE,
  col_select=all_of(c(
    "DateOfIssue",
    "OCR_GoogleVision_original",
    "Year_Quarter"
  )),
  show_col_types=FALSE
)

flags <- read_csv(
  FLAGS_FILE,
  col_select=all_of(c(
    "row_id",
    "DateOfIssue",
    "Brand",
    "any_flag"
  )),
  show_col_types=FALSE
)

recession_data <- read_csv(
  RECESSION_FILE,
  col_select=all_of(c(
    "Year_Quarter",
    "Recession_Majority"
  )),
  show_col_types=FALSE
)

cat("Original advertisements:", nrow(ads), "\n")

# ------------------------------------------------------------
# 3. REPRODUCE THE QUALITY FILTER
# ------------------------------------------------------------

# Python's row_id begins at zero.
ads$row_id <- seq_len(nrow(ads)) - 1L

if (nrow(ads) != nrow(flags)) {
  stop("Ads and quality flags have different row counts.")
}

if (!identical(as.integer(flags$row_id), ads$row_id)) {
  stop("Ads and quality flags have different row ordering.")
}

if (!identical(as.character(ads$DateOfIssue),
               as.character(flags$DateOfIssue))) {
  stop("Ads and flags have different dates or row ordering.")
}

blocklist <- c(
  "letters are welcome",
  "subscription service",
  "offer to readers"
)

flagged <- tolower(trimws(as.character(flags$any_flag))) == "true"
blocked <- tolower(trimws(as.character(flags$Brand))) %in% blocklist

exclude <- flagged | blocked
exclude[is.na(exclude)] <- FALSE

corpus <- ads[!exclude, ]

corpus$text <- as.character(corpus$OCR_GoogleVision_original)
corpus$text[is.na(corpus$text)] <- ""

corpus <- corpus[nchar(corpus$text) > 0, ]

cat("Advertisements after quality filtering:", nrow(corpus), "\n")

# ------------------------------------------------------------
# 4. REPRODUCIBLE 20,000-AD SAMPLE
# ------------------------------------------------------------

if (nrow(corpus) < SAMPLE_SIZE) {
  stop("Not enough advertisements for a 20,000-document sample.")
}

set.seed(SAMPLE_SEED)

# Same filtered population as Daniel's LDA.
# R and Python have different random-number implementations.
sample_indices <- sample(seq_len(nrow(corpus)), SAMPLE_SIZE)

data <- corpus[sample_indices, ]
rownames(data) <- NULL

cat("STM sample:", nrow(data), "advertisements\n")

# ------------------------------------------------------------
# 5. ADD RECESSION AND DECADE VARIABLES
# ------------------------------------------------------------

if (anyDuplicated(recession_data$Year_Quarter)) {
  stop("Duplicate quarters in recession table.")
}

quarter_match <- match(
  as.character(data$Year_Quarter),
  as.character(recession_data$Year_Quarter)
)

if (anyNA(quarter_match)) {
  stop("Some advertisements do not match the recession table.")
}

data$recession <- as.integer(
  tolower(trimws(as.character(
    recession_data$Recession_Majority[quarter_match]
  ))) == "true"
)

data$year <- as.integer(
  substr(as.character(data$Year_Quarter), 1, 4)
)

if (anyNA(data$year)) {
  stop("Could not extract years from Year_Quarter.")
}

data$decade <- factor((data$year %/% 10) * 10)

cat("Recession advertisements:", sum(data$recession), "\n")
cat("Expansion advertisements:", sum(data$recession == 0), "\n")

write_csv(
  data[, c("row_id", "Year_Quarter", "year", "decade", "recession", "text")],
  file.path(CSV_OUT, "stm_sample_metadata.csv")
)

# ------------------------------------------------------------
# 6. TEXT PREPROCESSING
# ------------------------------------------------------------

cat("\nTokenizing advertisements...\n")

toks <- tokens(
  data$text,
  remove_punct=TRUE,
  remove_numbers=TRUE,
  remove_symbols=TRUE
)

toks <- tokens_tolower(toks)

extra_stopwords <- c(
  "economist", "www", "com", "http", "https",
  "html", "org", "net", "tel", "fax", "ltd",
  "inc", "plc", "llc", "co", "per", "cent"
)

toks <- tokens_remove(
  toks,
  pattern=c(stopwords("en"), extra_stopwords)
)

dtm <- dfm(toks)

dtm <- dfm_trim(
  dtm,
  min_docfreq=20,
  docfreq_type="count"
)

dtm <- dfm_trim(
  dtm,
  max_docfreq=0.50,
  docfreq_type="prop"
)

# Keep 5,000 frequent terms for comparability with LDA.

top_words <- names(topfeatures(dtm, n=5000))

dtm <- dfm_select(
  dtm,
  pattern=top_words,
  selection="keep",
  valuetype="fixed"
)

cat("Vocabulary size:", nfeat(dtm), "\n")

# ------------------------------------------------------------
# 7. CONVERT TEXT TO STM FORMAT
# ------------------------------------------------------------

stm_input <- convert(dtm, to="stm")

metadata <- data[, c(
  "row_id", "Year_Quarter", "year", "decade", "recession"
)]

# Remove documents with no words remaining after preprocessing.
keep <- ntoken(dtm) > 0

stm_input <- convert(dtm[keep, ], to="stm")
metadata <- metadata[keep, , drop=FALSE]

prepared <- prepDocuments(
  stm_input$documents,
  stm_input$vocab,
  meta=metadata,
  lower.thresh=0
)

stopifnot(length(prepared$documents) == nrow(prepared$meta))

cat("Documents retained:", length(prepared$documents), "\n")

# ------------------------------------------------------------
# 8. FIT STRUCTURAL TOPIC MODEL
# ------------------------------------------------------------

cat("\nFitting STM with 10 topics...\n")

model <- stm(
  documents=prepared$documents,
  vocab=prepared$vocab,
  K=K,
  prevalence=~ recession + decade,
  data=prepared$meta,
  init.type="Spectral",
  seed=314,
  max.em.its=75,
  verbose=TRUE
)

saveRDS(
  model,
  file.path(OUT, "stm_model_10_topics.rds")
)

cat("\nSTM model successfully fitted!\n")

# ------------------------------------------------------------
# 9. EXPORT TOPIC WORDS
# ------------------------------------------------------------

topic_labels <- labelTopics(model, n=10)

topic_table <- data.frame(
  topic=1:K,
  top_words=apply(
    topic_labels$prob,
    1,
    paste,
    collapse=", "
  )
)

write_csv(
  topic_table,
  file.path(CSV_OUT, "stm_topic_words.csv")
)

print(topic_table)

# ------------------------------------------------------------
# 10. ESTIMATE RECESSION EFFECTS
# ------------------------------------------------------------

cat("\nEstimating recession effects...\n")

effects <- estimateEffect(
  1:K ~ recession + decade,
  stmobj=model,
  metadata=prepared$meta,
  uncertainty="Global"
)

saveRDS(
  effects,
  file.path(OUT, "stm_recession_effects.rds")
)

capture.output(
  summary(effects),
  file=file.path(CSV_OUT, "stm_recession_effects_summary.txt")
)

# ------------------------------------------------------------
# 11. GRAPH RECESSION EFFECTS
# ------------------------------------------------------------

pdf(
  file.path(IMG_OUT, "stm_recession_effects.pdf"),
  width=11,
  height=8
)

plot(
  effects,
  covariate="recession",
  topics=1:K,
  method="difference",
  cov.value1=1,
  cov.value2=0,
  model=model,
  xlab="Difference in topic prevalence: recession minus expansion",
  main="Recessions and Advertising Topics"
)

dev.off()

# ------------------------------------------------------------
# 12. EXPORT DOCUMENT TOPIC PREVALENCE
# ------------------------------------------------------------

prevalence <- as.data.frame(model$theta)

names(prevalence) <- paste0("topic_", 1:K)

prevalence$row_id <- prepared$meta$row_id
prevalence$Year_Quarter <- prepared$meta$Year_Quarter
prevalence$recession <- prepared$meta$recession
prevalence$decade <- prepared$meta$decade

write_csv(
  prevalence,
  file.path(CSV_OUT, "stm_document_topic_prevalence.csv")
)

# ------------------------------------------------------------
# 13. FINISHED
# ------------------------------------------------------------

cat("\n========================================\n")
cat("STRUCTURAL TOPIC MODEL COMPLETE\n")
cat("========================================\n")
cat("Topics:", K, "\n")
cat("Documents:", length(prepared$documents), "\n")
cat("Outputs:", OUT, "\n")



# ============================================================
# 14. QUARTER-LEVEL ROBUSTNESS CHECK
# HAC STANDARD ERRORS + HOLM CORRECTION
# ============================================================

library(dplyr)
library(lmtest)
library(sandwich)
library(ggplot2)

cat("\nRunning quarterly robustness checks...\n")

# Use the STM document-topic prevalence already saved.
topic_data <- read_csv(
  file.path(CSV_OUT, "stm_document_topic_prevalence.csv"),
  show_col_types=FALSE
)

# Aggregate advertisements to quarterly mean topic shares.
quarterly <- topic_data %>%
  group_by(Year_Quarter) %>%
  summarise(
    recession=first(recession),
    decade=first(decade),
    n_ads=n(),
    across(starts_with("topic_"), mean, na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(
    year=as.integer(substr(Year_Quarter, 1, 4)),
    quarter=as.integer(sub(".*Q", "", Year_Quarter))
  ) %>%
  arrange(year, quarter)

if (anyNA(quarterly$year) || anyNA(quarterly$quarter)) {
  stop("Could not parse year and quarter from Year_Quarter.")
}

if (any(quarterly$quarter < 1 | quarterly$quarter > 4)) {
  stop("Quarter values must be between 1 and 4.")
}

quarterly$decade <- factor(quarterly$decade)

cat("Quarters analyzed:", nrow(quarterly), "\n")

write_csv(
  quarterly,
  file.path(CSV_OUT, "stm_quarterly_topic_prevalence.csv")
)

# ------------------------------------------------------------
# Run regression for each topic
# Topic prevalence ~ recession + decade
# HAC standard errors: 4 lags
# ------------------------------------------------------------

results <- lapply(1:K, function(k) {

  topic_name <- paste0("topic_", k)

  formula <- reformulate(
    c("recession", "decade"),
    response=topic_name
  )

  fit <- lm(formula, data=quarterly)

  hac <- coeftest(
    fit,
    vcov.=NeweyWest(
      fit,
      lag=4,
      prewhite=FALSE,
      adjust=TRUE
    )
  )

  data.frame(
    topic=k,
    coefficient=hac["recession", "Estimate"],
    std_error=hac["recession", "Std. Error"],
    p_value=hac["recession", "Pr(>|t|)"]
  )
})

results <- bind_rows(results)

# Holm correction for ten simultaneous topic tests.
results$p_holm <- p.adjust(results$p_value, method="holm")
results$significant <- results$p_holm < 0.05

# Express effects in percentage points.
results$effect_pp <- 100 * results$coefficient
results$lower_95_pp <- 100 * (results$coefficient - 1.96 * results$std_error)
results$upper_95_pp <- 100 * (results$coefficient + 1.96 * results$std_error)

# Add top words for interpretation.
results <- results %>%
  left_join(topic_table, by="topic") %>%
  arrange(p_holm)

write_csv(
  results,
  file.path(CSV_OUT, "stm_quarterly_hac_holm_results.csv")
)

print(
  results[, c(
    "topic", "effect_pp", "p_value",
    "p_holm", "significant", "top_words"
  )]
)

# ------------------------------------------------------------
# Create publication-ready figure
# ------------------------------------------------------------

plot_data <- results %>%
  mutate(
    topic_label=paste0("Topic ", topic),
    topic_label=factor(
      topic_label,
      levels=paste0("Topic ", 10:1)
    )
  )

p <- ggplot(
  plot_data,
  aes(x=effect_pp, y=topic_label)
) +
  geom_vline(xintercept=0, linetype="dashed") +
  geom_errorbarh(
    aes(xmin=lower_95_pp, xmax=upper_95_pp),
    height=0.2
  ) +
  geom_point(size=2.7) +
  labs(
    title="Recessions and Advertising Topic Prevalence",
    subtitle="Quarterly regressions with decade controls and HAC standard errors",
    x="Change in topic prevalence (percentage points)",
    y="",
    caption="95% unadjusted confidence intervals; Holm-adjusted p-values in CSV"
  ) +
  theme_minimal(base_size=12)

ggsave(
  file.path(IMG_OUT, "stm_quarterly_hac_holm.png"),
  plot=p,
  width=11,
  height=7,
  dpi=300
)

cat("\n========================================\n")
cat("QUARTERLY ROBUSTNESS CHECK COMPLETE\n")
cat("========================================\n")
cat("Quarters:", nrow(quarterly), "\n")
cat("Topics significant after Holm:", sum(results$significant), "\n")
cat("Results saved to:", CSV_OUT, "\n")
