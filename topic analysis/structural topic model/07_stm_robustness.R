
# ============================================================
# TASK F: STM QUARTERLY ROBUSTNESS CHECK
# HAC standard errors and Holm correction
# Does NOT rerun the STM model
# ============================================================

.libPaths(c(path.expand("~/R/library"), .libPaths()))

library(readr)
library(dplyr)
library(lmtest)
library(sandwich)
library(ggplot2)

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

CSV_OUT <- "topic analysis/structural topic model/csv outputs"
IMG_OUT <- "topic analysis/structural topic model/diagnostic images"

dir.create(CSV_OUT, recursive=TRUE, showWarnings=FALSE)
dir.create(IMG_OUT, recursive=TRUE, showWarnings=FALSE)

K <- 10

# ------------------------------------------------------------
# 2. LOAD EXISTING STM RESULTS
# ------------------------------------------------------------

cat("Loading saved STM results...\n")

topic_data <- read_csv(
  file.path(CSV_OUT, "stm_document_topic_prevalence.csv"),
  show_col_types=FALSE
)

topic_table <- read_csv(
  file.path(CSV_OUT, "stm_topic_words.csv"),
  show_col_types=FALSE
)

# ------------------------------------------------------------
# 3. AGGREGATE ADVERTISEMENTS BY QUARTER
# ------------------------------------------------------------

quarterly <- topic_data %>%
  group_by(Year_Quarter) %>%
  summarise(
    recession=first(recession),
    decade=first(decade),
    n_ads=n(),
    across(all_of(paste0("topic_", 1:K)), mean, na.rm=TRUE),
    .groups="drop"
  ) %>%
  mutate(
    year=as.integer(substr(Year_Quarter, 1, 4)),
    quarter=as.integer(sub(".*Q", "", Year_Quarter))
  ) %>%
  arrange(year, quarter)

if (anyNA(quarterly$year) || anyNA(quarterly$quarter)) {
  stop("Could not parse Year_Quarter.")
}

if (any(quarterly$quarter < 1 | quarterly$quarter > 4)) {
  stop("Quarter must be between 1 and 4.")
}

quarterly$decade <- factor(quarterly$decade)

cat("Quarters analyzed:", nrow(quarterly), "\n")

write_csv(
  quarterly,
  file.path(CSV_OUT, "stm_quarterly_topic_prevalence.csv")
)

# ------------------------------------------------------------
# 4. RUN REGRESSIONS WITH HAC STANDARD ERRORS
# ------------------------------------------------------------

results <- lapply(1:K, function(k) {

  topic_name <- paste0("topic_", k)

  fit <- lm(
    reformulate(c("recession", "decade"), response=topic_name),
    data=quarterly
  )

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

# ------------------------------------------------------------
# 5. HOLM CORRECTION AND EFFECT SIZES
# ------------------------------------------------------------

results$p_holm <- p.adjust(results$p_value, method="holm")
results$significant <- results$p_holm < 0.05

results$effect_pp <- results$coefficient * 100
results$lower_95_pp <- (results$coefficient - 1.96 * results$std_error) * 100
results$upper_95_pp <- (results$coefficient + 1.96 * results$std_error) * 100

results <- results %>%
  left_join(topic_table, by="topic") %>%
  arrange(p_holm)

write_csv(
  results,
  file.path(CSV_OUT, "stm_quarterly_hac_holm_results.csv")
)

cat("\nROBUSTNESS RESULTS:\n")

print(
  as_tibble(results[, c(
    "topic", "effect_pp", "p_value",
    "p_holm", "significant", "top_words"
  )]),
  n=10,
  width=Inf
)
# ------------------------------------------------------------
# 6. CREATE FIGURE
# ------------------------------------------------------------

plot_data <- results %>%
  mutate(
    topic_label=factor(
      paste0("Topic ", topic),
      levels=paste0("Topic ", 10:1)
    )
  )

p <- ggplot(
  plot_data,
  aes(x=effect_pp, y=topic_label)
) +
  geom_vline(xintercept=0, linetype="dashed") +
  geom_errorbar(
    aes(xmin=lower_95_pp, xmax=upper_95_pp),
    orientation="y",
    width=0.2
  ) +
  geom_point(size=2.7) +
  labs(
    title="Recessions and Advertising Topics",
    subtitle="Quarterly regression with decade controls and HAC standard errors",
    x="Recession effect (percentage points)",
    y="",
    caption="95% unadjusted confidence intervals. Holm-adjusted p-values in CSV."
  ) +
  theme_minimal(base_size=12)

ggsave(
  file.path(IMG_OUT, "stm_quarterly_hac_holm.png"),
  plot=p,
  width=11,
  height=7,
  dpi=300
)

# ------------------------------------------------------------
# 7. FINISHED
# ------------------------------------------------------------

cat("\n========================================\n")
cat("ROBUSTNESS CHECK COMPLETE\n")
cat("========================================\n")
cat("Quarters:", nrow(quarterly), "\n")
cat("Topics significant after Holm:", sum(results$significant), "\n")
cat("Results saved to:", CSV_OUT, "\n")
