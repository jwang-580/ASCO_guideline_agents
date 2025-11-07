# Load libraries
library(readxl)
library(dplyr)

df <- read_excel("test.xlsx")

# Convert YES/NO to binary (1/0)
df <- df %>%
  mutate(across(everything(), ~ ifelse(. == "YES", 1, 0)))

# Define your model (the multi-agent system)
my_col <- "multi_agent"

# Define comparators
comparators <- c("ASCO_guidelines_assistant",
                 "claude_3_7",
                 "gpt_4o",
                 "gemini_2_5",
                 "deepseek_R1")

# Function to run McNemar's test
run_mcnemar <- function(col1, col2, name2) {
  tbl <- table(df[[col1]], df[[col2]])
  if (all(dim(tbl) == c(2,2))) {
    test <- mcnemar.test(tbl, correct = TRUE)
    cat("\n=============================\n")
    cat("Comparison:", col1, "vs", name2, "\n")
    print(tbl)
    cat("Chi-squared:", round(test$statistic, 3),
        "  p-value:", signif(test$p.value, 3), "\n")
  } else {
    cat("\nInvalid 2x2 table for", name2, "\n")
  }
}

# Run McNemar’s test for all comparisons
for (comp in comparators) {
  run_mcnemar(my_col, comp, comp)
}
