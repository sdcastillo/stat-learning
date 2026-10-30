# Statistical learning labs, 2026
#
# Repurposed from the modeling chapters of the Predictive Analytics R Study Manual
# (glm, cv.glmnet, rpart prune, prcomp, kmeans, hclust) and from the ISLR lab.R
# files already in this repository.
#
# What changed for 2026:
# - Read the CSVs in data/. Do not call fix() or attach().
# - The original chapter lab.R files stay as the book transcriptions.
# - Compare each fit with a dumb baseline on a holdout built before the fit.
# - glmnet scales predictors by default. Say so, and keep the seed.
# - A confidence interval is for the mean. A prediction interval is for one row.
#
# Run from the repository root:
#   Rscript labs-2026/stat-learning-2026.R
#
# Packages: glmnet, rpart. Install them if the script stops at library().

set.seed(2026)

if (!requireNamespace("glmnet", quietly = TRUE) || !requireNamespace("rpart", quietly = TRUE)) {
  stop("Install glmnet and rpart, then rerun from the repository root.")
}

library(glmnet)
library(rpart)

auto <- read.csv("data/Auto.csv", na.strings = "?")
auto <- na.omit(auto)
auto$name <- NULL

college <- read.csv("data/College.csv")
# The file has no name for the first column, so read.csv stores the college name in X.
names(college)[1] <- "name"
college$bad_grad_rate <- college$Grad.Rate > 100
cat("Colleges with Grad.Rate above 100:", sum(college$bad_grad_rate), "\n")

# Holdout built before any fit. Same rows for every model below.
n <- nrow(auto)
test_id <- sample(seq_len(n), size = floor(0.2 * n))
train <- auto[-test_id, ]
test <- auto[test_id, ]

baseline <- mean(train$mpg)
baseline_mse <- mean((test$mpg - baseline)^2)
cat("Holdout n:", nrow(test), " baseline (training mean) MSE:", round(baseline_mse, 3), "\n")

# Simple linear regression. The book lab used Boston and attach().
# This is the Auto fit already knitted in ch3/applied.html.
fit_hp <- lm(mpg ~ horsepower, data = train)
cat("\n--- mpg ~ horsepower ---\n")
print(summary(fit_hp)$coefficients)
one_car <- data.frame(horsepower = 98)
print(predict(fit_hp, one_car, interval = "confidence"))
print(predict(fit_hp, one_car, interval = "prediction"))
pred_hp <- predict(fit_hp, test)
cat("Horsepower-only holdout MSE:", round(mean((test$mpg - pred_hp)^2), 3), "\n")

# Logistic regression, same idea as the PA manual's high-claim flag:
# a binary target defined from the continuous outcome, then binomial glm.
train$high_mpg <- as.integer(train$mpg > median(train$mpg))
test$high_mpg <- as.integer(test$mpg > median(train$mpg))
fit_logit <- glm(
  high_mpg ~ horsepower + weight + year,
  family = binomial,
  data = train
)
cat("\n--- logistic: high mpg ---\n")
print(coef(summary(fit_logit)))
prob <- predict(fit_logit, test, type = "response")
class_hat <- as.integer(prob > 0.5)
majority <- as.integer(mean(train$high_mpg) >= 0.5)
base_acc <- mean(test$high_mpg == majority)
model_acc <- mean(test$high_mpg == class_hat)
cat("Majority-class accuracy:", round(base_acc, 3), " model accuracy:", round(model_acc, 3), "\n")

# Ridge and lasso. glmnet standardizes x unless standardize = FALSE.
# Alpha 0 is ridge. Alpha 1 is the lasso. Same split as the linear fit.
x_train <- model.matrix(mpg ~ horsepower + weight + displacement + acceleration + year, train)[, -1]
x_test <- model.matrix(mpg ~ horsepower + weight + displacement + acceleration + year, test)[, -1]
y_train <- train$mpg

cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
ridge_hat <- as.numeric(predict(cv_ridge, x_test, s = "lambda.min"))
lasso_hat <- as.numeric(predict(cv_lasso, x_test, s = "lambda.min"))
cat("\n--- penalized, holdout MSE ---\n")
cat("ridge", round(mean((test$mpg - ridge_hat)^2), 3),
    " lasso", round(mean((test$mpg - lasso_hat)^2), 3),
    " lambda.min ridge", signif(cv_ridge$lambda.min, 4),
    " lasso", signif(cv_lasso$lambda.min, 4), "\n")
cat("Lasso coefficients at lambda.min (zeros are dropped predictors):\n")
print(coef(cv_lasso, s = "lambda.min"))

# Trees. The PA manual grew rpart on charges, read cptable$xerror, then pruned.
# Here the target is mpg. cp is the cost-complexity penalty, the tree analog of lambda.
tree <- rpart(
  mpg ~ horsepower + weight + displacement + year,
  data = train,
  control = rpart.control(cp = 0.001)
)
best_cp <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
pruned <- prune(tree, cp = best_cp)
tree_hat <- predict(pruned, test)
cat("\n--- pruned tree ---\n")
cat("cp at the minimum xerror:", best_cp, " holdout MSE:", round(mean((test$mpg - tree_hat)^2), 3), "\n")
print(tree$cptable)

# PCA on college numerics. Scale first. An unscaled PCA lets the large-unit
# column (Expend, Outstate) write the first component by itself.
college_x <- college[, c("Outstate", "Expend", "Top10perc", "S.F.Ratio", "PhD", "Grad.Rate")]
pca <- prcomp(college_x, scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
cat("\n--- college PCA, variance explained ---\n")
print(round(pve, 3))
print(round(pca$rotation[, 1:2], 3))

# Clustering. Scale, then k-means with several starts, then three linkages.
college_s <- scale(college_x)
km <- kmeans(college_s, centers = 2, nstart = 20)
cat("\nCluster sizes:", km$size, " within SS:", round(sum(km$withinss), 1), "\n")

d <- dist(college_s)
hc_complete <- hclust(d, method = "complete")
hc_average <- hclust(d, method = "average")
hc_single <- hclust(d, method = "single")
cat("Complete vs single cuts at k = 2. Sizes differ because linkage differs.\n")
print(table(complete = cutree(hc_complete, 2), single = cutree(hc_single, 2)))
cat("Average linkage is a third answer, not a tie-break you can skip.\n")
print(table(average = cutree(hc_average, 2)))
