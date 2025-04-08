# Predicting-Cardiovacular-Disease
# Load necessary libraries
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(ggplot2)

# Load the dataset
data <- read.csv("~/560 Data Mining/Final Exam/health_data.csv")
head(data)
# Remove ID column
data$id <- NULL
# Install and load smotefamily package
#install.packages("smotefamily")

# Convert target variable to factor
data$cardio <- as.factor(data$cardio)

# Separate features and target variable
X <- data[, !(names(data) %in% c("cardio"))]  # Exclude target column
target <- data$cardio  # Extract target variable

library(smotefamily)
# Apply SMOTE
set.seed(123)
smote_result <- SMOTE(X = X, target = target, K = 5, dup_size = 1)

# Convert back to a data frame
balanced_data <- smote_result$data
colnames(balanced_data)[ncol(balanced_data)] <- "cardio"

# Ensure target variable is a factor
balanced_data$cardio <- as.factor(balanced_data$cardio)

# Check class distribution
table(balanced_data$cardio)
# B. Create two different dataset sizes 
small_size <- balanced_data[sample(nrow(balanced_data), 500), ]
large_size <- balanced_data[sample(nrow(balanced_data), 1000), ]

# Split into training (70%) and test (30%)
set.seed(123)
trainIndex_small <- createDataPartition(small_size$cardio, p = 0.7, list = FALSE)
train_small <- small_size[trainIndex_small, ]
test_small <- small_size[-trainIndex_small, ]

trainIndex_large <- createDataPartition(large_size$cardio, p = 0.7, list = FALSE)
train_large <- large_size[trainIndex_large, ]
test_large <- large_size[-trainIndex_large, ]

# Train two classification models: Decision Tree and Random Forest
dt_model_small <- rpart(cardio ~ ., data = train_small, method = "class")
dt_model_large <- rpart(cardio ~ ., data = train_large, method = "class")

rf_model_small <- randomForest(cardio ~ ., data = train_small, importance = TRUE, ntree = 100)
rf_model_large <- randomForest(cardio ~ ., data = train_large, importance = TRUE, ntree = 100)

# Predictions and accuracy
pred_dt_small <- predict(dt_model_small, test_small, type = "class")
pred_dt_large <- predict(dt_model_large, test_large, type = "class")

pred_rf_small <- predict(rf_model_small, test_small, type = "response")
pred_rf_large <- predict(rf_model_large, test_large, type = "response")

# Ensure predictions are factors
pred_rf_small <- as.factor(pred_rf_small)
pred_rf_large <- as.factor(pred_rf_large)

# Evaluate models
confusionMatrix(pred_dt_small, test_small$cardio)
confusionMatrix(pred_dt_large, test_large$cardio)

confusionMatrix(pred_rf_small, test_small$cardio)
confusionMatrix(pred_rf_large, test_large$cardio)
