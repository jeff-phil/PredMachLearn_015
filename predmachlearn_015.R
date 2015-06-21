# Classe sitting-down, standing-up, standing, walking, and sitting

library(caret)

# Do things in parallel
#install.packages("doParallel")
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

setwd("/Users/jphilli/Documents/coursera/datasciencecoursera/predmachlearn-015")

# Save a bit of time downloading data and loading and filtering by checking if
# already have the data
if (!exists("training") || !exists("testing")) {

        if (!file.exists("pml-training.csv") || !file.exists("pml-testing.csv")) {
                download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                              destfile = "pml-training.csv", method ="curl")
                download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                              destfile = "pml-testing.csv", method ="curl")
        }

        # Load data from CSV, and remove first 7 columns that are more book-
        # keeping info. 
        training <- read.csv("pml-training.csv", 
                             na.strings = c("NA", "", " "))[,-(0:7)]
        testing <- read.csv("pml-testing.csv", 
                            na.strings = c("NA", "", " "))[,-(0:7)]
        print("Training dimensions of the data:")
        print(dim(training))
        
        
        # Remove columns that are heavy on NA's
        hasData<-apply(!is.na(training),2,sum)>5000
        training<-training[, hasData]
        testing<-testing[, hasData]
        
        # Remove columns with low variance
        nzv <- nearZeroVar(training, allowParallel = TRUE)
        training[, -nzv]
        testing[, -nzv]
        
        print("After cleaning up training data the resulting dimensions:")
        print(dim(training))
        print(levels(training$classe))
        qplot(classe, data = training, fill = classe)
}

set.seed(22271)

# Partition data into train and validate train: 60%, validate: 40%
inTrain <- createDataPartition(y = training$classe, p = 0.6, list = FALSE)
train <- training[inTrain, ]
validate <- training[-inTrain, ]

# Check accuracy using rpart - Note: was not too good at 48%, so commented
# modFit <- train(classe ~ ., data=train, method="rpart")
# pred <- predict(modFit, newdata=validate)
# print(paste("Accuracy rpart:", 
#             confusionMatrix(validate$classe, pred)$overall[1]))

# Use Cross-Validation to help combat overfitting of random forest
trainCtl <- trainControl(method = "cv", number = 10,  allowParallel = TRUE)
modFit <- train(classe ~ ., data=train, method = "rf", ntree = 25, 
                trainControl = trainCtl, importance = TRUE, prox=FALSE, 
                allowParallel = TRUE)

# Check out our accuracy for the model using confusionMatrix
pred <- predict(modFit, newdata = validate)
results <- confusionMatrix(validate$classe, pred)
print(paste("Accuracy Random Forest:", results$overall[1]))
print("Results")
print(results$table)
varImpPlot(modFit$finalModel, n.var = 27, main = "Top 27 Predictor Importance Measured by Random Forest")
plot(modFit$finalModel, main = "Error Rate by Number of Trees for Model Fit")
modFit$finalModel.legend <- if (is.null(modFit$finalModel$test$err.rate)) {
                colnames(modFit$finalModel$err.rate)
        } else {
                colnames(modFit$finalModel$test$err.rate)
        }
legend("top", cex =0.7, legend=modFit$finalModel.legend, 
       lty=c(1:6), col=c(1:6), horiz=T)

# Stop the parallel cluster
stopCluster(cl)
