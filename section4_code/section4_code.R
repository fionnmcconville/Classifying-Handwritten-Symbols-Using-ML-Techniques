library(caret)
library(tree)
library(rpart)
library(randomForest) # Gave a warning saying not writable, used "personal library"
library(class)

column_names1 <- c('label','index', 'nr_pix', 'height', 'width', 'tallness',
                   'rows_with_1','cols_with_1', 'rows_with_5_plus', 'cols_with_5_plus', 
                   'one_neighbour', 'three_or_more_neighbours', 'no_neighbours_below', 
                   'no_neighbours_above', 'no_neighbours_before', 'no_neighbours_after', 
                   'nr_regions', 'nr_eyes', 'r5_c5', 'extra', 'extra1', 'extra2')
#trainfeatures <- read.csv("C:\\Users\\fionn\\OneDrive\\Documents\\Assignment2_40156103\\features\\trainingfeatures.csv", header = FALSE, col.names = column_names1)
trainfeatures <- read.csv("../features/trainingfeatures.csv", header = FALSE, col.names = column_names1)
drops <- c("index") # Drop unnecessary columns
trainfeatures <- trainfeatures[ , !(names(trainfeatures) %in% drops)]


trainfeatures$label <- as.factor(trainfeatures$label)
trainfeatures <- trainfeatures[,1:18 ] # Getting rid of extra cols
testingfeatures <- read.csv("../features/testingfeatures.csv", header = FALSE, col.names = column_names1)
testingfeatures <- testingfeatures[,3:19 ]
str(testingfeatures) # label and index columns are basically useless

tc <- trainControl("cv",number=5, savePredictions = TRUE)
Grid <- expand.grid(mtry = c(4)) # mtry is number of predictors considered at each node

set.seed(3060)
ranforest <- train(label~., data = trainfeatures, 
                      method = 'rf',
                      tuneGrid = Grid, 
                      trControl = tc,
                      ntree = 235,
                      metric = "Accuracy")

ranforest # final prediction model


finalpredict <- predict(ranforest, newdata = testingfeatures)
#reloading the testingfetures so I can use index as rownames in final prediction output
indexes <-  sprintf("%03.0f", 0:2939)
indexes
finalpredict <- data.frame(finalpredict)
finalpredict
rownames(finalpredict) <- indexes
write.csv(finalpredict, file = "../40156103_section4_predictions.csv")

