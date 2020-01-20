# # Section 2.1
# ~~~~~~~~~~~~
library(class)
library(caret)
column_names1 <- c('label','index', 'nr_pix', 'height', 'width', 'tallness',
                   'rows_with_1','cols_with_1', 'rows_with_5_plus', 'cols_with_5_plus', 
                   'one_neighbour', 'three_or_more_neighbours', 'no_neighbours_below', 
                   'no_neighbours_above', 'no_neighbours_before', 'no_neighbours_after', 
                   'nr_regions', 'nr_eyes', 'r5_c5', 'extra', 'extra1', 'extra2')
trainfeatures <- read.csv("../features/trainingfeatures.csv", header = FALSE, col.names = column_names1)

trainfeatures <- trainfeatures[,1:19 ] # Don't need 3 custom features columns

#Add label's more descriptive values as well as adding appropriate category for each symbol
#8820 training items in total, so 2940 total observations for each Category
trainfeatures["Category"] <- NA
trainfeatures$Category[1:2940] <- "Digits"
trainfeatures$Category[2941:5880] <- "Letters"
trainfeatures$Category[5881:8820] <- "Math"
trainfeatures$Category <- as.factor(trainfeatures$Category)


trainfeatures$label[trainfeatures$label==11] <- '1'
trainfeatures$label[trainfeatures$label==12] <- '2'
trainfeatures$label[trainfeatures$label==13] <- '3'
trainfeatures$label[trainfeatures$label==14] <- '4'
trainfeatures$label[trainfeatures$label==15] <- '5'
trainfeatures$label[trainfeatures$label==16] <- '6'
trainfeatures$label[trainfeatures$label==17] <- '7'

trainfeatures$label[trainfeatures$label==21] <- 'a'
trainfeatures$label[trainfeatures$label==22] <- 'b'
trainfeatures$label[trainfeatures$label==23] <- 'c'
trainfeatures$label[trainfeatures$label==24] <- 'd'
trainfeatures$label[trainfeatures$label==25] <- 'e'
trainfeatures$label[trainfeatures$label==26] <- 'f'
trainfeatures$label[trainfeatures$label==27] <- 'g'

trainfeatures$label[trainfeatures$label==31] <- '<'
trainfeatures$label[trainfeatures$label==32] <- '>'
trainfeatures$label[trainfeatures$label==33] <- '='
trainfeatures$label[trainfeatures$label==34] <- '<='
trainfeatures$label[trainfeatures$label==35] <- '>='
trainfeatures$label[trainfeatures$label==36] <- 'not='
trainfeatures$label[trainfeatures$label==37] <- 'approx'

trainfeatures$label <- as.factor(trainfeatures$label)
str(trainfeatures)
set.seed(3060)
# #mix up the rows in my dataset so I can get reliable training and test datasets within the KNN testing
# rp <- runif(nrow(trainfeatures))
# trainfeatures <- trainfeatures[order(rp),]
# head(trainfeatures)
# 
# summary(trainfeatures)
# 
# #Now we'll have to standardise the feature data so it can work with our KNN classifier
# #We'll only do this on the first ten features because it's all we require
# 
# standardised.X <- scale(trainfeatures[,3:12])
# str(standardised.X)
# #standardised.X is the standardised feature data for the first 10 features
# 
# #Going to have a test set of 10% of the overall data which is 8820/10 = 882
# testlength <- 1:882
# training.X = standardised.X[-testlength,] # note the negative indexing
# testing.X = standardised.X[testlength,]
# 
# #summary(trainfeatures$label)
# #trainfeatures$label # response
# training.Y = trainfeatures$label[-testlength]
# testing.Y = trainfeatures$label[testlength]
# 
# set.seed(3060)
# ks <- c(1,3,5,7,9,11,15,19,23,31)
# accuracies = c()
# for(k in ks){
#   
# print(k)
# knn.pred <- knn(training.X, testing.X, training.Y, k = k)
# #table(knn.pred, testing.Y) #Confusion matrix for each value of K
# accuracies <- cbind(accuracies, mean(testing.Y==knn.pred)) #mean(testing.Y==knn.pred) is the percent of right calculations
# 
# }
# colnames(accuracies) <- ks
# accuracies # Vector containing the accuracy of KNN classifier for each value of k on feature data

#The above code was hard to work into a cross-validation so I abandoned the idea of it

#Section 2.1
#~~~~~~~~~~

vars <- paste(names(trainfeatures)[3:12],sep="")
tenfeatures <- paste("label ~", paste(vars, collapse="+"))
tenfeatures <- as.formula(tenfeatures)
tenfeatures
ks <- c(1,3,5,7,9,11,15,19,23,31)

set.seed(3060)
knnFit <- train(tenfeatures, data = trainfeatures,
                method = "knn", preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = ks))
#data is scaled and centered within the train function

knnFit #Output of kNN fit

knnFit$results[, 1:2] # Just K values and accuracy
#accuracies
#Above are results of accuracies of my original manual KNN classifier, results are similar so I have confidence in the caret method of KNN

# # Section 2.2
# # ~~~~~~~~~~~~

set.seed(3060)
ctrl <- trainControl(method="cv",number = 5, savePredictions = TRUE) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit1 <- train(tenfeatures, data = trainfeatures, method = "knn", preProcess = c("center","scale"), 
                 tuneGrid = expand.grid(k = ks), trControl = ctrl)

knnFit1 #Output of kNN fit

knnFit1$results[,1:2] #Just K values and accuracies

knnpredict1 <- predict(knnFit1, newdata = trainfeatures)
knnpredict1

onedivk <- 1/ks
onedivk

err_rate <- 1- knnFit$results['Accuracy']
err_rate["Cross_Validated"] <- "Classification Error Rate"
err_rate <- cbind(err_rate, onedivk)
err_rate
err_rate1 <- 1- knnFit1$results['Accuracy']
err_rate1["Cross_Validated"] <- "5-fold Crossvalidated Classification Error Rate"
err_rate1 <- cbind(err_rate1, onedivk)
err_rate1

error_rate <- rbind(err_rate, err_rate1)
error_rate 


p<-ggplot(error_rate, aes(x=onedivk, y=Accuracy, group=Cross_Validated)) +
  geom_line(aes(color=Cross_Validated))+
  geom_point(aes(color=Cross_Validated)) +
  labs(title="Error rate of models",x="1/K", y = "Error Rate", fill = "")+
  expand_limits(x = 0, y = 0.3)+
  theme(legend.position = c(0.7, 0.2))
p
#ggsave("Section2.2errorrategraph.png")

# Section 2.3
# ~~~~~~~~~~~~

matrix1 <- confusionMatrix(knnpredict1, trainfeatures$label)
matrix1 # Contains accuracy on the testing data
confusionmatrix1 <- matrix1$table
confusionmatrix1 # Confusion Matrix for 5-fold CV.

# Only need digits confusion matrix
digits = c('1','2','3','4','5','6','7')
confusionmatrix1[digits,digits]
#5 and 3 seem to get confused the most