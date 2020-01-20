# Section 3.1
# ~~~~~~~~~~~~

#install.packages("tree", "randomforest", "caret", "rpart", "class")
#install.packages("randomforest")
library(caret)
library(tree)
library(rpart)
library(randomForest)
library(class)
column_names1 <- c('label','index', 'nr_pix', 'height', 'width', 'tallness',
                   'rows_with_1','cols_with_1', 'rows_with_5_plus', 'cols_with_5_plus', 
                   'one_neighbour', 'three_or_more_neighbours', 'no_neighbours_below', 
                   'no_neighbours_above', 'no_neighbours_before', 'no_neighbours_after', 
                   'nr_regions', 'nr_eyes', 'r5_c5', 'extra', 'extra1', 'extra2')

trainfeatures <- read.csv("../features/trainingfeatures.csv", header = FALSE, col.names = column_names1)
trainfeatures
trainfeatures <- trainfeatures[,1:19 ] # Don't need 3 custom features columns

#Add label's more descriptive values as well as adding appropriate category for each symbol
#8820 training items in total, so 2940 total observations for each Category - already checked that they're in order
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
str(trainfeatures$label)

trainfeatures
drops <- c("index", "Category") # Drop unnecessary columns
trainfeatures <- trainfeatures[ , !(names(trainfeatures) %in% drops)]
trainfeatures

nbags <- c(25, 50, 200, 400)
ovrlist <- vector()
overacc <- vector()
# 5-fold Cross validated Accuracy
for (numbags in nbags){
  set.seed(3060)
  tc <- trainControl("cv", number = 5)
  fit.treebag <- train(label~., data = trainfeatures, trControl = tc,
                       method = 'treebag',nbagg = numbags)
  fit.treebag
  cvacc <- fit.treebag$results['Accuracy']
  overacc <- rbind(overacc, cvacc)
}
overacc

# Out of bag error
overacc1 <- vector()
for (numbags in nbags){
  set.seed(3060)
  tc1 <- trainControl("oob")
  fit.treebag <- train(label~., data = trainfeatures, trControl = tc1,
                       method = 'treebag', keepX = TRUE, coob = TRUE, nbagg = numbags)
  fit.treebag
  oobacc <- fit.treebag$results['Accuracy']
  overacc1 <- rbind(overacc1, oobacc)
}
overacc1
ovrlist <- cbind(nbags, overacc1, overacc)
names(ovrlist) <- c("No. Bags", "OOB Accuracy", "CV Accuracy")
ovrlist

# Section 3.2
# ~~~~~~~~~~~~
set.seed(3060)
tc <- trainControl("cv",number=5, savePredictions = TRUE,search = "grid")
Grid <- expand.grid(mtry = c(2,4,6,8)) # mtry is number of predictors considered at each node
modellist <- list()

#caret does not allow tuning of ntrees in tuning grid, will have to do with for loop
for (ntree in seq(25, 400,5)) {
  set.seed(3060)
  fit.treebag1 <- train(label~., data = trainfeatures, 
                        method = 'rf',
                        tuneGrid = Grid, 
                        trControl = tc,
                        ntree = ntree,
                        metric = "Accuracy")
  key <- toString(ntree)
  modellist[[key]] <- fit.treebag1
}
#The above for loop takes a good while to run ( at least 1 hour)

modellist[[1]] # An accumulated list of each combination of mtry and ntree

# forty <- modellist[['40']]
# forty$results[,1:2] # mtry and accuracy
# acc <- forty$results[,'Accuracy'] # Just accuracy
# acc
# max(acc)
# a <- which.max(acc)
# forty$results[,'mtry'][a] # Just mtry

permmax <- 0
permmtry <- 0
permntree <- 0
alllist <- vector()
for (i in 1:length(modellist)) {
  tempntree <- names(modellist)[i]
  acc <- modellist[[i]]$results[,'Accuracy']
  tempmax <- max(acc)
  index <- which.max(acc)
  tempmtry <- modellist[[i]]$results[,'mtry'][index]
  
  if(tempmax > permmax){
    permmax <- tempmax
    permmtry <- tempmtry
    permntree <- tempntree
  }
  templist <- modellist[[i]]$results[,1:2] # list containing all mtry and corresponding accuracy results for this iteration of i
  ntrees <- c(tempntree, tempntree, tempntree, tempntree)
  templist <- cbind(ntrees, templist)
  alllist <- rbind(templist, alllist)
}

alllist
#Accuracy:
print(permmax)
#Number of Preds at each node:
print(permmtry)
#Number of trees:
print(permntree)

optimumvals <- c(permntree, permmtry, permmax)
names(optimumvals) <- c("No. Trees", "No. preds at each node","Accuracy")
optimumvals


modellist$`235`$results
modellist$`315`$results

names(alllist) <- c("ntrees", "Preds_at_each_node","Accuracy")
alllist$Preds_at_each_node <- as.factor(alllist$Preds_at_each_node)
p<-ggplot(alllist, aes(x=ntrees, y=Accuracy, group=Preds_at_each_node)) +
  geom_line(aes(color=Preds_at_each_node))+
  geom_point(aes(color=Preds_at_each_node)) +
  labs(title="Accuracy of 5-fold CV random forest with varying no. preds and no. trees",x="Number of trees", y = "Accuracy") +
  #guides(fill =guide_legend(title = "No. of Predictors \nat each node")) +
  theme(axis.text.x = element_text(angle = 90))
p
#ggsave("Section3.2accuracy.png")



