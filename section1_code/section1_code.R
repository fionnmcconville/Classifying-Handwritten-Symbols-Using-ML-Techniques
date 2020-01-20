

column_names = c('label','index', 'nr_pix', 'height', 'width', 'tallness',
                 'rows_with_1','cols_with_1', 'rows_with_5_plus', 'cols_with_5_plus', 
                 'one_neighbour', 'three_or_more_neighbours', 'no_neighbours_below', 
                 'no_neighbours_above', 'no_neighbours_before', 'no_neighbours_after', 
                 'nr_regions', 'nr_eyes', 'r5_c5')
column_names

features <- read.csv("../features/40156103_features.csv", header = FALSE, col.names = column_names)

features["Category"] <- NA

features$Category[1:56] <- "Digits"
features$Category[57:112] <- "Letters"
features$Category[113:168] <- "Math"
features$Category <- as.factor(features$Category)

features$label[features$label==11] <- '1'
features$label[features$label==12] <- '2'
features$label[features$label==13] <- '3'
features$label[features$label==14] <- '4'
features$label[features$label==15] <- '5'
features$label[features$label==16] <- '6'
features$label[features$label==17] <- '7'

features$label[features$label==21] <- 'a'
features$label[features$label==22] <- 'b'
features$label[features$label==23] <- 'c'
features$label[features$label==24] <- 'd'
features$label[features$label==25] <- 'e'
features$label[features$label==26] <- 'f'
features$label[features$label==27] <- 'g'

features$label[features$label==31] <- '<'
features$label[features$label==32] <- '>'
features$label[features$label==33] <- '='
features$label[features$label==34] <- '<='
features$label[features$label==35] <- '>='
features$label[features$label==36] <- 'not='
features$label[features$label==37] <- 'approx'

features$label <- as.factor(features$label)

# Section 1.1
# ~~~~~~~~~~~

lettersdigitsdf <- features[1:112,]

# According to my statistical analysis in section 3.8 in assignment 1 the most significant feature is no_neighbours_after
library(ggplot2)
library(caret)
plt <- ggplot(lettersdigitsdf, aes(x=no_neighbours_after, fill=as.factor(Category))) +
  geom_histogram(binwidth = 2, alpha = .5, position = "identity") +
  labs(title="Histogram of no_neighbours_after feature between Digits and Letters",
       x="Value of no_neighbours_after feature", y = "Count", fill = "Category")
  
  
plt
ggsave("Histogram of no_neighbours_after feature between Digits and Letters.png")

lettersdigitsdf$dummy.category <- 0
lettersdigitsdf$dummy.category[lettersdigitsdf$Category == 'Letters'] <- 1
lettersdigitsdf$dummy.category

glmfit<-glm(dummy.category ~ no_neighbours_after, data = lettersdigitsdf, family = 'binomial')
summary(glmfit)


#plotting a fitted curve and lettersdigitsdf data
x.range = range(lettersdigitsdf[["no_neighbours_after"]])
x.range

x.values = seq(x.range[1],x.range[2],length.out=1000)
x.values

fitted.curve <- data.frame(no_neighbours_after = x.values)
fitted.curve[["dummy.category"]] = predict(glmfit, fitted.curve, type="response")

# Plot the training data and the fitted  curve:
plt <-ggplot(lettersdigitsdf, aes(x=no_neighbours_after, y=dummy.category)) + 
  geom_point(aes(colour = factor(dummy.category)), 
             show.legend = T, position="dodge")+
  geom_line(data=fitted.curve, colour="orange", size=1) +
  labs(title="Logistic regression with no_neighbours_after feature",
       x="Amount of no_neighbours_after", y = "Probability Class is Letter", color = "Category") +
  scale_color_manual(labels = c("Digits", "Letters"), values = c("blue", "red"))
  

plt
#ggsave('letters_digits_fitted_curve.png',scale=0.7,dpi=400) 

# The curve here shows the probability that a particular value of no_neighbours_after is in the "letters" category,
# i.e according to the graph above; the probability that a symbol has 3 no_neighbours after and is a letter is less than 0.25

# Assuming a p>0.5 cut-off, calculate accuracy on the overall data

x.values = lettersdigitsdf[["no_neighbours_after"]]

lettersdigitsdf[["predicted_val"]] = predict(glmfit, lettersdigitsdf, type="response")
lettersdigitsdf[["predicted_class"]] = 0
lettersdigitsdf[["predicted_class"]][lettersdigitsdf[["predicted_val"]] > 0.5] = 1

correct_items = lettersdigitsdf[["predicted_class"]] == lettersdigitsdf[["dummy.category"]] 

# proportion correct:
nrow(lettersdigitsdf[correct_items,])/nrow(lettersdigitsdf)
# Classifier is approx 70% accurate (0.6964286)

# proportion incorrect:
nrow(lettersdigitsdf[!correct_items,])/nrow(lettersdigitsdf)

#confusionMatrix(data=as.factor(lettersdigitsdf[["predicted_class"]]), as.factor(lettersdigitsdf$dummy.category))

# Section 1.2
# ~~~~~~~~~~~

lettersdigitsdf <- features[1:112,]

lettersdigitsdf$dummy.category <- 0
lettersdigitsdf$dummy.category[lettersdigitsdf$Category == 'Letters'] <- 1

str(lettersdigitsdf)
#Create a formula of the first 8 features rather than type it out each time

vars <- paste(names(lettersdigitsdf)[3:10],sep="")
firstfeatures <- paste("dummy.category ~", paste(vars, collapse="+"))
firstfeatures <- as.formula(firstfeatures)

#model below:

glmfit1<-glm(firstfeatures, data = lettersdigitsdf, family = 'binomial')
summary(glmfit1)

lettersdigitsdf[["predicted_val"]] = predict(glmfit1, lettersdigitsdf, type="response")
lettersdigitsdf[["predicted_class"]] = 0
lettersdigitsdf[["predicted_class"]][lettersdigitsdf[["predicted_val"]] > 0.5] = 1

correct_items = lettersdigitsdf[["predicted_class"]] == lettersdigitsdf[["dummy.category"]] 

# proportion correct:
nrow(lettersdigitsdf[correct_items,])/nrow(lettersdigitsdf)
# Classifier is approx 60% accurate

# proportion incorrect:
nrow(lettersdigitsdf[!correct_items,])/nrow(lettersdigitsdf)

#confusionMatrix(data=as.factor(lettersdigitsdf[["predicted_class"]]), as.factor(lettersdigitsdf$dummy.category))


# Section 1.3
# ~~~~~~~~~~~


set.seed(3060)

# 7 fold CV for optimal feature - no_neighbours_below
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #Second method of k fold cross validation
train_control1 <- trainControl(method = "cv", number = 7, savePredictions = TRUE)
#?trainControl
model2 <- train(dummy.category ~ no_neighbours_after,
               data = lettersdigitsdf,
               trControl = train_control1,
               method = "glm",
               family=binomial)

summary(model2)
#model2$pred


predictions = predict(model2, lettersdigitsdf)
pred <- function(t) ifelse(predictions > t, "Letters","Digits") 
#function makes prediction model binary; if a p value > 0.5 then prediction is that symbol is a letter, else it's a digit
pred(0.5)

# str(as.factor(pred))
# levels(pred)
# str(lettersdigitsdf$Category)
# levels(lettersdigitsdf$Category)

confusionMatrix(data=as.factor(pred(0.5)), as.factor(lettersdigitsdf$Category))
#Accuracy is the exact same as when it wasn't cross validated - 0.6964, (no hyper parameters).

# 7 fold CV for first 8 features
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(3060)
train_control3 <- trainControl(method = "cv", number = 7, savePredictions = TRUE)
#?trainControl
model3 <- train(firstfeatures,
                data = lettersdigitsdf,
                trControl = train_control3,
                method = "glm",
                family=binomial)

summary(model3)
model3$pred


predictions1 = predict(model3, lettersdigitsdf)
pred1 <- function(t) ifelse(predictions1 > t, "Letters","Digits") 
#function makes prediction model binary; if a p value > 0.5 then prediction is that symbol is a letter, else it's a digit

pred1(0.5)
confusionMatrix(data=as.factor(pred1(0.5)), as.factor(lettersdigitsdf$Category))
# Again, same accuracies. 0.6071 accuracy.


# # Section 1.4
# ~~~~~~~~~~~

# Accuracy of optimal feature (no_neighbours_after) logistic regression = 0.6964
# So amount of successes in sample -> 0.6984 X 112 = 78 - also it states this in confusion matrix

1 - pbinom(78, 112, 0.5) # P value returned is very small, so the model is significant

# Accuracy of first 8 features logistic regression = 0.6071
# So amount of successes in sample = 68

1 - pbinom(68, 112, 0.5) # P value returned is very small, so the model is significant

# # Section 1.5
# ~~~~~~~~~~~

#Best feature classifier

cmatrix <- confusionMatrix(data=as.factor(pred(0.5)), as.factor(lettersdigitsdf$Category))
justtable <- cmatrix$table
justtable # Confusion matrix for best feature classifier

# (a): Inaccurately classifies digit 18 times and 
# (b): Inaccurately classifies letter 16 times

#First 8 features classifier

cmatrix1 <- confusionMatrix(data=as.factor(pred1(0.5)), as.factor(lettersdigitsdf$Category))
justtable1 <- cmatrix1$table
justtable1 # Confusion matrix for best feature classifier

# (a): Inaccurately classifies digit 21 times and 
# (b): Inaccurately classifies letter 23 times


