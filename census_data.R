# reading in data
census_data <- read.csv("~/Eg_Datasets/census_data.txt", header=FALSE)
str(census_data)
head(census_data)
library(DataExplorer)
plot_missing(census_data)
summary(census_data)
table(census_data$V15)

#data partition 
library(caret)
par <- createDataPartition(census_data$V15,p=0.75,list=F)
training <-census_data[par,] 
testing <- census_data[-par,]

#################### logistic model ########################
fit1 <- glm(V15~.,data=training,family = binomial)
summary(fit1)

#prediction
p <- predict(fit1,newdata = testing,type="response")
head(p)
pred1 <- ifelse(p>0.5,">50k","<=50k")
tab <- table(pred1,testing$V15)
tab
sum(diag(tab))/sum(tab) #accuracy=84.7%
contrasts(census_data$V15)
varImp(fit1)

##trying different models to boost the accuracy
################### using caret package #####################
control <- trainControl(method = "cv", number=10)
fit2 <- train(V15~. ,data=training,trControl=control,method="glm",family="binomial")
summary(fit2)

#prediction
pred2 <- predict(fit2,newdata = testing)

#checking the accuracy
confusionMatrix(pred2,testing$V15)
varImp(fit2)

############# classification trees ###################
fit3 <- train(V15~. ,data=training,trControl=control,method="rpart")
fit3 

#prediction
pred3 <- predict(fit3,testing)
confusionMatrix(pred3,testing$V15)
library(rattle)
fancyRpartPlot(fit3$finalModel)

############### random forest ###################
library(randomForest)
set.seed(123)
rf <- randomForest(V15~.,data=training,ntree=150,mtry=3)
(rf)
pred4 <- predict(rf,testing)
confusionMatrix(pred4,testing$V15) #86.5% accuracy

#tune mtry
names(training)
tuneRF(training[,-15],training[,15],stepFactor = 1,plot=T,ntreeTry =350,trace = T,improve = 0.05 )

############ gbm ######################
#Creating grid
fit4 <- train(V15~. ,data=training,trControl=control,method="gbm")
p1 <- predict(fit4,testing)
confusionMatrix(p1,testing$V15)

############# Neural networks #####################
fit5 <- train(V15~. ,data=training,trControl=control,method="nnet")
plot(fit5)
p2 <- predict(fit5,testing)
confusionMatrix(p2,testing$V15)
varImp(fit5)
#lower accuracy of 79% with increased sensitivity of 96%