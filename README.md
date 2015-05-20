# Predictive Model to determine the exercising pattern
This document describe the analysis done for the prediction assignment of the practical machine learning course.

1.The first step is to load the csv file data to dataframe
```{r}
  pml_training<-read.csv("C:/Ebooks/R/coursera/Machine learning/Raw data/pml-training.csv")
  pml_testing<-read.csv("C:/Ebooks/R/coursera/Machine learning/Raw data/pml-testing.csv")
```

2.After loading the files, I split the pml_training data into test and training data so that I can cross validate the results/output.Pml_testing data is left untouched till finalization of the model.I have split the data such that 75% of the data is classified as training and rest as testing.
```{r}
  library(caret)
  library(kernlab)
  intrain<-createDataPartition(pml_training$classe,p=.75,list=FALSE)
  training<-pml_training[intrain,]
  test<-pml_training[-intrain,]
```
3.Next is the data cleaning stage.

  a. First the column x present in the data is removed since it is just an index and will not be helpful for the analysis
```{r}
  ###remove column x which is just the index###
  train_cln1<-training[,-1]
```
  b. Second remove the columns with more than 60% NAs. Columns with more than 60% NAs will not be good enough to contribute to the predictive model.
```{r}
  ###Removing variables with more than 60% NAs####
  remove_var<-rep(NA,1)#create a vector with length 1 containing value NA
  temp      <-vector('character') #empty vectory
  for (i in 1:length(train_cln1))
  {
   if (sum(is.na(train_cln1[i]))/nrow(train_cln1[i]) >=.6) temp<-colnames(train_cln1[i])#get colname if NAs >=.6
   if (length(temp)==1) remove_var<-unique(rbind(remove_var,temp))#collate all colnames with NAs>=.6
  }
  remove_var     <-as.vector(remove_var)#convert into vector
  varNA          <-names(train_cln1) %in% remove_var #finalizing columns with NAs >=.6
  train_cln2     <-train_cln1[!varNA] #dataset where columns with NAs>=.6 removed
```
 c. Next step is removal of near zero variance variables.Datasets sometimes contain variables which contain almost constant values throughout the data. These are non  informative and will not add any value to the model building process.
 ```{r}
  ####remove Near Zero Variance variables##########
  dataNZV        <- nearZeroVar(train_cln2, saveMetrics=TRUE)#function  to find near zero variance variables
  NZVvar         <-as.vector(row.names(dataNZV[dataNZV$nzv=="TRUE",]))#obtain column names of near zero variance variables
  NZVvar_bin     <-names(train_cln2) %in% NZVvar #match and pick the final near zero variance variables
  train_cln3     <-train_cln2[!NZVvar_bin] #remove the NZV from training data
  ```
 d. Final step is to clean training, validation and test datasets for removal of variables identified as non-important based on above 3 steps
 ```{r}
 #########cleaning both validation and test data sets########
cln1           <-names(train_cln3)#column names to be kept in training data
cln2           <-names(train_cln3[-58])#column names to be kept in testing data
test           <-test[cln1]#keep only column names in cln1 vector
pml_testing1    <-pml_testing[cln2]#keep column names present in cln2 vector
levels(pml_testing1$cvtd_timestamp)<-levels(train_cln3$cvtd_timestamp)#this step is included since R throws an error while predicting due to difference in levels of training and test data
```
4. Next stage is model building. Two models are build one is prediction with Decision tree and the other with Random Forest. Confusion matrix is obtained to check the accuracy of both the models built. Model with high accuracy is shortlisted for the final prediction.
```{r}
##############Prediction using Decision Tree############
dtree_fit<-rpart(classe~.,method="class",data=train_cln3)
library(rattle)
fancyRpartPlot(dtree_fit)
dtree_predict<-predict(dtree_fit,test,type="class")
confusionMatrix(dtree_predict,test$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1346   44    4    1    0
         B   29  785   55   44    0
         C   20  117  783  119   38
         D    0    3    5  508   34
         E    0    0    8  132  829

Overall Statistics
                                         
               Accuracy : 0.8668         
                 95% CI : (0.857, 0.8762)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.8315         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9649   0.8272   0.9158   0.6318   0.9201
Specificity            0.9860   0.9676   0.9274   0.9898   0.9650
Pos Pred Value         0.9649   0.8598   0.7270   0.9236   0.8555
Neg Pred Value         0.9860   0.9589   0.9812   0.9320   0.9817
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2745   0.1601   0.1597   0.1036   0.1690
Detection Prevalence   0.2845   0.1862   0.2196   0.1122   0.1976
Balanced Accuracy      0.9755   0.8974   0.9216   0.8108   0.9426

###########Predictino using random forest###########
rf_fit<-randomForest(classe~.,data=train_cln3)
rf_predict<-predict(rf_fit,test,type="class")
confusionMatrix(rf_predict,test$classe)


#######since random forest method gives better accuracy using that on test data####
rf_predict1<-predict(rf_fit,pml_testing1,type="class")



This analysis allows us to note two main points : 1 - Some numeric data have been imported as factor because of the presence of some characters ("#DIV/0!") 2 - Some columns have a really low completion rate (a lot of missing data)

To manage the first issue we need to reimport data ignoring "#DIV/0!" values :

data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv", na.strings=c("#DIV/0!") )

And force the cast to numeric values for the specified columns (i.e.: 8 to end) :

cData <- data
for(i in c(8:ncol(cData)-1)) {cData[,i] = as.numeric(as.character(cData[,i]))}

To manage the second issue we will select as feature only the column with a 100% completion rate ( as seen in analysis phase, the completion rate in this dataset is very binary) We will also filter some features which seem to be useless like "X"", timestamps, "new_window" and "num_window". We filter also user_name because we don't want learn from this feature (name cannot be a good feature in our case and we don't want to limit the classifier to the name existing in our training dataset)

featuresnames <- colnames(cData[colSums(is.na(cData)) == 0])[-(1:7)]
features <- cData[featuresnames]

We have now a dataframe "features which contains all the workable features. So the first step is to split the dataset in two part : the first for training and the second for testing.

xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]

We can now train a classifier with the training data. To do that we will use parallelise the processing with the foreach and doParallel package : we call registerDoParallel to instantiate the configuration. (By default it's assign the half of the core available on your laptop, for me it's 4, because of hyperthreading) So we ask to process 4 random forest with 150 trees each and combine then to have a random forest model with a total of 600 trees.

registerDoParallel()
model <- foreach(ntree=rep(150, 4), .combine=randomForest::combine) %dopar% randomForest(training[-ncol(training)], training$classe, ntree=ntree)

To evaluate the model we will use the confusionmatrix method and we will focus on accuracy, sensitivity & specificity metrics :

predictionsTr <- predict(model, newdata=training)
confusionMatrix(predictionsTr,training$classe)

## 
## Attaching package: 'e1071'
## 
## L'objet suivant est masqué from 'package:Hmisc':
## 
##     impute

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000

predictionsTe <- predict(model, newdata=testing)
confusionMatrix(predictionsTe,testing$classe)

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  946    6    0    0
##          C    0    2  849    6    1
##          D    0    0    0  798    1
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.993    0.993    0.998
## Specificity             1.000    0.998    0.998    1.000    1.000
## Pos Pred Value          0.999    0.994    0.990    0.999    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.183
## Detection Prevalence    0.285    0.194    0.175    0.163    0.183
## Balanced Accuracy       1.000    0.998    0.995    0.996    0.999

As seen by the result of the confusionmatrix, the model is good and efficient because it has an accuracy of 0.997 and very good sensitivity & specificity values on the testing dataset. (the lowest value is 0.992 for the sensitivity of the class C)

It seems also very good because It scores 100% (20/20) on the Course Project Submission (the 20 values to predict)

I also try to play with preprocessing generating PCA or scale & center the features but the accuracy was lower.



