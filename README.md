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
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1395    0    0    0    0
         B    0  949    2    0    0
         C    0    0  850    1    0
         D    0    0    3  802    0
         E    0    0    0    1  901

Overall Statistics
                                          
               Accuracy : 0.9986          
                 95% CI : (0.9971, 0.9994)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9982          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   0.9942   0.9975   1.0000
Specificity            1.0000   0.9995   0.9998   0.9993   0.9998
Pos Pred Value         1.0000   0.9979   0.9988   0.9963   0.9989
Neg Pred Value         1.0000   1.0000   0.9988   0.9995   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2845   0.1935   0.1733   0.1635   0.1837
Detection Prevalence   0.2845   0.1939   0.1735   0.1642   0.1839
Balanced Accuracy      1.0000   0.9997   0.9970   0.9984   0.9999
```
5. Obtain the predictions by applying the model on test data.Since random forest method gives better accuracy it is finalized and applied on the test data to obtain the predictions
```{r}
#######since random forest method gives better accuracy using that on test data####
rf_predict1<-predict(rf_fit,pml_testing1,type="class")
cbind(problem_id=pml_testing$problem_id,as.data.frame(rf_predict1))
   problem_id rf_predict1
1           1           B
2           2           A
3           3           B
4           4           A
5           5           A
6           6           E
7           7           D
8           8           B
9           9           A
10         10           A
11         11           B
12         12           C
13         13           B
14         14           A
15         15           E
16         16           E
17         17           A
18         18           B
19         19           B
20         20           B
```



