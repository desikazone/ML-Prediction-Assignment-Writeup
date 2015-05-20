# ML-Prediction-Assignment-Writeup
pml_training<-read.csv("C:/Ebooks/R/coursera/Machine learning/Raw data/pml-training.csv")
pml_testing<-read.csv("C:/Ebooks/R/coursera/Machine learning/Raw data/pml-testing.csv")

library(caret)
library(kernlab)
intrain<-createDataPartition(pml_training$classe,p=.75,list=FALSE)
training<-pml_training[intrain,]
test<-pml_training[-intrain,]

############Data cleaning#############
###remove column x which is the just the index###
train_cln1<-training[,-1]

###Removing variables with more than 60% NAs####
remove_var<-rep(NA,1)
temp      <-vector('character')
for (i in 1:length(train_cln1))
{
 if (sum(is.na(train_cln1[i]))/nrow(train_cln1[i]) >=.6) temp<-colnames(train_cln1[i])
 if (length(temp)==1) remove_var<-unique(rbind(remove_var,temp))
}
remove_var     <-as.vector(remove_var)
varNA          <-names(train_cln1) %in% remove_var
train_cln2     <-train_cln1[!varNA]

####remove Near Zero Variance variables##########
dataNZV        <- nearZeroVar(train_cln2, saveMetrics=TRUE)
NZVvar         <-as.vector(row.names(dataNZV[dataNZV$nzv=="TRUE",]))
NZVvar_bin     <-names(train_cln2) %in% NZVvar
train_cln3     <-train_cln2[!NZVvar_bin]


#########cleaning both validation and test data sets########
cln1           <-names(train_cln3)
cln2           <-names(train_cln3[-58])
test           <-test[cln1]
pml_testing1    <-pml_testing[cln2]
levels(pml_testing1$cvtd_timestamp)<-levels(train_cln3$cvtd_timestamp)


##############Prediction using Decision Tree############
dtree_fit<-rpart(classe~.,method="class",data=train_cln3)
library(rattle)
fancyRpartPlot(dtree_fit)
dtree_predict<-predict(dtree_fit,test,type="class")
confusionMatrix(dtree_predict,test$classe)

###########Predictino using random forest###########
rf_fit<-randomForest(classe~.,data=train_cln3)
rf_predict<-predict(rf_fit,test,type="class")
confusionMatrix(rf_predict,test$classe)


#######since random forest method gives better accuracy using that on test data####
rf_predict1<-predict(rf_fit,pml_testing1,type="class")

#####function to write output files###
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

getwd()
pml_write_files(rf_predict1)
