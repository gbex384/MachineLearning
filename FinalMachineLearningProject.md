---
title: "TBD"
author: "Philip Coyne"
date: "June 4, 2015"
output: 
  html_document:
    keep_md: true
---
=================================================
# Machine Learning Project
## Written by: *Philip Coyne*
### Date: June 4, 2015



*Background*:
Data acquired from self movement devices, such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* allows one to monitor how much excercise one performs, however the question of "How well excercise is done" arises.  In an effort to answer this question, scientists have conducted an experiment by attaching 4 monitor devices to individuals and analyzed metric data given by these devices.  The individuals were given specific instructions as to how to perform their excercise (lifting dumbells); six different "classes" were designated as "A", "B","C","D",and "E".  The metrics and the given classe information were then paired together in a massive Excel spreadsheet and made public for use.

**Goal**:

The information provided contains approximately 170 unique variables from six different subjects.  In the interest of demonstrating what has been learned from this class, the assignment is to employ a machine learning algorithm to create a model to predict the "classe" variable for a subject.  

## Part 1: Loading and Pre-Processing the Data
The following libraries were used throughout the development of the machine learning model, as well as preliminary examination of the raw data set.  In the interest of being concise, not everything will be shown



```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

```r
library("randomForest")
```

To begin, the data for the training and test sets were downloaded and loaded into R. 

```r
doop<-read.csv("pml-training.csv",header=TRUE,sep=",")
testing<-read.csv("pml-testing.csv",header=TRUE,sep=",")
```

Both the training and test set data are listed to have 160 unique variables (columns).  The training data has 19,622 observations (rows), wheras the testing data has 20 observations.

The data was noted to have a significant amount of missing or NA values for entire variables.  These variables were removed since they did not offer any valuable data.  The following R code was utilized to remove those NA and missing values.


```r
doopNACount<-colSums(is.na(doop))
filterCount<-doopNACount>.5*dim(doop)[1]
doopTest<-doop[filterCount]
naDoopNames<-names(doopTest)

nuDoop<-doop[,-which(names(doop)%in% c(naDoopNames))]

#Next we want to remove columns that have empty data slots
#If a column has more than 50% of the maximum rows as blank slots
#it should be removed.
nuDoop[nuDoop==""]<-NA

doopNACount<-colSums(is.na(nuDoop))
filterCount<-doopNACount>.5*dim(nuDoop)[1]
doopTest<-nuDoop[filterCount]
naDoopNames<-names(doopTest)

newNuDoop<-nuDoop[,-which(names(nuDoop)%in% c(naDoopNames))]
```

It was determined that if a variable has at least 50% of the maximum observations as NA or missing, it is not worth keeping.  This lack of data was assumed to be from an error during the experiment, or intentionally left blank.    

It should be noted that the test set was subjected to the same pre-processing.  The following R code was used to accomplish this.


```r
testNACount<-colSums(is.na(testing))
filterCount<-testNACount>.5*dim(testing)[1]
filterTest<-testing[filterCount]
naDoopNames<-names(filterTest)

testNAR<-testing[,-which(names(testing)%in% c(naDoopNames))]
```



Additional steps were requires to ensure that the data set was ready to be used in a machine learning model.  Information that is irrelevant to how well an excercise was performed was removed (Ex: Name of subject, time of day the excercise was performed, etc.).  


```r
filteredData<-newNuDoop[,8:60]
```

At this point, the data set known as "filteredData" contained only numeric data, save for the final variable "classe" which would be used to train the model.

## Step 2 Creating a model

Being introduced to this class, it is evident there are many models that can be used for this project.  It was noted that a paper was written based on this data.  A link ( http://groupware.les.inf.puc-rio.br/har) was provide to the describe the experiment and data set.  The paper describing this experiment (http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) provided some valuable insight.  Section 5 of the paper describes that a Random Forest approach was used. 

Looking more into the Random Forest approach, it was noted that R allows for automated pre-processing by boostrapping samples, and gorwing multiple decision trees at once.  The benefit of having accuracy amidst 53 unique variables was welcomed.  Despite this, overfitting is a concern.  


```r
modelFit <- randomForest (as.factor(filteredData$classe) ~ ., filteredData, ntree=50,norm.votes=FALSE)
```

The default value for ntree is 500; in the interest of saving time, ntree was set to 50 for the training set.  

## Step 3: Cross Validation and Out of Sample Error

Given the cross validation options provided in class, it was decided that k-folding would perhaps be the best approach.  A large k-fold value would give less bias between the training data and the data to be predicted.  According to the video lecture, there will be more variance in our predictions, which depends heavily on the random sampling of the training data.  With these two things in mind, 10 folds were used for Cross Validation purposes.


```r
k<-10
set.seed(32323)
folds<-createFolds(y=filteredData$classe,k=10,list=TRUE,returnTrain=TRUE)
err.vect<-rep(NA,k)
for(i in 1:k){
  
  foldTrainIndex<-folds[c(i)]
  foldTrain<-filteredData[foldTrainIndex[[1]],]
  foldTest<-filteredData[c(-foldTrainIndex[[1]]),]
  modelFit<-randomForest(as.factor(foldTrain$classe)~.,foldTrain,ntree=50,norm.votes=FALSE)
  predictionThing<-predict(modelFit,foldTest[,-53])
  err.vect[i]<-sum(foldTest[,53]==predictionThing)/(length(predictionThing))
  
  print(paste("Accuracy for fold",i,":",err.vect[i]))
  
  
}
```

```
## [1] "Accuracy for fold 1 : 0.996432212028542"
## [1] "Accuracy for fold 2 : 0.99592252803262"
## [1] "Accuracy for fold 3 : 0.997450280469148"
## [1] "Accuracy for fold 4 : 0.99592252803262"
## [1] "Accuracy for fold 5 : 0.993377483443709"
## [1] "Accuracy for fold 6 : 0.996435845213849"
## [1] "Accuracy for fold 7 : 0.995412844036697"
## [1] "Accuracy for fold 8 : 0.995412844036697"
## [1] "Accuracy for fold 9 : 0.997451580020387"
## [1] "Accuracy for fold 10 : 0.99592252803262"
```

```r
print(paste("Average Accuracy:",mean(err.vect)))
```

```
## [1] "Average Accuracy: 0.995974067334689"
```

Reviewing the numbers, we see that the average accuracy from the K-fold technique is ~99%.  This could indicate that there the model is overfitting for the training data set.  Additionally this would indicate that the out of sample error (as said in the lecture videos, larger than the in-sample error), could be anywhere between 1.1% to 100%.  

Given that there are 5 potential classes to determine though, it is likely that the out of sample error is low, perhaps around 5%.  

## Step 4: Apply the prediction model to the test set

The last step would be applying the model to the test set.  Afterwards a function was used to place the predcition into .txt files.


```r
pred<-predict(modelFit,testNAR)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
```

The .txt files were submitted into the course site and achieved a 100% success rate.  Operating under the assumption that the if there were a 21st observation to utilize the model, and it the prediction was incorrect, the out of sample error would be 1/21 or ~4%.  

