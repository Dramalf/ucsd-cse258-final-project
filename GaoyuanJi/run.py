import dataPreProcessing
import models
import eval
import pandas as pd
import torch
import numpy as np
import math
from sklearn.model_selection import train_test_split
'''
views=dataPreProcessing.dataLoader('US','views').to_list()
likes=dataPreProcessing.dataLoader('US','likes').to_list()
feature=[]
for i in likes:
    entry=[1]
    if(i!=0):
        entry.append(math.log(i))
    else:
        entry.append(0)
    feature.append(entry)
viewsLog=[math.log(i) for i in views]
XTrain, XTest, YTrain, YTest = train_test_split(feature, viewsLog, test_size=0.1, random_state=42)

prediction=models.signleFeatureRegression(XTrain,YTrain,XTest,YTest)
print('SIGNLE FEATURE : LIKES ',eval.evaluate_model(YTest,prediction,'regression',['mse','mae','r2']))
###########################################################
comments=dataPreProcessing.dataLoader('US','comment_count').to_list()
feature=[]
for i in comments:
    entry=[1]
    if(i!=0):
        entry.append(math.log(i))
    else:
        entry.append(0)
    feature.append(entry)

XTrain, XTest, YTrain, YTest = train_test_split(feature, viewsLog, test_size=0.1, random_state=42)
prediction=models.signleFeatureRegression(XTrain,YTrain,XTest,YTest)
print('SIGNLE FEATURE : COMMENTS ',eval.evaluate_model(YTest,prediction,'regression',['mse','mae','r2']))
###########################################################
dislikes=dataPreProcessing.dataLoader('US','dislikes').to_list()
feature=[]
for i in dislikes:
    entry=[1]
    if(i!=0):
        entry.append(math.log(i))
    else:
        entry.append(0)
    feature.append(entry)

XTrain, XTest, YTrain, YTest = train_test_split(feature, viewsLog, test_size=0.1, random_state=42)
prediction=models.signleFeatureRegression(XTrain,YTrain,XTest,YTest)
print('SIGNLE FEATURE : DISLIKES ',eval.evaluate_model(YTest,prediction,'regression',['mse','mae','r2']))
##################################################################
documents=dataPreProcessing.dataLoader('US','tags')
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000)
print('TFIDF : TAGS ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000,SVD=True)
print('TFIDF-SVD : TAGS ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
##################################################################
documents=dataPreProcessing.dataLoader('US','description')
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000)
print('TFIDF : DESCRIPTION ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000,SVD=True)
print('TFIDF-SVD : DESCRIPTION ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
##################################################################
documents=dataPreProcessing.dataLoader('US','title')
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000)
print('TFIDF : TITLE ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
prediction,Ytest=models.tfidfRegression(documents,viewsLog, 20000,SVD=True)
print('TFIDF-SVD : TITLE ',eval.evaluate_model(Ytest,prediction,'regression',['mse','mae','r2']))
'''

documents=dataPreProcessing.dataLoader('US','tags').to_list()
views=dataPreProcessing.dataLoader('US','views').to_list()


viewsLog=[math.log(i) for i in views]
model=models.transformer(documents,viewsLog)