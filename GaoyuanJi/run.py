import dataPreProcessing
import models
import eval
import pandas as pd
import torch
import numpy as np
import math
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
XTrain=feature[:(int)(len(feature)*0.9)]
XTest=feature[(int)(len(feature)*0.9):]
YTrain=viewsLog[:(int)(len(feature)*0.9)]
YTest=viewsLog[(int)(len(feature)*0.9):]

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

XTrain=feature[:(int)(len(feature)*0.9)]
XTest=feature[(int)(len(feature)*0.9):]
YTrain=viewsLog[:(int)(len(feature)*0.9)]
YTest=viewsLog[(int)(len(feature)*0.9):]
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

XTrain=feature[:(int)(len(feature)*0.9)]
XTest=feature[(int)(len(feature)*0.9):]
YTrain=viewsLog[:(int)(len(feature)*0.9)]
YTest=viewsLog[(int)(len(feature)*0.9):]
prediction=models.signleFeatureRegression(XTrain,YTrain,XTest,YTest)
print('SIGNLE FEATURE : DISLIKES ',eval.evaluate_model(YTest,prediction,'regression',['mse','mae','r2']))