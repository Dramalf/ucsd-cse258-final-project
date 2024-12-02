import dataPreProcessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import torch
import numpy as np
def signleFeatureRegression(trainSet,testSet):
    XTrain=trainSet[0]
    YTrain=trainSet[1]
    XTest=testSet[0]
    YTest=testSet[1]
    model = LogisticRegression(
    solver='saga',          
    penalty='l2',           
    max_iter=1000,           
    early_stopping=True,     
    validation_fraction=0.1, 
    n_iter_no_change=5,      
    tol=1e-4                
    )
    model.fit(XTrain, YTrain)
    return model.predict(XTest)