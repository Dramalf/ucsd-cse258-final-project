import dataPreProcessing
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import numpy as np
def signleFeatureRegression(XTrain,YTrain,XTest,YTest):
    
    model = LinearRegression(
             
         
    )
    model.fit(XTrain, YTrain)
    return model.predict(XTest)
