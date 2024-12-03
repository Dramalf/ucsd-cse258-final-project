import dataPreProcessing
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
def signleFeatureRegression(XTrain,YTrain,XTest,YTest):
    
    model = LinearRegression()
    model.fit(XTrain, YTrain)
    return model.predict(XTest)

def tfidfRegression(documents,labels,max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, 
    stop_words='english', 
    )

    
    tfidfMatrix = vectorizer.fit_transform(documents)
    XTrain, XTest, YTrain, YTest = train_test_split(tfidfMatrix, labels, test_size=0.1, random_state=42)
    model = LinearRegression()
    model.fit(XTrain, YTrain)
    return model.predict(XTest),YTest
