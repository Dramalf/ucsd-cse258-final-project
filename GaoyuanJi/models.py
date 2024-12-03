import dataPreProcessing
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import eval
def signleFeatureRegression(XTrain,YTrain,XTest,YTest):
    
    model = LinearRegression()
    model.fit(XTrain, YTrain)
    return model.predict(XTest)

def tfidfRegression(documents,labels,max_features,SVD=False):
    vectorizer = TfidfVectorizer(max_features=max_features, 
    stop_words='english', 
    ngram_range=(1,2)
    )
     
    
    tfidfMatrix = vectorizer.fit_transform(documents)
    if(SVD):
        svd = TruncatedSVD(n_components=5000) 
        XReduced = svd.fit_transform(tfidfMatrix) 
        XTrain, XTest, YTrain, YTest = train_test_split(XReduced, labels, test_size=0.1, random_state=42)
    else: 
        XTrain, XTest, YTrain, YTest = train_test_split(tfidfMatrix, labels, test_size=0.1, random_state=42)
    model = LinearRegression()
    model.fit(XTrain, YTrain)
    return model.predict(XTest),YTest
class TransformerRegressor(nn.Module):
    def __init__(self, model_name):
        super(TransformerRegressor, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1  
        )

    def forward(self,input_ids, attention_mask,token_type_ids):
        outputs = self.transformer( input_ids=input_ids,attention_mask=attention_mask)
        return outputs.logits  
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: self.inputs[key][idx] for key in self.inputs}, self.labels[idx]
def preprocess_data(data):
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer(
        data['text'].to_list(),  # 文本列表
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",  # 返回 PyTorch 张量
        
    )
    labels = torch.tensor(data['score'].to_list())  # 转换为张量
    return inputs, labels
def transformer(documents,labels,model=None):
    data=pd.DataFrame(data={"text": documents,"score":labels})
    
   
    train_data, val_data = train_test_split(data, test_size=0.2)
    train_inputs, train_labels = preprocess_data(train_data)
    val_inputs, val_labels = preprocess_data(val_data)

    train_dataset = TextDataset(train_inputs, train_labels)
    val_dataset = TextDataset(val_inputs, val_labels)
    
    if(model is None):
        model = TransformerRegressor("prajjwal1/bert-tiny")
    
    bestMse=10000
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("mps")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()
   

    
    
   
    
    epochs = 40
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}")
        model.eval()
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device).float()

                outputs = model(**inputs)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        mse = mean_squared_error(val_labels_list, val_preds)
        if(mse<bestMse):
            checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Model checkpoint saved for epoch {epoch+1}.")
            bestMse=mse
        
        print(eval.evaluate_model(val_labels_list,val_preds,'regression',['mse','mae','r2']))
    return model