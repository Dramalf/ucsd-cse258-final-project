import kagglehub
import json
import pandas as pd
def downloadDataset():
    '''download dataset to the default path (not '/dataset')'''
    dataset_path = kagglehub.dataset_download("datasnaek/youtube-new")
    config = {"DATASET_PATH": dataset_path}
    with open("config.json", "w") as file:
        json.dump(config, file)
    print("Dataset path saved to config.json")

def loadRawData(name:str):
    '''load raw csv data'''
    with open("config.json", "r") as file:
        config = json.load(file)
        dataset_path = config.get("DATASET_PATH", None)   
        data=pd.read_csv(dataset_path+'/'+name+'videos.csv')
        return data
def loadCleanedData(name:str):
    '''clean raw data'''
    rawData=loadRawData(name)
    
if __name__ == "__main__":
    loadRawData('US')