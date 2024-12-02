import kagglehub
import json
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import scipy.stats as stats
import numpy as np  
import re
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
    return rawData.dropna()
def numberDistribution(name:str):
    '''
    Basic information about numerical field in data

    Save numerical data distribution as "CountryFieldDistribution.png" in folder "/dataAnalyse" e.g. "USviewsDistrbution.png"

    Save basic statastic data as "CountryBasicStatistic.csv" in folder "/dataAnalyse" e.g. "CABasicStatistic.csv"
    '''
    pd.options.display.float_format = "{:.2f}".format
    data=loadCleanedData(name)
    number=data.select_dtypes(include=['number'])
    number.drop('category_id',axis=1,inplace=True)
    number.describe().to_csv('dataAnalyse/'+name+'BasicStatistic.csv')
    
    print(number.describe())
    for column in number:       
        params = stats.expon.fit(number[column].to_numpy().transpose())        
        loc, scale = params
        fig, ax1 = plt.subplots()
        percentile_90 = number[column].quantile(0.9)   
        sns.histplot(number[number[column]<percentile_90][column], kde=True, bins=500, color='blue', alpha=0.6,lw=2,label='Views (<= 90% percentile)')
        
        ax2 = plt.twinx()
        xmin, xmax = plt.xlim()  
        plt.xlim(0,xmax)
        
        x = np.linspace(xmin, xmax, 100)
        pdf_fitted = stats.expon.pdf(x, loc, scale)
        ax2.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Exponential')
        plt.ylim(bottom=0)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('Histogram with KDE'+' : '+column)
        fig.savefig('dataAnalyse/'+name+column+'Distrbution.png')
    
def wordFrequency(name:str):
    """
    Word frequency of string type fields in data

    Save word cloud as "CountryFieldWordCloud.png" in folder "/dataAnalyse" e.g. "USchannel_titleWordCloud.png"

    Save most 1000 common words for each column as "CountryFieldMostCommon1000.csv" in folder "/dataAnalyse" e.g. "UStagsMostCommon1000.csv"
    """
    data=loadCleanedData(name)
    data.drop(labels=['video_id','trending_date','thumbnail_link','publish_time'],axis=1,inplace=True)
    text=data.select_dtypes(include=['object'])
    
    stop_words = set(stopwords.words('english'))
    for column in text:
        allText=''.join(text[column])
        allText=re.sub(r'http[s]?://\S+', '', allText)
        allText = re.sub(r'[^\w\s]', '', allText)
        allText = allText.lower()
        allText = ' '.join([word for word in allText.split() if word not in stop_words])
        allWords = allText.split()
        wordCounts=Counter(allWords)
        mostCommon=wordCounts.most_common(1000)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordCounts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud in field : '+column)
        plt.axis('off')
        img=wordcloud.to_image()
        img.save('dataAnalyse/'+name+column+'WordCloud.png')
        pd.DataFrame(mostCommon).to_csv('dataAnalyse/'+name+column+'MostCommon1000.csv')
def dataLoader(name:str,column:str):
    return loadCleanedData(name)[column]
if __name__ == "__main__":
    print(dataLoader('US','views').to_numpy())
    