from turkishnlp import detector
obj = detector.TurkishNLP()
obj.download()
obj.create_word_set()
import turkishnlp
import json
from time import time
import GetOldTweets3 as got
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px #visualization
import pandas as pd
from nltk import word_tokenize
import preprocessor as p
from nltk.corpus import stopwords
import numpy as np
import re
import string
import preprocessor as p
from snowballstemmer import TurkishStemmer
from textblob import TextBlob

import GetOldTweets3 as got

def get_tweets(query,onceki,sonraki):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                            .setSince(onceki)\
                                            .setUntil(sonraki)\
                                            .setMaxTweets(100)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    
    text_tweets = [[tw.text,
                tw.date,
                tw.hashtags] for tw in tweet]
    #df = pd.DataFrame(text_tweets, columns = ['Text', 'Date','Hashtags'])
    return text_tweets

onceki = ['2017-01-01','2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01', '2017-07-01',
       '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01','2018-01-01','2018-02-01','2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01', '2018-07-01',
       '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01','2019-01-01','2019-02-01','2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01', '2019-07-01',
       '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01','2020-02-01', '2020-03-01','2020-04-01']
sonraki = ['2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01', '2017-07-01',
       '2017-08-01', '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01','2018-01-01','2018-02-01','2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01', '2018-07-01',
       '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01','2019-01-01','2019-02-01','2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01', '2019-07-01',
       '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01','2020-02-01', '2020-03-01','2020-04-01','2020-05-01'] 

def getData(query):
    b=[]
    for i,j in zip(onceki,sonraki):
        tweet=get_tweets(query,i,j)
        b+=tweet
    df = pd.DataFrame(b, columns = ['Text', 'Date','Hashtags']) 
    df.to_csv(query+".csv")
    return df    

stop_word_list = nltk.corpus.stopwords.words('turkish') # stop words
neg=pd.read_csv('negative_words_tr.txt') 
pos=pd.read_csv('positive_words_tr.txt')
def token(values):
    filtered_words = [word for word in values.split() if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc
def stemming_tokenizer(text): 
    stemmer = TurkishStemmer()
    return [stemmer.stemWord(w) for w in word_tokenize(text)] # köklere ayırma
def pos_finder(x):
    df_words = set(" ".join(x).split(" "))
    extract_words =  set(neg.eski.to_list()).intersection(df_words)
    return ', '.join(extract_words)
def neg_finder(x):
    df_words = set(" ".join(x).split(" "))
    extract_words =  set(pos.olan.to_list()).intersection(df_words)
    return ', '.join(extract_words)
def fixNeg(neg):
    if not (len(neg)==0):
        return len(neg.split(','))
    else:
        return 0
def fixPos(pos):
    if not (len(pos)==0):
        return len(pos.split(','))
    else:
        return 0
df=getData("acık eczane")
df=pd.read_csv("ilac.csv")
df=df.drop(columns=['Unnamed: 0',"Hashtags"z])
df=df.dropna()
df.info()

docs = df['Text']
docs = docs.map(lambda x: re.sub('[,\.!?();:$%&#"]', '', x)) #punc are removed
docs = docs.map(lambda x: x.lower()) # küçük harf
docs = docs.map(lambda x: x.strip())

docs = docs.map(lambda x: token(x))
df['Text'] = docs

#def anlamlıOl(kelime):
#    lwords = obj.list_words(kelime)
#    corrected_words = obj.auto_correct(lwords)
#    corrected_string = " ".join(corrected_words)
#    return corrected_string
#df.Text=df.Text.map(anlamlıOl)

df['text_token'] = df['Text'].apply(lambda text: stemming_tokenizer(text))
df.head()

df['neg'] = df.text_token.apply(pos_finder)
df['pos'] = df.text_token.apply(neg_finder)

df["len_neg"]=df.neg.map(fixNeg)
df["len_pos"]=df.pos.map(fixPos)

df=df[~((df.len_neg==0) & (df.len_pos==0))] # Burda stratejim eğer iki türlü de negatif ya da pozitif kelime bulamadıysa yabancı kelimedir.


df['Text_count'] = df['Text'].str.split().apply(len).value_counts()
df['polarity'] = df['len_pos'] - df['len_neg']
df['sent_rate']=(df['len_pos'] + df['len_neg'])/ len(df['Text_count'])

sentiment_open_pharm = open_pharm.groupby(["Date"]).polarity.mean().reset_index()
sentiment_open_pharm = sentiment_open_pharm[sentiment_open_pharm.Date<="202004"]

sentiment_open_pharm = sentiment_open_pharm.rename(columns={"Date":"Period","polarity":"Polarity"})
sentiment_open_pharm.to_csv("acık_eczane_sentiment_results.csv")