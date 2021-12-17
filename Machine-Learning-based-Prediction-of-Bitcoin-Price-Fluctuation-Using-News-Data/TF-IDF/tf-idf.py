# -*- coding:utf-8 -*-
import math
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
import sklearn
#import textblob
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import texthero as hero
import openpyxl
from collections import defaultdict
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
############################################################################################################################
def text_to_wordlist(text, remove_stop_words=True, lemma_words=True):
    ''' Clean each document into a list of words:
    1. convert abbrevations to full words
    2. tokenize the text
    3. remove non-alphabetic characters and one-letter words, including numbers and punctuations
    4. remove stop words
    '''
    # clean the text, convert only the abbrs that are meaningful
    text = re.sub(r" A.T.M. ", " Automated Teller Machine ", str(text))
    text = re.sub(r" C.I.A. ", " Central Intelligence Agency ", str(text))
    text = re.sub(r" D.C. ", " District of columbia ", str(text))
    text = re.sub(r" E.U. ", " Europian Union ", str(text))
    text = re.sub(r" F.B.I. ", " Federal Bureau of Investigation ", str(text))
    text = re.sub(r" H.I.V. ", " Human immunodeficiency virus ", str(text))
    text = re.sub(r" I.H.T. ", " inheritance tax ", str(text))
    text = re.sub(r" I.M.F. ", " International Monetary Fund ", str(text))
    text = re.sub(r" I.D. ", " identification ", str(text))
    text = re.sub(r" L.G.B.T. ", " minority ", str(text))
    text = re.sub(r" M.A. ", " Massachusetts ", str(text))
    text = re.sub(r" N.J. ", " new jersey ", str(text))
    text = re.sub(r" N.K. ", " north korea ", str(text))
    text = re.sub(r" N.S.A. ", " National Security Agency ", str(text))
    text = re.sub(r" N.Y. ", " new york ", str(text))
    text = re.sub(r" P.E.I. ", " Prince Edward Island ", str(text))
    text = re.sub(r" P.M. ", " prime minister ", str(text))
    text = re.sub(r" P.R.C ", " china ", str(text))
    text = re.sub(r" S.A. ", " south africa ", str(text))
    text = re.sub(r" R.I. ", " Rhode Island ", str(text))
    text = re.sub(r" U.A.E. ", " United Arab Emirates ", str(text))
    text = re.sub(r" U.K. ", " england ", str(text))
    text = re.sub(r" U.N. ", " new jersey ", str(text))
    text = re.sub(r" U.S. ", " america ", str(text))
    text = re.sub(r" U.S.C. ", " university of south california ", str(text))
    text = re.sub(r" W.H.O ", " world health organization ", str(text))
    text = re.sub(r" a.m. ", " morning ", str(text))
    text = re.sub(r" p.m. ", " afternoon ", str(text))
    text = re.sub(r" Ph.D. ", " doctor of philosophy ", str(text))
    text = re.sub(r" sq.m. ", " square meter ", str(text))

    # Tokenize the string into word tokens
    tokens = word_tokenize(text)

    # further clean the tokens: split toekns like "b'Russia" which still have punctuations in the token
    ls = []
    for word in tokens:
        if "'" in word:
            ls = ls + word.split("'")
    tokens = tokens + ls

    # Optionally, shorten words to their stems
    if lemma_words:
        tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]

    # Remove one letter tokens & non-alphabetic tokens, such as punctuation, then lower the tokens
    tokens = [word.lower() for word in tokens if (word.isalpha() and len(word) > 1)]
    final_stop=stopwords.words('english')
    # remove stop words
    if remove_stop_words:
        tokens = [word for word in tokens if word not in final_stop]

    return tokens
##################################################################################################################################
start = time.time()  # 시작 시간 저장

file_list=glob.glob('*.csv')

df_telegrph=pd.read_csv("2017-01-17_telegraph.csv", names=['press','author','date','setiment','title','paragraph'],engine='python', error_bad_lines=False)
df_btcnews=pd.read_csv("2017-1-17_btcnews.csv", names=['press','date','title','paragraph'],engine='python', error_bad_lines=False)
df_forbes=pd.read_csv("2021-2-25_forbes.csv", names=['press','author','date','setiment','title','paragraph'],engine='python', error_bad_lines=False)     ##csv 읽기

df_telegrph=df_telegrph.drop(['press','author', 'setiment'],axis=1)
df_btcnews=df_btcnews.drop(['press'],axis=1)
df_forbes=df_forbes.drop(['press','author', 'setiment'],axis=1)                          ##필요없는 열 제거


df_3news=pd.concat([df_telegrph,df_forbes,df_btcnews])                                      ##3개 언론사 합치기
df_3news['title']=df_3news['title']+" "                                                   ##sum시에 글자끼리 붙어버려서 공간 필요
df_3news['paragraph']=df_3news['paragraph']+" "
df_3news['date']=pd.to_datetime(df_3news['date'],dayfirst=True, errors='coerce')

print("중복제거 전:",df_3news)



df_3news.paragraph = df_3news.paragraph.apply(lambda x: text_to_wordlist(x))                ###본문 불용어 처리,토큰화
df_3news['paragraph']=[" ".join(review) for review in df_3news['paragraph'].values]
df_3news.title = df_3news.title.apply(lambda x: text_to_wordlist(x))                        ###제목 불용어 처리,토큰화
df_3news['title']=[" ".join(reviewz) for reviewz in df_3news['title'].values]

df_3news=df_3news.drop_duplicates()                                                                  ##중복값 제거
pd.set_option('display.max_rows',None)
print("중복제거 후:",df_3news)

grouped=df_3news.groupby(df_3news['date'],as_index=False)                   ##날짜 기준으로 그룹화
df_extraction=grouped.sum()
df_extraction['num_news']=grouped.count().title

print("날짜 그릅화 후 df.columns_extractions:",df_extraction.columns)
print("날짜 그릅화 후 df_extraction:",df_extraction)


df_extraction['title_tfidf'] = hero.tfidf(df_extraction['title'])                      #sum된 타이틀들이 tfidf
df_extraction['paragraph_tfidf'] = hero.tfidf(df_extraction['paragraph'])              #sum된 본문들의  tfidf
# df_extraction=df_extraction.drop(['tfidf'],axis=1)
df_extraction=df_extraction.sort_values(by=['date'],axis=0)
print(df_extraction.head())
print(df_extraction)
df_filtered =df_extraction.loc[df_extraction["date"].between('2018-08-01', '2021-03-15')]
print("날짜 처리 된 df :",df_filtered)


df_filtered.to_csv("extractedfile.csv", mode='w')
df_filtered.to_excel("extractedfile.xlsx")
print("time :", time.time() - start)                                                                # 현재시각 - 시작시간 = 실행 시간


