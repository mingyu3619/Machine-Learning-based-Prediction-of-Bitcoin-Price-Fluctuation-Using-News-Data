from sklearn import svm, naive_bayes, neighbors, ensemble
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xlrd
import os
from sklearn.preprocessing import LabelEncoder
import openpyxl
from sklearn import preprocessing
#######################################################################################################################
models = ["lr_model", "nb_model", "knn_model", "svc_model", "rf_model", "et_model", "ada_model"]


def baseline_model_filter(modellist, X, y):
    ''' 1. split the train data further into train and validation (17%).
        2. fit the train data into each model of the model list
        3. get the classification report based on the model performance on validation data
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.17,random_state=777)
    for model_name in modellist:
        curr_model = eval(model_name)
        curr_model.fit(X_train, y_train.values.ravel())
        print(f'{model_name} \n report:{classification_report(y_valid, curr_model.predict(X_valid))}')

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
#######################################################################################################################
# df_x= pd.read_excel(
#      os.path.join("C:\\Users\\d\\PycharmProjects\\", "news_predict_experiment", "extractedfile.xlsx"),
#      engine='openpyxl')
df_y = pd.read_csv("BTC_USD_2013-10-01_2021-03-21-CoinDesk.csv",encoding='utf-8')
df_x=pd.read_csv("extractedfile.csv",encoding='utf-8')
print(df_x)
print(df_x.columns)
print(df_x.dtypes)



print(type(df_x.title_tfidf))
print(type(df_x.title_tfidf[0]))
print(df_y)
print(df_y.columns)

df_x['rise']=df_y['rise']
TEST_SIZE = 7
train = df_x[:-TEST_SIZE]
test = df_x[-TEST_SIZE:]

feature_cols = ['num_news', 'title_tfidf','paragraph_tfidf']
label_cols = ['rise']
train_feature = train[feature_cols]
train_label = train[label_cols]

#x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

test_feature = test[feature_cols]
test_label = test[label_cols]                                                                           ####train test나누는 과정,valid 또한!
arr_all=np.array([])
#arr_all=[]

for y in range(len(train_feature)):
    tmp_str=[]
    tmp_str = train_feature['title_tfidf'][y].replace("[","")
    tmp_str = tmp_str.replace("]","")
    #print(tmp_str)
    arr = np.array([])
    for i in (tmp_str.split(",")):
        #print(i,type(i))
        arr.append(float(i))
    arr_all.append(arr)
#print(arr_all)
arr_to_np=np.array(arr_all)
train_feature['title_tfidf']=pd.DataFrame(arr_to_np)

#arr_all=[]
arr_all=np.array([])
for y in range(len(train_feature)):
    tmp_str=[]
    tmp_str = train_feature['paragraph_tfidf'][y].replace("[","")
    tmp_str = tmp_str.replace("]","")
    #print(tmp_str)
    arr = np.array([])
    for i in (tmp_str.split(",")):
        #print(i,type(i))
        arr.append(float(i))
    arr_all.append(arr)
#print(arr_all)
arr_to_np=np.array(arr_all)
train_feature['paragraph_tfidf']=pd.DataFrame(arr_all)
print('pd.title_tfidf.shape:',train_feature.shape)
print('pd.title_tfidf[0].shape:',train_feature[0].shape)
lr_model = LogisticRegression()
nb_model = naive_bayes.GaussianNB()
knn_model = neighbors.KNeighborsClassifier()
svc_model = svm.SVC(probability=True, gamma="scale")
rf_model = ensemble.RandomForestClassifier(n_estimators=100)
et_model = ensemble.ExtraTreesClassifier(n_estimators=100)
ada_model = ensemble.AdaBoostClassifier()

print(train_feature['title_tfidf'].shape)
print(train_feature['title_tfidf'][0].shape)
print(train_feature['paragraph_tfidf'].shape)
print(train_feature['paragraph_tfidf'][0].shape)


#baseline_model_filter(models,train_feature,train_label)

