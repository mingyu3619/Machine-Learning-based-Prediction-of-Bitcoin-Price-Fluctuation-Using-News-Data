# Machine-Learning-based-Prediction-of-Bitcoin-Price-Fluctuation-Using-News-Data

2020 knom conference에 기재된 '뉴스 데이터를 이용한 머신러닝 기반 비트코인 가격 등락 예측'의 파일들입니다.
뉴스 데이터를 이용해 비트코인의 가격을 예측하는 논문으로 이 과정에서 TF-IDF를 통해 뉴스 기사들을 벡터화 시켜 머신러닝 모델의 
input으로 사용하고 

로지스틱 회귀,
나이브 베이즈,
KNN,
SVM,
랜덤 포레스트,
Extra Tree,
AdaBoost,
XGBoost

8개의 머신러닝 모델에 사용하여 성능을 비교합니다. 

# 개요 

데이터 수집 - 데이터 전처리 - 실험  -결론 순으로 구성되어 있습니다.

# 데이터 수집

 “LSTM 기반 감성분석을 이용한 비트코인 가격 등락 예측” 2020년도 한국통신학회 추계종합학술발표회, 온라인 개최, Nov. 13, 2020, pp. 1-2
의 뉴스 순위에 따라 2013~2020년 사이의 비트코인 관련 영항력있는 4개의 언론사에서 뉴스기사를 크롤링합니다.

(crawling 폴더의 4개의 python file들)

#데이터 전처리

TF-IDF: 정보 검색론 분야에서 사용하는 가중치를 구하는 알고리즘으로, 단어 빈도 수를 기반으로 가중치를 구한다.
![image](https://user-images.githubusercontent.com/86222639/146505868-b399301b-36d4-40c8-b9d4-ba7c497e5d01.png)

뉴스 기사를 머신러닝 모델의 input으로 활용하기 위해 TF-IDF 알고리즘을 이용하며 텍스트형태의 기사들을 벡터화시켜줍니다.
(TF-IDF 폴더의 tf-idf.py 를 통해 [통해 뉴스 기사 수, 제목의 TF-IDF값, 본문의 TF-iDF값]  추출)

추출된 값 예시:
![image](https://user-images.githubusercontent.com/86222639/146506689-4e3515f7-72b5-4b83-888e-19c6da51a5a2.png)

# 머신러닝 실험

로지스틱 회귀,나이브 베이즈,KNN,SVM,랜덤 포레스트,Extra Tree,AdaBoost,XGBoost 모델을 사용해 학습

(machine learning/experiment.py 를 통해 싫험)

# 결과

![image](https://user-images.githubusercontent.com/86222639/146506937-1ebc790e-2404-483c-b2ae-46b09ddffdac.png)

모델 평가 결과, 입력 값으로 전날의 종가만을 준 모델들은 40% 후반 대에서 50% 후반까지 고르게 분포한 모습을 보여주었으며, XGBoost 모델이 가장 좋은 정확도인 60%를 보여주었다. 

또한 입력 값으로 가격 데이터와 함께 뉴스 데이터(기사 수,제목의 TF-IDF값,본문의 TF-IDF값)를 넣어 주었을 때 나이브 베이즈를 제외한 모든 모델에서 precision, recall, f1-score, accuracy가 소폭 상승하거나 유지했다


