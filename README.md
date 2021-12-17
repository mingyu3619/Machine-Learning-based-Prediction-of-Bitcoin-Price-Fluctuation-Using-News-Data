# Machine-Learning-based-Prediction-of-Bitcoin-Price-Fluctuation-Using-News-Data

2020 knom conference에 기재된 '뉴스 데이터를 이용한 머신러닝 기반 비트코인 가격 등락 예측'의 파일들입니다.
뉴스 데이터를 이용해 비트코인의 가격을 예측하는 논문으로 이 과정에서 TF-IDF를 통해 뉴스 기사들을 벡터화 시켜 머신러닝 모델의 
input으로 사용하고 

로지스틱 회귀
나이브 베이즈
KNN
SVM
랜덤 포레스트
Extra Tree
AdaBoost
XGBoost

8개의 머신러닝 모델에 사용하여 성능을 비교합니다. 

# 개요 

데이터 수집 - 데이터 전처리 - 실험  -결론 순으로 구성되어 있습니다.

# 데이터 수집

강민규, 김보선, 신무곤, 백의준, 김명섭, “LSTM 기반 감성분석을 이용한 비트코인 가격 등락 예측” 2020년도 한국통신학회 추계종합학술발표회, 온라인 개최, Nov. 13, 2020, pp. 1-2
의 뉴스 순위에 따라 2013~2020년 사이의 비트코인 관련 영항력있는 4개의 언론사에서 뉴스기사를 크롤링합니다.

(crawling 폴더의 4개의 python file들)

#데이터 전처리



