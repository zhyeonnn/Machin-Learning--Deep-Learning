#%%
# K-최근접 이웃 학습의 iris

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# 열 이름 설정 : 파일안에는 열 이름이 없어야 string으로 인한 오류가 발생하지 않는다.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pd.read_csv(r'C:\Users\813-18\Documents\sjh\sjh\2주차 딥러닝 기초 및 실습_자료\csv\iris.csv', names=names)
#%%
print(dataset.head())
print(dataset.iloc[:,:-1])
#%%
# 훈련과 검증 데이터셋 분리

X = dataset.iloc[:,:-1].values # 맨 뒤 1열 빼고 가져옴 = int
y = dataset.iloc[:,4].values # 맨 앞부터 4번째 열만 가져옴 = str
# iloc[행],[열]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)
# train_test_split : 훈련과 검증으로 분리할 것
#test_size : 훈련과 검증을 몇 퍼센트로 나눌것인지, test = 검증

from sklearn.preprocessing import StandardScaler
s = StandardScaler() #특성 스케일링
# 스케일링은 상대적인 데이터간의 격차를 줄여주는 것
# 상대적으로 숫자가 너무 작을 경우에 값이 표현이 안 되기 때문에 0와 1사이로 정해서 최대와 최소값을 정한다.
s.fit(X_train) # 학습 데이터 기준 평균, 표준편차 계산?? - 다시 알아보기 ★
# 평균이 0, 표준편차가 1이 되도록 변환
X_train = s.transform(X_train)
X_test = s.transform(X_test)
#%%
# 모델 생성 및 훈련

from sklearn.neighbors import KNeighborsClassifier
# k=50인 k-최근접 이웃 모델 생성
# k = 50은 거리상 가장 가까운 데이터 50개를 선택하여 50개중에 가장 많은 데이터 분류로 할당됨
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train,y_train) # 모델 훈련
print(knn.score(X_train,y_train)) # --> 이게 뭐에 대한 출력이지?
#%%
# 모델 정확도

from sklearn.metrics import accuracy_score
#predict() ★
y_pred = knn.predict(X_test)
print("정확도: {}".format(accuracy_score(y_test, y_pred)))
#%%
# 최적의 K 찾기

k = 10
acc_array = np.zeros(k)

for k in np.arange(1, k+1, 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    print("c: ",y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("정확도 : ",max_acc,"으로 최적의 k는 ",k+1,"입니다.")
# K=50일 때 정확도가 93%
# K=1일 때 정확도가 1(최대)로 100%로 높아졌다. K값에 따라 성능이 달라질 수 있음
#%%
# 서포트 벡터 머신의 iris

from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#'TF_CPP_MIN_LOG_LEVEL'를 사용하여 로깅을 제어
# 0 = 기본값
# 1 = 모든 로그 표시, info로그를 필터링
# 2 = WARNING 로그를 필터링