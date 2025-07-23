
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics


digits = load_digits()
# 데이터셋의 형태 확인
print("image data shape",digits.data.shape) #1797개의 이미지와 64차원
print("label data shape", digits.target.shape) # 1797개의 이미지
# digits.target = 해당 이미지에 해당하는 정답 숫자

# figure(figsize = (20,4)) = 새로운 도화지의 사이즈(20,4) 생성
plt.figure(figsize = (20,4))

# 0번째부터 5번째의 이미지(data)와 숫자 정답값(target)
# index는 정의만해서 0부터
# subplot : 그림위에 작은 여러 개의 그림은 나눠 그릴 수 있다.
# reshape : 변형할 이미지, 원하는 배열의 크기를 넣으면 배열을 바꿔줌
# i는 정수를 가져오는 것(%s와 같음)
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n'%label,fontsize=20)
# plt.show()

# 훈련 데이터와 테스트 데이터 분리
# test_size = 테스트 데이터를 전체 데이터의 25%사용하겠다는 의미
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.25, random_state=0 )
logisticRegr = LogisticRegression()
# 모델 훈련
logisticRegr.fit(x_train, y_train)
print(logisticRegr.fit(x_train, y_train))

# 모델 예측
# reshape : (샘플 수, 특성 수)로 출력, -1은 자동 계산 / (1,64)즉, 1개의 64차원에 대한 예측을 해줌
numpyArrayPrint = logisticRegr.predict(x_test[0].reshape(1,-1))
# 이미지 10개에 대한 예측을 한번에 출력
arrayPrint = logisticRegr.predict(x_test[0:10])

print(numpyArrayPrint)
print(arrayPrint)

# 모델 성능을 측정하는 방법 = 혼동행렬(정확도, 정밀도, 재현율)
# 전체 데이터에 대한 예측(정답인 y_test제외하고 예측)
predictions = logisticRegr.predict(x_test)
# score를 사용한 성능 측정 = 정확도
score = logisticRegr.score(x_test, y_test)
print("score : ",score)

# 혼동행렬(confusion_matrix)
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
#heatmap으로 표현
sns.heatmap(cm, annot=True, fmt= ".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size = 15);
plt.show()
