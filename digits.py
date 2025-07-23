
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()