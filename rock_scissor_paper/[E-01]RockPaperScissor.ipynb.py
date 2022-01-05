from PIL import Image
import glob
import numpy as np
import os


def load_data(img_path, number_of_data=314):  
    # 가위 : 0, 바위 : 1, 보 : 2
    # 내가 만든 train 이미지는 총 314개이다.
    
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    
    imgs = np.zeros(number_of_data*img_size*img_size*color, dtype=np.int32).reshape(number_of_data, img_size, img_size, color)
    # 1-4. 딥러닝 네트워크 학습시키기에 나온 내용 (컬러 채널값(3)을 만드는 reshape를 실행해야한다.)
    # 스텝에서 첫 번째 레이어에 input_shape=(28,28,1)로 지정했던 것을 기억하시나요? 
    # 그런데 print(x_train.shape) 을 해보면,(60000, 28, 28) 로 채널수에 대한 정보가 없습니다. 
    # 따라서 (60000, 28, 28, 1) 로 만들어 주어야 합니다
    # 라고 적혀있네요. 
    # mnist에서는 흑백이라서 1이었지만, 지금 이미지는 컬러이기 때문에 3
    # (314, 28, 28, 3) : 28*28 컬러사진 314장
    
    labels = np.zeros(number_of_data, dtype=np.int32)
    # np.zeros() : 0으로 초기화 된 shape 차원의 ndarray 배열 객체를 반환
    # dtype=np.int32 : ndarray에 담긴 데이터 티입을 정수형 숫자로 지정
    # https://rfriend.tistory.com/285 에서 참고함

    idx=0
    
    for file in glob.iglob(img_path+'/img/train/scissor/*.jpg'):    
        # glob.iglob() :실제로 동시에 저장하지 않고 glob()과 같은 값을 산출하는 이터레이터(반복가능한 객체)를 반환
        # glob는 해당 디렉토리의 파일명을 리스트 형식으로 반환한다.
        # * 는 모든 파일을 지정
        
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1
    # 가위 전체 파일을 for문으로 반복해서 열어서, 0 으로 분류    

    for file in glob.iglob(img_path+'/img/train/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    # 바위 전체 파일을 for문으로 반복해서 열어서, 0 으로 분류 
    
    for file in glob.iglob(img_path+'/img/train/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
     # 보 전체 파일을 for문으로 반복해서 열어서, 0 으로 분류 
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.") 
    # idx가 0으로 시작해서 for문을 세 번 돌고나서 +1씩 쌓인 누적 숫자 = 이미지 파일의 갯수
                           
    return imgs, labels



image_dir_path = '/img/train' # train 데이터 폴더 전체 오픈

(x_train, y_train)=load_data(image_dir_path)
# 위에서 작성한 함수 load_data 함수에 넣어서 x_train은 imgs, y_train은 labels로 반환된다.
# imgs는 (314, 28, 28, 3)으로 reshape된 img 배열이다.
# labels -> 가위 : 0, 바위 : 1, 보 : 2

x_train_norm = x_train/255.0   
# 입력은 0~1 사이의 값으로 정규화 (rgb값이 0~255기 때문에 255로 나눠준다.)
# x 데이터만 정규화한다. y는 하지 않음


print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[4]:


plt.imshow(x_train[0])
print('라벨: ', y_train[0])


# In[13]:


import tensorflow as tf
from tensorflow import keras
import numpy as np

# model을 직접 만들어 보세요.
# Hint! model의 입력/출력부에 특히 유의해 주세요. 가위바위보 데이터셋은 MNIST 데이터셋과 어떤 점이 달라졌나요?
# [[YOUR CODE]]

# 이 값일 때, 평가 정확도가 가장 높게 나왔음
n_channel_1=64
n_channel_2=128
n_dense=32

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[14]:


model.fit(x_train_norm, y_train, epochs=30, validation_split=0.3)


# In[15]:


# x_test, y_test를 만드는 방법은 x_train, y_train을 만드는 방법과 아주 유사합니다.
# [[YOUR CODE]]

# 슬기님이 주신 테스트 데이터

import numpy as np

def load_data(img_path, number_of_data=334):  
    # 가위 : 0, 바위 : 1, 보 : 2
    # 슬기님의 사진은 334장
    
    img_size=28
    color=3
    
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    
    for file in glob.iglob(img_path+'/img/test/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/img/test/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/img/test/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_test)의 이미지 개수는", idx,"입니다.")
    
    return imgs, labels

image_dir_path = '/img/test'
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0  

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))

# In[16]:


# 모델 평가
model.evaluate(x_test_norm, y_test)


# In[ ]:





# In[ ]:




