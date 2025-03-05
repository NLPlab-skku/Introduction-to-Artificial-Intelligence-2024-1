import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers


# 랜덤 시드 설정
np.random.seed(0)

# 필요한 특성 개수를 지정
number_of_features = 5000

# 특성과 타깃 데이터 로드
data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data

# 특성 데이터를 원-핫 인코딩된 특성 행렬로 변환
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode='binary')
features_test = tokenizer.sequences_to_matrix(data_test, mode='binary')

# 타깃 벡터를 원-핫 인코딩하여 타깃 행렬 생성
target_train = to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)

# 신경망 모델 생성
network = models.Sequential()

# Relu activation function을 사용한 완전 연결층 추가
network.add(layers.Dense(units=100, activation='relu', input_shape=(number_of_features,)))

# 2번째 layer
network.add(layers.Dense(units=100, activation='relu'))

# 3번째 layer with sigmoid activation
network.add(layers.Dense(units=46, activation='softmax'))

# 신경망 모델 설정 완료
network.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 신경망 훈련
history = network.fit(features_train, target_train, epochs=3, verbose=0, batch_size=100, validation_data=(features_test, target_test))

print(target_train)
network.evaluate(features_test, target_test)
