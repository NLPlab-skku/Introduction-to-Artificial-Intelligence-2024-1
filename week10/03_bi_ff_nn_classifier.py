import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers

# 랜덤 시드 설정
np.random.seed(0)

# 필요한 특성 개수를 지정
number_of_features = 1000

# 영화 리뷰 데이터에서 훈련데이터와 타깃 벡터를 로드
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

# 영화 리뷰 데이터를 원-핫 인코딩된 특성 행렬로 변환
Tokenizer = Tokenizer(num_words=number_of_features)
features_train = Tokenizer.sequences_to_matrix(data_train, mode='binary')
features_test = Tokenizer.sequences_to_matrix(data_test, mode='binary')

# 신경망 모델 생성
network = models.Sequential()

# Relu activation function을 사용한 완전 연결층 추가
network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))

# 2번째 layer
network.add(layers.Dense(units=16, activation='relu'))

# 3번째 layer with sigmoid activation
network.add(layers.Dense(units=1, activation='sigmoid'))

# 신경망 모델 설정 완료
network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 신경망 훈련
history = network.fit(features_train, target_train, epochs=3, verbose=1, batch_size=100, validation_data=(features_test, target_test))

print(features_train.shape)
network.evaluate(features_test, target_test)