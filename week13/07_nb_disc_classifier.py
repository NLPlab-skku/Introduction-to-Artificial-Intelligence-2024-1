import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 텍스트 생성
text_data = np.array(['I love Brazil. Brazil!', 'Brazil is best', 'Germany beats both'])

# BOW 생성
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# 특성 행렬 생성
features = bag_of_words.toarray()
print(features)

# 타깃 벡터 생성
target = np.array([0,0,1])

# 각 클래스별 사전 확률 지정한 다항 NB 객체 생성
classifier = MultinomialNB(class_prior=[0.5,0.5])

# 모델 훈련
model = classifier.fit(features, target)

# 새로운 샘플 생성
new_observation = [[0,0,0,1,0,1,0]]

# 클래스 예측
print(model.predict(new_observation))