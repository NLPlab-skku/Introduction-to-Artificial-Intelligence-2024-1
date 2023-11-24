from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

iris = datasets.load_iris()

features = iris.data
target = iris.target

# 랜덤 포레스트 분류기 객에 생성
randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)

# 모델을 훈련합니다.
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

# 모델을 훈련합니다
model = randomforest.fit(features, target)

# 새로운 샘플 생성
observation = [[5,4,3,2]]

# 샘플 클래스를 예측합니다
print(model.predict(observation))

# 랜덤 포레스트 분류기 객에 생성
randomforest_entropy = RandomForestClassifier(criterion='entropy',random_state=0)

# 모델을 훈련합니다
model_entropy = randomforest_entropy.fit(features, target)

# 샘플 클래스를 예측합니다
print(model_entropy.predict(observation))

