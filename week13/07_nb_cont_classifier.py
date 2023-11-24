from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 데이터 로드
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 가우시안 나이브 베이즈 객체 생성
#classifier = GaussianNB()
classifier = GaussianNB(priors=[0.25,0.25,0.5])

# 모델 훈련
model = classifier.fit(features, target)

# 새로운 샘프
new_observation = [[4, 4, 4, 0.4]]

# 클래스 예측
print(model.predict(new_observation))