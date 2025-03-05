#라이브러리를 임포트 
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets


#데이터를 로드
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 결정 트리 분류기 생성
decisiontree = DecisionTreeClassifier(random_state=0)

#모델 훈련
model = decisiontree.fit(features, target)

#새로운 샘플 생성
observation = [[5, 4, 3, 2]]

#세개의 클래스 예측 확률 
print(model.predict_proba(observation))

# 샘플의 클래스 예측
print(model.predict(observation))

# 결정 트리 분류기 생성
decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)

#모델 훈련
model_entropy = decisiontree_entropy.fit(features, target)

#세개의 클래스 예측 확률 
print(model_entropy.predict_proba(observation))

# 샘플의 클래스 예측
print(model_entropy.predict(observation))
