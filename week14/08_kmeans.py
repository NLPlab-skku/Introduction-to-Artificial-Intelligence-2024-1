from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 데이터 로드
iris = datasets.load_iris()
features = iris.data

# 특성을 표준화
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# k-평균 객체를 생성
cluster = KMeans(n_clusters=4,random_state=0, n_init='auto')

# 모델 훈련
model = cluster.fit(features_std)

# 클러스트 중심 확인
print(model.cluster_centers_)

# 새로운 샘플 생성
new_observation = [[0.8,0.8,0.8,0.8]]

# 샘플의 클러서트를 예측
print(model.predict(new_observation))