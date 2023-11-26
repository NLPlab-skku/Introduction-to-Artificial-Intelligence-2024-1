# 라이브러리를 임포트 합니다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# 데이터를 로드 합니다.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 표준화 객체를 만듭니다.
standardizer = StandardScaler()

# 특성을 표준화합니다.
X_std = standardizer.fit_transform(X)

# 5개의 이웃을 사용한 KNN 분류기 훈련
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)

# 두 개의 샘플을 만듭니다.
new_observations = [[0.75,0.75,0.75,0.75],[1,1,1,1]]

# 각 샘플의 세 클래스에 속할 확률을 확인
print(knn.predict_proba(new_observations))

# 두 샘플의 클래스를 예측
print(knn.predict(new_observations))