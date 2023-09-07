import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, parser = 'auto', as_frame=False)
print(mnist.keys())

X, y = mnist["data"], mnist["target"]
print(X.shape, y.shape)

#integer로 타입을 바꾸기
y = y.astype(np.uint8)

#학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print('y_train :', y_train)
print('y_test  :', y_test)

# binary classifier 구축하기
y_train_5 = (y_train == 5)
print('y_train_5 :', y_train_5)

y_test_5 = (y_test == 5)
print('y_test_5  :', y_test_5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([X_test[0],X_test[9998]]))
