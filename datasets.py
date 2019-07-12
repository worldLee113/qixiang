from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

iris = datasets.load_iris()
# 获取sklearn自带的图像
digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images)

X = digits.data
Y = digits.target
X_train,X_validition,Y_train,Y_validition = model_selection.train_test_split(X,Y,test_size=0.20,random_state=7)

models =[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names =[]
for name,result in models:
    cv_results = model_selection.cross_val_score(result,X_train,Y_train,cv=10,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)

KNN = KNeighborsClassifier()
KNN.fit(X_train,Y_train)
predictions = KNN.predict(digits.data[-1:])
# print(accuracy_score(Y_validition,predictions))
# print(confusion_matrix(Y_validition,predictions))
print(predictions)
# 绘图
print(digits.images[-1:])
images_and_predictions = list(zip(digits.images[-1:], predictions))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()