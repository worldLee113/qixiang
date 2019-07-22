import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv('bd2019-weather-prediction-training-20190608.csv')

print(data.isnull().sum())
data.fillna("00,", inplace = True)

data['date'] = pd.to_datetime(data['date'])
data = data[~data['rain20'].isin([999990])]
data = data[~data['rain08'].isin([999990])]
data = data[~data['wind_speed'].isin([999999])]
data = data[~data['wind_direction'].isin([999999])]
# data = data[~data['cloud'].isin([999999])]
data = data[~data['visibility'].isin([999999])]
data = data[~data['temperature'].isin([999999])]
data = data[~data['humidity'].isin([999999])]
data['cloud'] = preprocessing.maxabs_scale(data['cloud'])
data['wind_direction'].replace(999001,0,inplace= True)
data['wind_direction'].replace(999002,22.5,inplace= True)
data['wind_direction'].replace(999003,45,inplace= True)
data['wind_direction'].replace(999004,67.5,inplace= True)
data['wind_direction'].replace(999005,90,inplace= True)
data['wind_direction'].replace(999006,112.5,inplace= True)
data['wind_direction'].replace(999007,135,inplace= True)
data['wind_direction'].replace(999008,157.5,inplace= True)
data['wind_direction'].replace(999009,180,inplace= True)
data['wind_direction'].replace(999010,202.5,inplace= True)
data['wind_direction'].replace(999011,225,inplace= True)
data['wind_direction'].replace(999012,247.5,inplace= True)
data['wind_direction'].replace(999013,270,inplace= True)
data['wind_direction'].replace(999014,292.5,inplace= True)
data['wind_direction'].replace(999015,315,inplace= True)
data['wind_direction'].replace(999016,315,inplace= True)
data['wind_direction'].replace(999017,0,inplace= True)
# data.set_index('date', inplace=True)

data = data[['date','station','temperature','humidity']]
data.date = pd.to_datetime(data.date)
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data.drop('date',1)
X = np.array(data.drop(['temperature','humidity'],1))
Y = np.array(data[['temperature']])
Y1 = np.array(data[['humidity']])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=700)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)  # 0.965156260089
start_year = 2018
start_mouth = 1
predict_X = []
XXX = []
YYY = []
for start_day in range(1, 8):
    stationame = data['station'].unique()
    for i in stationame:
        predict_X.append([i,start_year, start_mouth, start_day])
        YYY.append(i)
        XXX.append(str(start_year) + '-' + str(start_mouth) + '-' + str(start_day))

predict_X = np.array(predict_X)
predict_Y = clf.predict(predict_X)
print(predict_Y)

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y1, train_size=0.8, test_size=0.2)
clf_H = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=700)
clf_H.fit(X_train1, Y_train1)
accuracy = clf_H.score(X_test1, Y_test1)
print(accuracy)  # 0.965156260089
predict_X = np.array(predict_X)
predict_Y111 = clf_H.predict(predict_X)

a = pd.DataFrame({'date': XXX, 'station':YYY,'temperature': predict_Y,'humidity':predict_Y111})

a.to_csv("predict_result.csv",index=False)










