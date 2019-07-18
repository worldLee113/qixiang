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
data.set_index('date', inplace=True)

stationame = data['station'].unique()
print(data['station'].unique())
b = []
for i in stationame:
    x = data[data['station'].isin([i])]
    x = x[['temperature']]
    # print(x)
    x.reset_index(inplace=True)
    # 把date字段转为日期
    x.date = pd.to_datetime(x.date)
    # 把日期分成年月日
    x['year'] = x['date'].dt.year
    x['month'] = x['date'].dt.month
    x['day'] = x['date'].dt.day
    x = x.drop('date', 1)
    X = np.array(x.drop(['temperature'],axis=1))
    Y = np.array(x['temperature'])

    # 模型评估效果
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    clf = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=700)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    print(i)
    print(accuracy)  # 0.965156260089
    predict_X = []
    stations = []
    XXX = []

    start_year = 2018
    start_mouth = 1
    for start_day in range(1,31):
        predict_X.append([start_year, start_mouth,start_day])
        stations.append(i)
        XXX.append(str(start_year) + '-' + str(start_mouth) + '-' + str(start_day))
    XXX = pd.to_datetime(XXX)
    pridict_Y = clf.predict(predict_X)
    a = pd.DataFrame({'date': XXX, 'station':stations,'temperature': pridict_Y})
    b.append(a)
result = pd.concat(b,axis=0,ignore_index=True)
print( pd.concat(b,axis=0,ignore_index=True))
result.to_csv("predict_result.csv",index=False)

# dataA = data[data['city'].isin(['A'])]
# dataB = data[data['city'].isin(['B'])]
# dataC = data[data['city'].isin(['C'])]
# dataD = data[data['city'].isin(['D'])]
# dataE = data[data['city'].isin(['E'])]
# dataF = data[data['city'].isin(['F'])]
# dataG = data[data['city'].isin(['G'])]
# dataH = data[data['city'].isin(['H'])]
# dataI = data[data['city'].isin(['I'])]
# dataJ = data[data['city'].isin(['J'])]
# # print(dataG['pressure'].head(120))
# x = dataG[['temperature']].resample('1D',closed='left').mean()
# print(x)
# x.dropna(axis=0, how='any', inplace=True)
# x.reset_index(inplace=True)
# plt.plot(x.visibility.head(365),x.humidity.head(365),'ro')
#
# plt.show()
#
# 把date字段转为日期
# x.date = pd.to_datetime(x.date)
# # 把日期分成年月日
# x['year'] = x['date'].dt.year
# x['month'] = x['date'].dt.month
# x['day'] = x['date'].dt.day
# x = x.drop('date', 1)
# X = np.array(x.drop(['temperature'], 1))
# Y = np.array(x['temperature'])
#
# # 模型评估效果
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
# clf = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=500)
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy)  # 0.965156260089

# pridict_Y = clf.predict(X_test)








