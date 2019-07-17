import pandas as pd
import numpy as np
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
data = data[~data['cloud'].isin([999999])]
data = data[~data['visibility'].isin([999999])]
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
dataA = data[data['city'].isin(['A'])]
dataB = data[data['city'].isin(['B'])]
dataC = data[data['city'].isin(['C'])]
dataD = data[data['city'].isin(['D'])]
dataE = data[data['city'].isin(['E'])]
dataF = data[data['city'].isin(['F'])]
dataG = data[data['city'].isin(['G'])]
dataH = data[data['city'].isin(['H'])]
dataI = data[data['city'].isin(['I'])]
dataJ = data[data['city'].isin(['J'])]
# print(dataG['pressure'].head(120))
x = dataG[['temperature']].resample('1D',closed='left').mean()

x.dropna(axis=0, how='any', inplace=True)
x.reset_index(inplace=True)

# 把date字段转为日期
x.date = pd.to_datetime(x.date)
# 把日期分成年月日
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x = x.drop('date', 1)
X = np.array(x.drop(['temperature'], 1))
Y = np.array(x['temperature'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=500)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)  # 0.965156260089

# plt.plot(x.index,x['visibility'],'ro')
# plt.show()

# plt.axes(polar = True)
# theta = dataA['wind_direction'].head(365)
# radii = dataA['temperature'].head(365)
#
#
# plt.bar(theta,radii, width=(1/10))
# plt.show()






