from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None
data = pd.read_csv('bd2019-weather-prediction-training-20190608.csv')
data = data[~data['rain20'].isin([999990])]
data = data[~data['rain08'].isin([999990])]
data = data[~data['wind_speed'].isin([999999])]
data = data[~data['wind_direction'].isin([999999])]
# data = data[~data['cloud'].isin([999999])]
data = data[~data['visibility'].isin([999999])]
data = data[~data['temperature'].isin([999999])]
data = data[~data['humidity'].isin([999999])]
data = data[~data['pressure'].isin([999999])]
data['date'] = pd.to_datetime(data['date'])
# data_201 = data[data['station'].isin(['201'])]
data_201 = data
# print (data_201.corr().round(2))



data_201 = data_201[['date','humidity','temperature','rain20','phenomenon']]
data_201.set_index('date',inplace=True)
data_201.reset_index(inplace=True)
# data_201['year'] = data_201['date'].dt.year
# data_201['month'] = data_201['date'].dt.month
# data_201['day'] = data_201['date'].dt.day


# data_201['sum_rain20']=data_201.groupby(['month'])['rain20'].cumsum()
# print(data_201.head(60))
# pyplot.bar(data_201.month.head(1200),data_201.sum_rain20.head(1200))
# pyplot.show()
data_201= data_201.drop('date',1)

X = np.array(data_201.drop(['humidity','temperature','rain20'],axis=1))
Y = np.array(data_201['rain20'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf_rain = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=700)
clf_rain.fit(X_train, Y_train)
accuracy = clf_rain.score(X_test, Y_test)
print(accuracy)
# start_year = 2018
# start_mouth = 1
# predict_X = []
# for start_day in range(1, 8):
#     predict_X.append([start_year, start_mouth, start_day])
# predict_Y = clf_rain.predict(predict_X)
# print(predict_Y)