import pandas as pd
from sklearn import preprocessing
import os

test_dataset="./TESTSET_DIR/"
training_dataset="./DATASET_DIR/"
if not os.path.exists(test_dataset):
    os.makedirs(test_dataset)
if not os.path.exists(training_dataset):
    os.makedirs(training_dataset)

data = pd.read_csv('bd2019-weather-prediction-training-20190608.csv')

print(data.isnull().sum())
data.fillna("00,", inplace = True)

data['date'] = pd.to_datetime(data['date'])
data = data[~data['rain20'].isin([999990])]
data = data[~data['rain08'].isin([999990])]
data = data[~data['wind_speed'].isin([999999])]
data = data[~data['wind_direction'].isin([999999])]
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

data = data[['date','station','temperature','humidity']]
data.date = pd.to_datetime(data.date)
data.set_index('date',inplace=True)

train_data = data.truncate(after='2017-12-01')
test_data = data.truncate(before='2017-12-02')
train_data.reset_index(inplace=True)
test_data.reset_index(inplace=True)
train_data.to_csv('train_data.csv',index = False)
test_data.to_csv('test_data.csv',index=False)