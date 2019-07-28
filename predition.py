import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

model="./MODEL_DIR/"
test_dataset="./TESTSET_DIR/"
prediction_file="./PREDICTION_FILE/"
clf = joblib.load(model+"checkpoint.m")
data = pd.read_csv(test_dataset+'test_data.csv')

data.date = pd.to_datetime(data.date)
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

data = data.drop('date',1)
X = np.array(data.drop(['temperature','humidity'],1))
Y = np.array(data[['temperature','humidity']])

prediction = clf.predict(X)
print(prediction)
predictions = cross_val_score(clf,Y,prediction, cv=5)
print(predictions)
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
        x = pd.Timestamp(start_year, start_mouth, start_day)
        XXX.append(x)
predict_Y = clf.predict(predict_X)
print(XXX)

a = pd.DataFrame({'date': XXX, 'station':YYY,'temperature': predict_Y[:,0],'humidity':predict_Y[:,1]})
a.to_csv(prediction_file+"predict_result.csv",index=False)
