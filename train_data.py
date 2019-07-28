import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.externals import joblib
import os
training_dataset="./DATASET_DIR/"
model = "./MODEL_DIR"
if not os.path.exists(model):
    os.makedirs(model)

data = pd.read_csv(training_dataset+'train_data.csv')
data.date = pd.to_datetime(data.date)
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

data = data.drop('date',1)
X = np.array(data.drop(['temperature','humidity'],1))
Y = np.array(data[['temperature','humidity']])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
clf_a = GradientBoostingRegressor(learning_rate=0.1, max_features=0.1, n_estimators=700)
clf = MultiOutputRegressor(clf_a)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
print("Model save......")
save_model_path = model+"/checkpoint.m"
joblib.dump(clf,save_model_path)
print("Model save finish.")











