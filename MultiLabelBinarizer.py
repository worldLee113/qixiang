from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
print(mlb .fit_transform([(1, 2), (3,),(4,),(1,3)]))
print(mlb)
print(mlb.classes_)

print(mlb.fit_transform([{'python','C++'},{'JAVA'}]))
print(mlb.classes_)