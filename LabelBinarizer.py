from sklearn.preprocessing import LabelBinarizer

y = [0,1,1,2]
y = LabelBinarizer().fit_transform(y)

print(y)