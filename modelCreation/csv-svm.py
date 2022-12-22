import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, recall_score, precision_score
from sklearn.svm import SVC
import pickle
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv(r"/Users/harikrish/Desktop/codes and csv file (1)/dataset.csv")
df.columns = [i for i in range(df.shape[1])]
df = df.rename(columns={42: 'Output'})

print("Uncleaned dataset shape =", df.shape)

all_null_values = df[df.iloc[:, 0] == 0]
print("Number of null values =", len(all_null_values.index))

df.drop(all_null_values.index, inplace=True)

print("Cleaned dataset shape =", df.shape)

X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:,-1]
print("Labels shape =", Y.shape)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

svm = SVC(C=50, gamma=0.1, kernel='rbf')
svm.fit(x_train, y_train)

print("Training score =", svm.score(x_train, y_train))

y_pred = svm.predict(x_test)
print("Testing score =", accuracy_score(y_test, y_pred))

plot_confusion_matrix(svm,x_test,y_test)
cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
print(f1, recall, precision)

filename = r'/Users/harikrish/Desktop/codes and csv file (1)/broo.sav'
pickle.dump(svm, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)