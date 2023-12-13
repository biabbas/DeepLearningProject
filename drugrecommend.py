import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
filename='Drug.csv'
data=read_csv(filename)

print(data.isnull().sum())
data.replace({'Gender':{'Female':0,'Male':1}},inplace=True)

x=data[['Disease']]
x.Disease.unique()
data.replace({'Disease':{'Acne':0, 'Allergy':1, 'Diabetes':2, 'Fungal infection':3,
       'Urinary tract infection':4, 'Malaria':5, 'Migraine':6, 'Hepatitis B':7,
       'AIDS':8}},inplace=True)

df_x=data[['Disease','Gender','Age']]
df_y=data[['Drug']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(df_x,np.ravel(df_y))

from sklearn.metrics import accuracy_score
y_pred=rf.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))

print(rf.score(x_test,y_test))

prediction=rf.predict(x_test)
print(prediction[0:10])

test=[5,1,24]
test=np.array(test)
test=np.array(test).reshape(1,-1)
print(test.shape)

prediction=rf.predict(test)
print(prediction[0])

import joblib as joblib
joblib.dump(rf,'healthcare/model/medical_rf.pkl')

clf=joblib.load('healthcare/model/medical_rf.pkl')

prediction=clf.predict(test)
print(prediction[0])

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb=gnb.fit(df_x,np.ravel(df_y))

y_pred=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred,normalize=False))

gnb.score(x_test,y_test)

result=gnb.predict(test)
print(result[0])

joblib.dump(gnb,'healthcare/model/medical_nb.pkl')