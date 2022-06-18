from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score

obesity = pd.read_csv("C:/Users/krzys/Downloads/obesity1.csv", usecols=[i for i in range(0,17)], header=0)
feature_cols = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
X = obesity[feature_cols]
Y = obesity.NObeyesdad

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.2, random_state = 42)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf=rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)

scores = cross_val_score(rf,X,Y,cv=5)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
