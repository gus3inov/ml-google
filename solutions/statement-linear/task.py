import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('datasets/percepton/train.csv', header=None)
df_test = pd.read_csv('datasets/percepton/test.csv', header=None)

target_train = df_train.iloc[:, 0]
signs_train = df_train.iloc[:, 1:3]

target_test = df_test.iloc[:, 0]
signs_test = df_test.iloc[:, 1:3]

clf = Perceptron(random_state=241)

clf.fit(signs_train, target_train)

predictions = clf.predict(signs_test)

score_test = accuracy_score(target_test, predictions)

scaler = StandardScaler()

signs_train_scaled = scaler.fit_transform(signs_train)
signs_test_scaled = scaler.transform(signs_test)

clf = Perceptron(random_state=241)

clf.fit(signs_train_scaled, target_train)

predictions_normalized = clf.predict(signs_test_scaled)

score_test_normalized = accuracy_score(target_test, predictions_normalized)

print score_test_normalized - score_test