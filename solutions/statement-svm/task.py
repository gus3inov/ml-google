import numpy as np
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('datasets/svm-data.csv', header=None)

classes = df.iloc[:, 0]
signs = df.iloc[:, 1:3]

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(signs, classes)

print clf.support_ + 1