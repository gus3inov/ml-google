import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import pandas as pd

df = pd.read_csv('datasets/wine/wine.data')

classes = df.iloc[:, 0]
signs = scale(df.iloc[:, 1:14])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc = 0
result_k = 0

for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(neigh, signs, classes, cv=kf)
    mean_score = np.mean(score)
    if mean_score > acc:
        acc = mean_score
        result_k = k

print result_k, acc
