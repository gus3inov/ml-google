import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

(data, target) = load_boston(return_X_y=True)

target_scales = scale(data)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc = None
result_p = 0

for p in np.linspace(1, 10, num=200):
    neigh = KNeighborsRegressor(
        n_neighbors=5,
        metric="minkowski",
        weights="distance",
        p=p)
    score = cross_val_score(neigh, target_scales, target, scoring='neg_mean_squared_error', cv=kf)
    mean_score = np.mean(score)
    if acc is None:
        acc = mean_score
    if mean_score > acc:
        acc = mean_score
        result_p = p

print acc, result_p