import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, newsgroups.target)

C_best = gs.best_params_['C']

cv2 = KFold(n_splits=5, shuffle=True, random_state=241)
clf2 = SVC(kernel='linear', C=C_best, random_state=241)
clf2.fit(X, newsgroups.target)

coefs = clf2.coef_.todense()
mapped_coefs = np.abs(np.asarray(coefs)).reshape(-1)
word_indexes = np.argsort(mapped_coefs)[-10:]

words = [feature_mapping[i] for i in word_indexes]

print words