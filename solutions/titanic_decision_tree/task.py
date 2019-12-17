import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('datasets/titanic.csv', index_col='PassengerId')

constraint_data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
constraint_data['Sex'] = constraint_data['Sex'].map({'male': 1, 'female': 0})
target_var = constraint_data['Survived']
constraint_data = constraint_data.drop(['Survived'], axis=1)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(constraint_data, target_var)

importances = clf.feature_importances_

print importances
