import pandas
import numpy as np

data = pandas.read_csv('datasets/titanic.csv', index_col='PassengerId')

# ----- 1 -----

sex = data['Sex']
sex_counts = sex.value_counts()
male = sex_counts[0]
female = sex_counts[1]

print(male, female)

# ----- 2 -----

sv = data['Survived']
sv_counts = sv.value_counts()
svf = sv_counts[0]
svt = sv_counts[1]
svt_sum = svf + svt

print svt / (svt_sum * 0.01)

# ----- 3 -----

pc = data['Pclass']
pc_counts = pc.value_counts()
pc_third = pc_counts[3]
pc_second = pc_counts[2]
pc_first = pc_counts[1]
pc_sum = pc_first + pc_second + pc_third

print pc_first / (pc_sum * 0.01)

# ----- 4 -----

age = data['Age']
average_age = age.mean()
median_age = age.median()

print(average_age, median_age)

# ----- 5 -----

depends = data[['SibSp', 'Parch']]
corr_depends = depends.corr(method="pearson")

print corr_depends

# ----- 6 -----

females = data.loc[data['Sex'] == 'female']
names = females['Name'].str.split(" ", n=3, expand=True)
first_names = names[2].value_counts()

print first_names
