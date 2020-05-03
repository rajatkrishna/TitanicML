import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
import re

trainset = pd.read_csv("train.csv")
feat = trainset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = False)

tar = trainset['Survived']

train_age_mean = feat['Age'].mean()
feat['Sex'][feat['Sex'] == 'male'] = 1
feat['Sex'][feat['Sex'] == 'female'] = 2
feat['Sex'][feat['Sex'].isnull()] = 0
feat['Embarked'][feat['Embarked'] == 'S'] = 1
feat['Embarked'][feat['Embarked'] == 'C'] = 2
feat['Embarked'][feat['Embarked'] == 'Q'] = 3
feat['Embarked'][feat['Embarked'].isnull()] = 0
feat['Age'][feat['Age'].isnull()] = 0

testset = pd.read_csv("test.csv")
test = testset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = False)
test['Sex'][test['Sex'] == 'male'] = 1
test['Sex'][test['Sex'] == 'female'] = 2
test['Sex'][test['Sex'].isnull()] = 0
test['Embarked'][test['Embarked'] == 'S'] = 1
test['Embarked'][test['Embarked'] == 'C'] = 2
test['Embarked'][test['Embarked'] == 'Q'] = 3
test['Embarked'][test['Embarked'].isnull()] = 0
test_age_mean = test['Age'].mean()
test_fare_mean = test['Fare'].mean()
test['Age'][test['Age'].isnull()] = 0
test['Fare'][test['Fare'].isnull()] = 0

train_names = trainset['Name']
test_names = testset['Name']

test_titles = [re.findall(r'(?<=, ).*?(?=\.)', x)[0] for x in test_names]
train_titles = [re.findall(r'(?<=, ).*?(?=\.)', x)[0] for x in train_names]
title2idx = {}
i = 1
for t in set(train_titles).union(set(test_titles)):
    title2idx[t] = i
    i += 1
train_titles = np.array([title2idx[x] for x in train_titles])
test_titles = np.array([title2idx[x] for x in test_titles])

model = rf()
model.fit(feat, tar)

res = pd.DataFrame(columns = ('PassengerId', 'Survived'))
res['PassengerId'] = testset['PassengerId']
res['Survived'] = model.predict(test)

res.to_csv('Result.csv')
