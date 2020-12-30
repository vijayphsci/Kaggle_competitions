import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%%
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#%%
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
#%%
train_test_data = [train, test] # combining train and test dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#%%
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
#%%
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#%%
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#%%
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#%%
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    
#%%
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#%%
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
#%%
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#%%
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

#%%
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform(lambda x: x.value_counts().index[0]), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform(lambda x: x.value_counts().index[0]), inplace=True)
#%%
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
#%%
train.columns
#%%
cols=['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize']
#%%
x=train[cols]
y=train['Survived']
xtest=test[cols]
#%%
import sklearn.model_selection as skms
import sklearn.metrics as skmet
import sklearn.ensemble as ske
model=ske.RandomForestClassifier(criterion='entropy',random_state=0)
acc=skms.cross_val_score(estimator=model, X = x, y = y, cv =10)
print(cols)
print('accuracy',acc.mean())

#%%
xtrain,xval,ytrain,yval=skms.train_test_split(x,y,test_size=0.3,random_state=0)
#%%
nodes=list(range(1,31,1))
nodes.append(None)
nodes
#%%
model=ske.RandomForestClassifier(criterion='entropy',random_state=0)
parameter = {'max_depth':nodes,'n_estimators': [i*20 for i in range(1,21)]}
grid_search = skms.GridSearchCV(estimator =model , param_grid = parameter,scoring = 'accuracy',cv = 10,n_jobs = -1)
grid_search.fit(x,y)
#%%
best_model=grid_search.best_estimator_
best_model.fit(xtrain,ytrain)
print(cols)
print('best_score grid search',grid_search.best_score_)
print('best_parameters',grid_search.best_params_)
print('train accuracy',skmet.accuracy_score(ytrain,best_model.predict(xtrain)))
print('val accuracy',skmet.accuracy_score(yval,best_model.predict(xval)))

#%%
final_model=ske.RandomForestClassifier(max_depth=8, n_estimators=100,criterion='entropy',random_state=0)
final_model.fit(x,y)
#%%
new=pd.DataFrame()
new['PassengerId']=test['PassengerId']
new['Survived']=final_model.predict(xtest)
new.to_csv('submission_10.csv',index=False)
#%%
import sklearn.preprocessing as skp
scx=skp.StandardScaler()
scx.fit(x)
x=scx.transform(x)
xtest=scx.transform(x)
#%%
import sklearn.svm as skvm
model=skvm.SVC()
acc=skms.cross_val_score(estimator=model, X = x, y = y, cv =8)
print(acc.mean())
print(acc.std())

#%%
model=skvm.SVC(kernel='rbf',random_state=0)
parameter={'C': [0.25,0.5,1, 10, 100, 1000], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1,'scale']}
grid_search = skms.GridSearchCV(estimator =model,param_grid = parameter,scoring = 'accuracy',cv = 10,n_jobs = -1)
grid_search.fit(x,y)
#%%
xtrain,xval,ytrain,yval=skms.train_test_split(x,y,test_size=0.3,random_state=0)
#%%
best_model=grid_search.best_estimator_
best_model.fit(xtrain,ytrain)
print(cols)
print('best_score grid search',grid_search.best_score_)
print('best_parameters',grid_search.best_params_)
print('train accuracy',skmet.accuracy_score(ytrain,best_model.predict(xtrain)))
print('val accuracy',skmet.accuracy_score(yval,best_model.predict(xval)))

