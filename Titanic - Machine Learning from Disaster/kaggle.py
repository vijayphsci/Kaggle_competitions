#importing  libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%%
#importing dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#%%
print(train.columns)
#%%
#missing values
print(train.isnull().sum())
#%%
print(test.isnull().sum())
#%%
#visualizing the distribution for positive and negative examples
def see_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=False, figsize=(10,8))
#return dictionary containing probability of survival for different classes within indivisual feature   
def probability_of_survival(feature):
    t1=train[train['Survived']==0][feature].value_counts()
    t2=train[train['Survived']==1][feature].value_counts()
    t3=t2.loc[t1.index]
    dictn={}
    for val in t1.index:
        dictn[val]=round(t3.loc[val]/(t1.loc[val]+t3.loc[val]),2)
    return dictn
#%%
see_chart('Pclass')
print(probability_of_survival('Pclass'))
#data shows there are 63% chance of survival if person belongs to 1st class , 47% for 2nd class , 24% for 3rd class
#therfore feature "Pclass" is strong indicator for prediction
#%%
see_chart('Sex')
print(probability_of_survival('Sex'))
#data shows there are 74% chance of survival if person is female and 19% if person is male
#therfore feature "Sex" is strong indicator for prediction
#%%
see_chart('SibSp')
#data show less chance of survival if person is single with 0 sibling
#%%
see_chart('Parch')
#data show less chance of survival if person have 0 parch
#%%
see_chart('Embarked')
print(probability_of_survival('Embarked'))
#data show 34% chance of survival if person belong to S embarked , 39 % for Q embarked , 55% for C embarked
#%%
#Age distribution 
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()
#people with age group of less than 18 is more likely to survive
#%%
#fare distribution 
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.show()

#%%
#feature Enginnering 
#extracting more information from data 
#Name Title of person('Mr','Miss',Ms.,etc.) may be good indicator
train_test_data = [train, test] # combining train and test dataset
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#%%
#extracting first character of feature Cabin 
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
#%%
print(train['Title'].value_counts()) 
#%%
print(test['Title'].value_counts())
#%%
#mapping titles
title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_map)
#%%
#mapping Sex 
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#%%
#imputing missing values
#filling null values in Age columns by median of age gropued by Titles 
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#%%
#null values in Embarked column is only 2 in train data  we can fill by most frequent category 
train['Embarked'].fillna(train['Embarked'].value_counts().index[0],inplace=True)
#%%
#mapping embarked columns
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#%%
#imputing Fare columns by mean of fare grouped by Classes
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("mean"), inplace=True)
#%%
#most of data is empty in Cabin is empty 687 in train data and 327 in test data
#distribution of cabin among different classes
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#%%
#from the plot we can see 1st class persons were in cabin C 
#therefore we can fill null value of cabin by most frequent cabin grouped by pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform(lambda x: x.value_counts().index[0]), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform(lambda x: x.value_counts().index[0]), inplace=True)
#%%
#cabin mapping
#label encoding Cabin 
import sklearn.preprocessing as skp
lable_cabin=skp.LabelEncoder()
train['Cabin']=lable_cabin.fit_transform(train['Cabin'])
test['Cabin']=lable_cabin.transform(test['Cabin'])
#%%
#creating new column containing family size of person sibling+parch 
train["Family"] = train["SibSp"] + train["Parch"] + 1
test["Family"] = test["SibSp"] + test["Parch"] + 1
#%%
#final features  for train and test
feature=['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Embarked', 'Title', 'Family']
#%%
ftrain=train[feature]
ftest=test[feature]
#%%
#feature scalling 
fs=skp.StandardScaler()
ftrain=pd.DataFrame(fs.fit_transform(ftrain))
ftest=pd.DataFrame(fs.transform(ftest))
ftrain.columns=feature
ftest.columns=feature
#%%
#target value
y=train['Survived']
#%%
#training from different models and evaluting performance
#no hyperparameter tuning 
#choosing model which gives highest accuracy score
import sklearn.metrics as skmet
import sklearn.model_selection as skms
#LOGISTIC REGRESSION
import sklearn.linear_model as sklm
model_logistic_regression=sklm.LogisticRegression()
#K foldcross validation cv=5 
accuracy=skms.cross_val_score(estimator=model_logistic_regression,X=ftrain,y=y,cv=5)
#mean of accuracy score
accuracy_logistic_regression=accuracy.mean()
print('logistic regreesion model accuracy',accuracy_logistic_regression)
#%%
#SUPPORT VECTOR MACHINE
#SVC LINEAR 
import sklearn.svm as skvm
model_svc_linear=skvm.SVC(kernel='linear')
accuracy=skms.cross_val_score(estimator=model_svc_linear,X=ftrain,y=y,cv=5)
accuracy_svc_linear=accuracy.mean()
print('SVC linear model accuracy',accuracy_svc_linear)
#%%
#SVC NON LINEAR (kernel RBF)
import sklearn.svm as skvm
model_svc_rbf=skvm.SVC(kernel='rbf')
accuracy=skms.cross_val_score(estimator=model_svc_rbf,X=ftrain,y=y,cv=5)
accuracy_svc_rbf=accuracy.mean()
print('SVC kernel rbf model accuracy',accuracy_svc_rbf)
#%%
#DECISION TREE CLASSIFIER
import sklearn.tree as skt
model_decision_tree=skt.DecisionTreeClassifier()
accuracy=skms.cross_val_score(estimator=model_decision_tree,X=ftrain,y=y,cv=5)
accuracy_decision_tree=accuracy.mean()
print('decision tree model accuracy',accuracy_decision_tree)
#%%
#RANDOM FOREST CLASSIFIER 
import sklearn.ensemble as ske
model_random_forest=ske.RandomForestClassifier()
accuracy=skms.cross_val_score(estimator=model_random_forest,X=ftrain,y=y,cv=5)
accuracy_random_forest=accuracy.mean()
print('random forest model accuracy',accuracy_random_forest)
#%%
#NAIVE BAYES CLASSIFIER
import sklearn.naive_bayes as sknb
model_naive_bayes=sknb.GaussianNB()
accuracy=skms.cross_val_score(estimator=model_naive_bayes,X=ftrain,y=y,cv=5)
accuracy_naive_bayes=accuracy.mean()
print('naive bayes model accuracy',accuracy_naive_bayes)
#%%
#KNN 
import sklearn.neighbors as skn
model_knn=skn.KNeighborsClassifier(n_neighbors=5)
accuracy=skms.cross_val_score(estimator=model_knn,X=ftrain,y=y,cv=5)
accuracy_knn=accuracy.mean()
print('KNN model accuracy',accuracy_knn)
#%%
#XGBOOST CLASSIFIER
import xgboost as xgb
model_xgb=xgb.XGBClassifier()
accuracy=skms.cross_val_score(estimator=model_xgb,X=ftrain,y=y,cv=5)
accuracy_xgb=accuracy.mean()
print('Xbg classifier model accuracy',accuracy_xgb)
#%%
#we performed model training on 8 different classification models
#overall accuracy performance, model which gives highest accuracy score will be chosen to train on data
print('logistic regreesion model accuracy',accuracy_logistic_regression)
print('SVC linear model accuracy',accuracy_svc_linear)
print('SVC kernel rbf model accuracy',accuracy_svc_rbf)
print('decision tree model accuracy',accuracy_decision_tree)
print('random forest model accuracy',accuracy_random_forest)
print('naive bayes model accuracy',accuracy_naive_bayes)
print('KNN model accuracy',accuracy_knn)
print('Xbg classifier model accuracy',accuracy_xgb)
#%%
#best model is SVC (kernel =rbf) 

#%%
#hyperparameter tuning 
#rgrid search 
#finding best parameters which can fit model best
model=skvm.SVC(kernel='rbf',random_state=0)
parameter={'C': [0.1,0.25,0.5,1,2,5, 10,100,1000], 'gamma': [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9,1,'scale']}
grid_search = skms.GridSearchCV(estimator =model, param_grid = parameter,scoring = 'accuracy',cv = 5,n_jobs = -1)
grid_search.fit(ftrain,y)
print('best_score grid search',grid_search.best_score_)
print('best_parameters',grid_search.best_params_)
#%%
#fitting data on  best estimator
best_model=grid_search.best_estimator_
best_model.fit(ftrain,y)
#%%
#prediction of test data
test_prediction=best_model.predict(ftest)
#%%
#this test prediction resulted in accuracy score of  0.77511 on submitting at kaggle titanic competition 
#means 324 out of 418 test data were correctly predicted
#%%
new=pd.DataFrame()
new['PassengerId']=test['PassengerId']
new['Survived']=test_prediction
new.to_csv('submission.csv',index=False)

