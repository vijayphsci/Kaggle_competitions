import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#%%
#importing dataset
input_train=pd.read_csv('train.csv')
input_test=pd.read_csv('test.csv')
#%%
train=input_train.drop(['Id','SalePrice'],axis=1)
test=input_test.drop(['Id'],axis=1)
y=input_train['SalePrice']#target value 
#%%
print('train shape',train.shape)
print('test shape',test.shape)
#%%
#null values in train data
for val in train.columns:
    print(val,train[val].isnull().sum())
#%%
#null values in test data
for val in test.columns:
    print(val,test[val].isnull().sum())
#%%
#null limit is maximum number of null values that particular column can hold if  exceeds then drop column
null_limit=500 
train_remove_cols=[val for val in train.columns if train[val].isnull().sum()>null_limit]
test_remove_cols=[val for val in test.columns if test[val].isnull().sum()>null_limit]
#%%
print(train_remove_cols)
print(test_remove_cols)
#%%
#removing columns with high null values
train.drop(train_remove_cols,axis=1,inplace=True)
test.drop(train_remove_cols,axis=1,inplace=True)
#%%
#columns with null values
train_null_cols=[val for val in train.columns if train[val].isnull().any()]
test_null_cols=[val for val in test.columns if test[val].isnull().any()]
#%%
#HANDLING NULL VALUES
#if column contain categorical variable then fill null value of column by most frequent class
#if column contain numerical varible and if number of unique values is greater than "fill_limit" then fill
#null values by mean of column else fill null value with most frequent unique value
#also numerical column is considered categorial if number of unique values in column is less than equal to "fill_limit" 
fill_limit=5
for col in train_null_cols:
    if train[col].dtypes=='object':
        mode=train[col].mode()[0]
        train[col].fillna(mode,inplace=True)
    else:
        if len(train[col].unique())>fill_limit:
            mean=train[col].mean()
            train[col].fillna(mean,inplace=True)
        else:
            mode=train[col].mode()[0]
            train[col].fillna(mode,inplace=True)
for col in test_null_cols:
    if test[col].dtypes=='object':
        mode=test[col].mode()[0]
        test[col].fillna(mode,inplace=True)
    else:
        if len(test[col].unique())>fill_limit:
            mean=test[col].mean()
            test[col].fillna(mean,inplace=True)
        else:
            mode=test[col].mode()[0]
            test[col].fillna(mode,inplace=True)
#%%
#HANDLING CATEGORICAL COLUMNS
#columns with categorical values
categorical_cols=[col for col in train.columns if train[col].dtypes==object]
#%%
#number of differnt classes in categorical columns
for col in categorical_cols:
    print(col,len(train[col].unique()))    
#%%
#dummy_cols is categorical columns to be one hot encoded if number of unique values in particular column is less than or equal to "fill_limit"
#label_cols is categorical columns to be label encoded if number of unique values in particular column is greater than "fill_limit"     
dummy_cols=[col for col in categorical_cols if len(train[col].unique())<=fill_limit]
label_cols=[col for col in categorical_cols if len(train[col].unique())>fill_limit]
#%%
train_dummy=train[dummy_cols]
test_dummy=test[dummy_cols]
#%%
#one hot encoding dummy_cols
import sklearn.preprocessing as skp
one_hot=skp.OneHotEncoder(sparse=False,drop='first')
train_dummy=pd.DataFrame(one_hot.fit_transform(train_dummy))
test_dummy=pd.DataFrame(one_hot.transform(test_dummy))
#%%
#label encoding label_cols
for col in label_cols:
    a=skp.LabelEncoder()
    train[col]=a.fit_transform(train[col])
    test[col]=a.transform(test[col])
#%%    
#dropping dummy_cols 
train.drop(dummy_cols,inplace=True,axis=1)
test.drop(dummy_cols,inplace=True,axis=1)
#%%
#feature scalling 
#no feature scalling on one hot encoded columns
scx=skp.StandardScaler()
temp_train=pd.DataFrame(scx.fit_transform(train))
temp_test=pd.DataFrame(scx.transform(test))
#feature scalling target value
scy=skp.StandardScaler()
y=scy.fit_transform(input_train[['SalePrice']]).ravel()
#%%
#preserving column names
temp_train.columns=train.columns
temp_test.columns=test.columns
#%%
#final train and test dataset with feature scalling 
ftrain=pd.concat([temp_train,train_dummy],axis=1)
ftest=pd.concat([temp_test,test_dummy],axis=1)
#%%
#MODEL PERFORMANCE TEST
# K FOLD CROSS VALIDATION FOR DIFFRENT REGRESSION MODEL
#no hyper parameter tuning now
#selecting model with highest r2 score
#%%
X=ftrain.copy()
Xtest=ftest.copy()
print(fill_limit)
print(X.shape)
#%%
import sklearn.model_selection as skms
#%%
#LINEAR REGRESSION
import sklearn.linear_model as sklm
model_linear_reg=sklm.LinearRegression()
score=skms.cross_val_score(estimator=model_linear_reg,X=X,y=y,cv=5,n_jobs=-1)
score_linear_regression=score.mean()
print('r2score linear regression',score_linear_regression)
#%%
#RIDGE LINEAR REGRESSION
model_ridge_linear=sklm.Ridge()
score=skms.cross_val_score(estimator=model_ridge_linear,X=X,y=y,cv=5,n_jobs=-1)
score_ridge_linear=score.mean()
print('r2score ridge linear regression',score_ridge_linear)
#%%
#SUPPORT VECTOR REGRESSION
#SVR LINEAR
import sklearn.svm as skvm
model_svr_linear=skvm.SVR(kernel='linear')
score=skms.cross_val_score(estimator=model_svr_linear,X=X,y=y,cv=5,n_jobs=-1)
score_svr_linear=score.mean()
print('r2score SVR linear',score_svr_linear)
#%%
#SVR RBF kernel 
model_svr_rbf=skvm.SVR(kernel='rbf')
score=skms.cross_val_score(estimator=model_svr_rbf,X=X,y=y,cv=5,n_jobs=-1)
score_svr_rbf=score.mean()
print('r2score SVR kernel RBF',score_svr_rbf)
#%%
#DECISION TREE REGRESSOR
import sklearn.tree as skt
model_decision_tree=skt.DecisionTreeRegressor()
score=skms.cross_val_score(estimator=model_decision_tree,X=X,y=y,cv=5,n_jobs=-1)
score_decsion_tree=score.mean()
print('r2score decision tree',score_decsion_tree)
#%%
#RANDOM FOREST REGRESSOR
import sklearn.ensemble as ske
model_random_forest=ske.RandomForestRegressor()
score=skms.cross_val_score(estimator=model_random_forest,X=X,y=y,cv=5,n_jobs=-1)
score_random_forest=score.mean()
print('r2score random forest',score_random_forest)
#%%
#XGBOOST REGRESSOR
import xgboost as xgb
model_xgb=xgb.XGBRegressor()
score=skms.cross_val_score(estimator=model_xgb,X=X,y=y,cv=5,n_jobs=-1)
score_xgb=score.mean()
print('r2score XGBoost',score_xgb)
#%%
print('r2score linear regression',score_linear_regression)
print('r2score ridge linear regression',score_ridge_linear)
print('r2score SVR linear',score_svr_linear)
print('r2score SVR kernel RBF',score_svr_rbf)
print('r2score decision tree',score_decsion_tree)
print('r2score random forest',score_random_forest)
print('r2score XGBoost',score_xgb)
#%%
#XGBoost regressor is best model with highest r2 score
#hyperparameter tuning 
#rgrid search 
#finding best parameters which can fit model best
n_estimators=[20*i for i in range(1,31)]
max_depth=[i for i in range(1,51)]
max_depth.append(None)
model=xgb.XGBRegressor(random_state=0)
parameter={'n_estimators':n_estimators ,'max_depth':max_depth,'Learning_rate':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
grid_search = skms.RandomizedSearchCV(estimator=model,param_distributions = parameter,cv = 5,n_jobs = -1)
grid_search.fit(X,y)
print('best_score grid search',grid_search.best_score_)
print('best_parameters',grid_search.best_params_)
#%%
best_model=grid_search.best_estimator_
best_model.fit(X,y)
#%%
import sklearn.metrics as skmet
ypred=best_model.predict(X)
#%%
print('root mean squred log error',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(y),scy.inverse_transform(ypred))))
#%%
print('r2 score',skmet.r2_score(y,ypred))
#%%
test_prediction=scy.inverse_transform(best_model.predict(Xtest))
#%%
new=pd.DataFrame()
new['Id']=input_test['Id']
new['SalePrice']=test_prediction
new.to_csv('mysubmission.csv',index=False)