import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics as skmet
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import xgboost as xgb
import sklearn.tree as skt
import sklearn.svm as skvm
import sklearn.linear_model as sklm
import sklearn.ensemble as ske
import sklearn.decomposition as skd
#%%
input_train=pd.read_csv('train.csv')
input_test=pd.read_csv('test.csv')
train=input_train.drop(['Id','SalePrice'],axis=1)
test=input_test.drop(['Id'],axis=1)
y=input_train['SalePrice']
#%%
null_limit=800 
train_remove_cols=[val for val in train.columns if train[val].isnull().sum()>null_limit]
test_remove_cols=[val for val in test.columns if test[val].isnull().sum()>null_limit]
#%%
train.drop(train_remove_cols,axis=1,inplace=True)
test.drop(train_remove_cols,axis=1,inplace=True)
#%%
def preprocess(train_data,test_data,numeric_null_replace='mean',cardinality_limit=5,dummy_limit=5,median_replace=True,test_replace_train=True):    
    train=train_data.copy()
    test=test_data.copy()
    train_null_cols=[val for val in train.columns if train[val].isnull().any()]
    test_null_cols=[val for val in test.columns if test[val].isnull().any()]
    for col in train_null_cols:
        if train[col].dtypes=='object':
                rep_val=train[col].mode()[0]
                train[col].fillna(rep_val,inplace=True)
                test[col].fillna(rep_val,inplace=True)
        else:
            if numeric_null_replace=='mean':
                if len(train[col].unique())>cardinality_limit:
                    rep_val=train[col].mean()
                else:
                    if median_replace:
                        rep_val=train[col].median()
                    else:
                        rep_val=train[col].mode()[0]
            elif numeric_null_replace=='mode':
                rep_val=train[col].mode()[0]
            elif numeric_null_replace=='median':
                rep_val=train[col].median()
                
            train[col].fillna(rep_val,inplace=True)
            test[col].fillna(rep_val,inplace=True)
    for col in test_null_cols:
        if col not in train_null_cols:
            if test[col].dtypes=='object':
                if test_replace_train:
                    rep_val=train[col].mode()[0]
                else:
                    rep_val=test[col].mode()[0]
                test[col].fillna(rep_val,inplace=True)
            else:
                if numeric_null_replace=='mean':
                    if len(train[col].unique())>cardinality_limit:
                        if test_replace_train:
                            rep_val=train[col].mean()
                        else:
                            rep_val=test[col].mean()
                    else:
                        if median_replace:
                            if test_replace_train:
                                rep_val=train[col].median()
                            else:
                                rep_val=test[col].median()
                        else:
                            if test_replace_train:
                                rep_val=train[col].mode()[0]
                            else:
                                rep_val=test[col].mode()[0]
                elif numeric_null_replace=='mode':
                    if test_replace_train:
                        rep_val=train[col].mode()[0]
                    else:
                        rep_val=test[col].mode()[0]
                elif numeric_null_replace=='median':
                    if test_replace_train:
                        rep_val=train[col].median()
                    else:
                        rep_val=test[col].median()

                test[col].fillna(rep_val,inplace=True)
    
    categorical_cols=[col for col in train.columns if train[col].dtypes==object]
    dummy_cols=[col for col in categorical_cols if len(train[col].unique())<=dummy_limit]
    label_cols=[col for col in categorical_cols if len(train[col].unique())>dummy_limit]
    train_dummy=train[dummy_cols]
    test_dummy=test[dummy_cols]
    one_hot=skp.OneHotEncoder(sparse=False,drop='first')
    train_dummy=pd.DataFrame(one_hot.fit_transform(train_dummy))
    test_dummy=pd.DataFrame(one_hot.transform(test_dummy))
    for col in label_cols:
        a=skp.LabelEncoder()
        train[col]=a.fit_transform(train[col])
        test[col]=a.transform(test[col])
    train.drop(dummy_cols,inplace=True,axis=1)
    test.drop(dummy_cols,inplace=True,axis=1)
    scx=skp.StandardScaler()
    temp_train=pd.DataFrame(scx.fit_transform(train))
    temp_test=pd.DataFrame(scx.transform(test))
    temp_train.columns=train.columns
    temp_test.columns=test.columns
    ftrain=pd.concat([temp_train,train_dummy],axis=1)
    ftest=pd.concat([temp_test,test_dummy],axis=1)
    return ftrain,ftest,train_dummy.shape[1]
def model_xgb(X,y,size):
    xtrain,xval,ytrain,yval=skms.train_test_split(X,y,test_size=size,random_state=1)
    best_model=xgb.XGBRegressor() 
    best_model.fit(xtrain,ytrain)
    pred_train=best_model.predict(xtrain)
    pred_val=best_model.predict(xval)
    print('xgb regressor')
    print('r2 score train',skmet.r2_score(ytrain,pred_train))
    print('r2 score val',skmet.r2_score(yval,pred_val))
    print('rmslogerror train',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(ytrain),scy.inverse_transform(pred_train))))
    print('rmslogerror val',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(yval),scy.inverse_transform(pred_val))))

def model_linear_reg(X,y,size):
    xtrain,xval,ytrain,yval=skms.train_test_split(X,y,test_size=size,random_state=1)
    best_model=sklm.LinearRegression()
    best_model.fit(xtrain,ytrain)
    pred_train=best_model.predict(xtrain)
    pred_val=best_model.predict(xval)
    print('linear regression')
    print('r2 score train',skmet.r2_score(ytrain,pred_train))
    print('r2 score val',skmet.r2_score(yval,pred_val))
    print('rmslogerror train',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(ytrain),scy.inverse_transform(pred_train))))
    print('rmslogerror val',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(yval),scy.inverse_transform(pred_val))))

def remove_corr(X,Xtest,corr_limit=0.8):
    corr=X.corr()
    n=X.shape[1]
    cols=X.columns
    corr_dict={}
    for i in range(n):
        temp=[]
        for j in range(n):
            if j>i:
                if abs(corr.iloc[i,j])>corr_limit:
                    temp.append(cols[j])
        if temp:
            corr_dict[cols[i]]=temp
    remove_corr_cols=list(corr_dict.keys())
    Xnew=X.drop(remove_corr_cols,axis=1)
    Xtestnew=Xtest.drop(remove_corr_cols,axis=1)
    print('feature decrease by',-Xnew.shape[1]+X.shape[1])
    return Xnew,Xtestnew
def poly(X,Xtest,degree=2):
    pf=skp.PolynomialFeatures(degree=degree,include_bias=False)
    pf.fit(X)
    Xp=pd.DataFrame(pf.transform(X))
    Xtestp=pd.DataFrame(pf.transform(Xtest))
    t1=X.shape
    t2=Xp.shape
    print('old shape',t1,'new shape',t2)
    print('feature increase by',t2[1]-t1[1])
    return Xp,Xtestp
def pca(X,Xtest,select=50):
    t=skd.PCA(n_components=select)
    Xpca=t.fit_transform(X)
    Xtestpca=t.transform(X)
    return Xpca,Xtestpca
def model_ridge(X,y,alpha=[0.1,1,10,100],size=0.3):
    xtrain,xval,ytrain,yval=skms.train_test_split(X,y,test_size=size,random_state=1)
    best_model=sklm.RidgeCV(alphas=alpha)
    best_model.fit(xtrain,ytrain)
    pred_train=best_model.predict(xtrain)
    pred_val=best_model.predict(xval)
    print('Ridge regression')
    print('r2 score train',skmet.r2_score(ytrain,pred_train))
    print('r2 score val',skmet.r2_score(yval,pred_val))
    print('rmslogerror train',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(ytrain),scy.inverse_transform(pred_train))))
    print('rmslogerror val',np.sqrt(skmet.mean_squared_log_error(scy.inverse_transform(yval),scy.inverse_transform(pred_val))))

def duplicate(X):
    print('duplicated any:',X.columns.duplicated().any())
#%%
scy=skp.StandardScaler()
y=scy.fit_transform(input_train[['SalePrice']]).ravel()
#%%
dummy_limit=25
numeric_null_replace='median'
cardinality_limit=5
median_replace=False
test_replace_train=True
train_data=train
test_data=test
alpha=[0.05,0.1,0.5,1,5,10,50,100,1000,5000]
X,Xtest,dummies=preprocess(train_data,test_data,numeric_null_replace,cardinality_limit,dummy_limit,median_replace,test_replace_train)
#%%
print('dummies',dummies,'features',X.shape[1])
duplicate(X)
#%%
X_rc,Xtest_rc=remove_corr(X,Xtest,0.9)
print('old shape',X.shape,'new shape',X_rc.shape)
#%%
Xp,Xtestp=poly(X_rc,Xtest_rc,degree=2) 
#%%
print('dummy_limit',dummy_limit,'cardinality_limit',cardinality_limit)
#model_xgb(X_rc,y,0.3)
#model_linear_reg(X_rc,y,0.3)
model_ridge(Xp,y,alpha,0.25)
#%%
model=sklm.Ridge()
alpha=[10,50,100,500,1000,1500,2000,2500,3000]
max_itr=[200*i for i in range(1,21)]
parameter={'alpha':alpha ,'max_iter':max_itr}
grid_search = skms.RandomizedSearchCV(estimator=model,param_distributions = parameter,cv = 5,n_jobs = -1)
grid_search.fit(Xp,y)
print('best_score grid search',grid_search.best_score_)
print('best_parameters',grid_search.best_params_)
#%%
best_model=grid_search.best_estimator_
best_model.fit(Xp,y)
#%%
test_prediction=scy.inverse_transform(best_model.predict(Xtestp))
#%%
new=pd.DataFrame()
new['Id']=input_test['Id']
new['SalePrice']=test_prediction
new.to_csv('mynewsubmission01.csv',index=False)










