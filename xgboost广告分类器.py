# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:33:02 2020

@author: 86178
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization

'''数据来源http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/'''

data=pd.read_csv('ad.data',header=None,error_bad_lines=False)
data.rename(columns={0:'height',1:'width',2:'ratio(width/height)',1558:'label'},inplace=True)

data=data.replace('[?]',np.nan,regex=True)

data.head()

results=data.iloc[:,:].isnull().sum()
null_columns=[]

for index,value in results.iteritems():
    if value!=0:
        print('{} :{}'.format(index,value))
        null_columns.append(index)
        
data[null_columns]=data[null_columns].astype('float')
data.label=data.label.replace(['ad.','nonad.'],[1,0])

mask=np.random.rand(len(data))<0.8
train=data[mask]
test=data[~mask]

train['label'].hist(bins=5,figsize=(4,3))
plt.show()

cont_feas=['height','width','ratio(width/height)']
train[cont_feas].hist(bins=100,figsize=(9,5))
plt.show()

sns.violinplot(x='label',y='width',data=train)
plt.show()

plt.subplots(figsize=(12,6))
corr_mat=train[cont_feas].corr()
sns.heatmap(corr_mat,annot=True)
plt.show()

train_d=train.drop(['label'],axis=1).dropna()
test_d=test.drop(['label'],axis=1).dropna()

train_d['label']=1
test_d['label']=0

all_data=pd.concat((train_d,test_d))

all_data=all_data.iloc[np.random.permutation(len(all_data))]
all_data.reset_index(drop=True,inplace=True)

x=all_data.drop(['label'],axis=1)
y=all_data.label

train_size=1200 

x_train=x[:train_size]
x_test=x[train_size:]
y_train=y[:train_size]
y_test=y[train_size:]

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
auc=roc_auc_score(y_test,pred)
print('逻辑回归AUC：%.4f'%(auc))

'''PAC降维'''
from sklearn.decomposition import PCA

all_data=all_data.iloc[np.random.permutation(len(all_data))]

x=all_data.iloc[:,:1557]
y=all_data.iloc[:,1557:]
pca=PCA(n_components=2)

x_pca=pca.fit_transform(x)

label=all_data.label
pos_mask=label>=0.5
neg_mask=label<0.5
pos=x_pca[pos_mask]
neg=x_pca[neg_mask]

plt.scatter(pos[:,0],pos[:,1],s=60,marker='o',c='r')
plt.scatter(neg[:,0],neg[:,1],s=60,marker='^',c='b')
plt.title(u'PCA降维')
plt.xlabel('元素1')
plt.ylabel('元素2')
plt.legend()
plt.show()

'''平衡正负样本'''
pos_num=len(train[train['label']==1])
neg_num=len(train[train['label']==0])

'''xgboost参数'''
params={
        'objective':'binary:logistic',
        'booster':'gbtree',
        'eta':0.1,
        'eval_metric':'auc',
        'scale_pos_weight':neg_num/pos_num,
        'max_depth':6
        }
num_round=50

import xgboost as xgb

xgb_train=xgb.DMatrix(train.iloc[:,:1557],train['label'])
xgb_test=xgb.DMatrix(test.iloc[:,:1557],test['label'])

watchlist=[(xgb_train,'train'),(xgb_test,'test')]
model=xgb.train(params,xgb_train,num_round,watchlist)

model.save_model('./model.dat')
y_pred=model.predict(xgb_test)
auc=roc_auc_score(test.label,y_pred)
print('AUC得分： %f'%auc)

xgb.plot_importance(model, max_num_features=20,height=0.5)
plt.show()

'''贝叶斯优化'''

def xgb_optimize(learning_rate,n_estimators,
                 min_child_weight,colsample_bytree,
                 max_depth,subsample,
                 gamma,alpha):
    params={}
    params['learning_rate']=float(learning_rate)
    params['min_child_weight']=int(min_child_weight)
    params['colsample_bytree']=max(min(colsample_bytree,1),0)
    params['max_depth']=int(max_depth)
    params['subsample']=max(min(subsample, 1) ,0)
    params['gamma']=max(gamma,0)
    params['alpha']=max(alpha,0)
    params['objective']='binary:logistic'
    
    cv_result=xgb.cv(params,xgb_train,
                     num_boost_round=int(n_estimators),
                     nfold=5,seed=10,metrics=['auc'],
                     early_stopping_rounds=[xgb.callback.early_stop(30)])
    return cv_result['test-auc-mean'].iloc[-1]

pbounds={
    'learning_rate':(0.05,0.5),
    'n_estimators':(50,200),
    'min_child_weight':(1,10),
    'colsample_bytree':(0.5,1),
    'max_depth':(4,10),
    'subsample':(0.5,1),
    'gamma':(0,10),
    'alpha':(0,10)
    }

xgb_opt=BayesianOptimization(xgb_optimize,pbounds)
xgb_opt.maximize(init_points=5,n_iter=30)



bys_params={
        'objective':'binary:logistic',
        'booster':'gbtree',
        'eta':0.2453,
        'eval_metric':'auc',
        'alpha':0.3106,
        'gamma':0.3456,
        'colsample_bytree':0.8463,
        'subsample':0.9660,
        'min_child_weight':1,
        'scale_pos_weight':neg_num/pos_num,
        'max_depth':8
        }
bys_num_round=61

bys_model=xgb.train(bys_params,xgb_train,bys_num_round,watchlist)
bys_y_pred=bys_model.predict(xgb_test)
bys_auc=roc_auc_score(test.label,bys_y_pred)
print('AUC得分： %f'%bys_auc)

























































