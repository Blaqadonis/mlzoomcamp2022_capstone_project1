#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!pip install -U scikit-learn

import numpy as np
import pandas as pd

data = pd.read_csv('Cristiano_Penalties.csv')

data1 = data.copy()

data1= data1.applymap(lambda s:s.lower().replace(' ', '_') if type(s) == str else s)
data1.columns = [x.lower().replace(' ', '_') for x in data1.columns]
data1.columns = [x.lower().replace(':', '') for x in data1.columns]
data1.columns = [x.lower().replace('*', '') for x in data1.columns]
data1.columns = [x.lower().replace('.', '') for x in data1.columns]

data1['minute_taken'] = data1['minute_taken'].fillna(0).astype(int)  #to turn the column to integer

data1['target'] = data1['aim'].replace(['left'], 0)
data1['target'] = data1['target'].replace(['right'], 2)
data1['target'] = data1['target'].replace(['middle'], 1)
y = data1['target']
data1 = data1.drop('target',axis=1)
data1 = data1.drop('aim',axis=1)
data1 = data1.drop('position_style',axis=1)
data1 = data1.drop('natural_foot',axis=1)


from sklearn.model_selection import train_test_split
train_data, test_data, y_train_data, y_test = train_test_split(data1, y, test_size = 0.2, random_state=30)
print(train_data.shape,' ', test_data.shape)

train, val, y_train, y_val = train_test_split(train_data, y_train_data, test_size = 0.25, random_state=11)
print(train.shape,' ', val.shape)

train = train.reset_index()
train_data = train_data.reset_index()
test_data = test_data.reset_index()
val = val.reset_index()

from sklearn.feature_extraction import DictVectorizer

dict_train_data = train_data.to_dict(orient='records')
dict_test_data = test_data.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(dict_train_data)
X_test = dv.transform(dict_test_data)

import xgboost as xgb

Dtrain = xgb.DMatrix(X_train, label=y_train_data)
Dtest = xgb.DMatrix(X_test, label=y_test)

xgb_params = {
    'eta': 0.5,
    'max_depth': 6,
    'min_child_weight': 10,
'num_class':3,
    'objective':'multi:softmax',
    'nthread': 8,
    'seed': 1
}

from sklearn.metrics import accuracy_score
model = xgb.train(xgb_params, Dtrain, num_boost_round=10)
y_pred = model.predict(Dtest)
score = np.sqrt(accuracy_score(y_test, y_pred))
print("Accuracy: %f" % (score))

#!pip install bentoml==1.0.7
import bentoml
bentoml.xgboost.save_model("cristiano_penalty_aim_model",model, custom_objects={"DictVectorizer":dv},
signatures={"predict": {"batchable":True,"batch_dim":0,}})


# In[ ]:




