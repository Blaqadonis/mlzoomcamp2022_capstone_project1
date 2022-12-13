#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!pip install xgboost
#!pip install -U scikit-learn
#!pip install seaborn
#!pip install matplotlib
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

data1['target'] = data1['aim'].replace(['left'], 1)
data1['target'] = data1['target'].replace(['right'], 3)
data1['target'] = data1['target'].replace(['middle'], 2)
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

dict_train = train.to_dict(orient='records')
dict_val = val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

x_train = dv.fit_transform(dict_train)
x_val = dv.transform(dict_val)

import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)

xgb_params = {
    'eta': 0.5,
    'max_depth': 9,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1
}
model = xgb.train(xgb_params, dtrain,
                  num_boost_round=501, verbose_eval=10)

y_pred= model.predict(dval)

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE: %f" % (rmse))

pip install bentoml==1.0.7
import bentoml
bentoml.xgboost.save_model("cristiano_penalty_aim_model",model, custom_objects={"DictVectorizer":dv},
signatures={"predict": {"batchable":True,"batch_dim":0,}})


# In[ ]:




