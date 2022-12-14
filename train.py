#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

data1['previous_penalty'] = data1['previous_penalty'].fillna(1).astype(int)  #to turn the column to integer

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

dict_train_data = train_data.to_dict(orient='records')
dict_test_data = test_data.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(dict_train_data)
X_test = dv.transform(dict_test_data)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.metrics import accuracy_score

gnb.fit(X_train, y_train_data)

y_pred_train = gnb.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train_data, y_pred_train)))

y_pred = gnb.predict(X_test)
print('Test Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print(y_pred[0:10])


accuracy_score(y_val,y_pred)*100



#!pip install bentoml==1.0.7
import bentoml

bentoml.sklearn.save_model("cristiano_penalty_aim_model",gnb, custom_objects={"DictVectorizer":dv},
signatures={"predict": {"batchable":True,"batch_dim":0,}})


# In[ ]:




