# cristiano_penalty_aim
Local deployment
```pip install pipenv```
```pipenv install numpy pandas seaborn bentoml==1.0.7 scikit-learn==1.1.3 xgboost==1.7.1 pydantic==1.10.2```
Enter shell. To open the notebook.ipynb and see all the models
```pipenv shell```
For the following you need to run train.py
```pipenv run python train.py```
Then, get the service running on localhost
```pipenv run bentoml serve predict.py:svc```
Click on 'Try it out' and make accurate predictions that is guaranteed to blow the minds of your pals.
Optional: Run locust to test server, make sure you have installed it
```pipenv install locust```
```pipenv run locust -H http://localhost:3000```
and check it out on browser.
Production deployment with BentoML
You need to have Docker installed (I used Docker Desktop with WSL2)


First we need to build the bento with

```pipenv run bentoml build```
Docker container
Once we have the bento tag we containerize it.
```pipenv run bentoml containerize cristiano_penalty_aim:tag```
Replace tag with the tag you get from bentoml build.


## WHAT THE MODEL DOES
It reads the csv file 'Cristiano_Penalties.csv' from its directory.
data = pd.read_csv('Cristiano_Penalties.csv')

Then it copies the data to data1.
data1 = data.copy()

data1 is then prepared.
data1= data1.applymap(lambda s:s.lower().replace(' ', '_') if type(s) == str else s)
data1.columns = [x.lower().replace(' ', '_') for x in data1.columns]
data1.columns = [x.lower().replace(':', '') for x in data1.columns]
data1.columns = [x.lower().replace('*', '') for x in data1.columns]
data1.columns = [x.lower().replace('.', '') for x in data1.columns]
data1['previous_penalty'] = data1['previous_penalty'].fillna(0).astype(int)  #to turn the column to integer

Here, a new feature called 'target, which is just the logic of feature 'aim' is engineered, and this feature will stand in for feature 'aim' when data1 is served to the model. Note that i chose '1,2,3'. I initially wanted to go with '0,1,2' for 'left', 'middle','right', but I get errors; something about Gaussian Naives Bayes model not accepting nan input or something but I am sure that it has to do with the zero input.
data1['target'] = data1['aim'].replace(['left'], 1)
data1['target'] = data1['target'].replace(['right'], 3)
data1['target'] = data1['target'].replace(['middle'], 2)

# Just defining input variables, and dropping the ones not needed as input to the model.
y = data1['target']
data1 = data1.drop('target',axis=1)
data1 = data1.drop('aim',axis=1)
data1 = data1.drop('position_style',axis=1)
data1 = data1.drop('natural_foot',axis=1)

# Here, I am splitting the data into different sets training, testing, and validation, and assigning weights to them from data1 and target variable 'y'.
train_data, test_data, y_train_data, y_test = train_test_split(data1, y, test_size = 0.2, random_state=30)
print(train_data.shape,' ', test_data.shape)

train, val, y_train, y_val = train_test_split(train_data, y_train_data, test_size = 0.25, random_state=11)
print(train.shape,' ', val.shape)

train = train.reset_index()
train_data = train_data.reset_index()
test_data = test_data.reset_index()
val = val.reset_index()

# Here, I am transforming input data to dictionaries.
dict_train_data = train_data.to_dict(orient='records')
dict_test_data = test_data.to_dict(orient='records')

# Changing the dicts to vector form
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train_data)
X_test = dv.transform(dict_test_data)

# model init
gnb = GaussianNB()

# Fitting vectors in model and learning
gnb.fit(X_train, y_train_data)

# predictions on training set
y_pred_train = gnb.predict(X_train)

# predictions on never-before-seen
y_pred = gnb.predict(X_test)
