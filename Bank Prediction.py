
# coding: utf-8

# In[58]:


import pandas as pd
bank1 = pd.read_csv("bank.csv")
bank1


# In[59]:


bank1.isnull().sum()
#There's only 1 missing value and it's somebody's age
bank = bank1.dropna()
#I dropped the data containing missing value


# In[60]:


#The data points have very different ranges. They also have different magnitudes
#some columns are mostly on 0.1 scale whereas some columns work with thousands
bank.describe()


# In[61]:


#encoding categorical attributes
categorical_feature_mask = bank.dtypes==object
categorical_cols = bank.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

bank[categorical_cols] = bank[categorical_cols].apply(lambda col: le.fit_transform(col))
bank[categorical_cols].head(10)


# In[128]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(bank, test_size=0.2)
test


# In[63]:


#I will use age, job, marital status, education, default, housing, loan, pdays, poutcome, cons.price.idx, cons.conf.idx
#These fields are the ones I found most relevant to the end goal


# In[150]:


predictors = ['age', 'job', 'marital', 'education','default','housing','loan','pdays','poutcome','cons.price.idx','cons.conf.idx']
X_train = train[predictors].to_numpy()
y_train = train['y'].to_numpy()
model = LogisticRegression(solver = 'liblinear')
model.fit(X_train, y_train)


# In[152]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits = 5, shuffle = True)
scores = cross_val_score(model, X_train, y_train, cv = kfold)
print("Accuracy: %0.2f +/- %0.2f" % (scores.mean(), scores.std()))
#My accuracy is 90%


# In[153]:


X_test = test[predictors].to_numpy()
y_test = test['y'].to_numpy()
preds = model.predict(X_test)

correct=0
for x in range(8236): #8236 is the size of the test array
    if(preds[x] == y_test[x]):
        correct = correct+1
comparison= (correct/8236)*100
print('Test Accuracy: ' + str(comparison) + ' percent')
#Judging based on my test dataset, accuracy is pretty good.


# In[154]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, preds) 
  
print('Confusion Matrix :')
print(results) 
print('Accuracy Score: ' + str(accuracy_score(y_test, preds)))
print('Report : ')
print(classification_report(y_test, preds))
#The model seems to be pretty accurate although there are some false predictions.


# In[155]:


#predictions on unknown data
posb = pd.read_csv("bank-unknown.csv")
categorical_feature_mask1 = posb.dtypes==object
categorical_cols1 = posb.columns[categorical_feature_mask1].tolist()

posb[categorical_cols1] = posb[categorical_cols1].apply(lambda col: le.fit_transform(col))
x_posb = posb[predictors].to_numpy()
model.predict(x_posb)
#According to the model the first and the fifth customers will accept a loan

