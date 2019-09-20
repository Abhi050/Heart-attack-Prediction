#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 

# In[16]:


df=pd.read_csv("E:\\Jupyter\\heart.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[17]:


df.columns


# In[15]:


pip install autopep8


# In[19]:


df1=df.copy()
df2=df.copy()


# df1 is for knn while df2 is for randomforest

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Feature Selection

# In[29]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[30]:


plt.figure(figsize=(30,30))
df.hist()


# Data Preprocessing

# In[31]:


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
#they are categorical variables


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[33]:


dataset.head()


# `Trianing

# In[34]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)


# In[50]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[51]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5,forward=True)
fig.savefig('test2png.png', dpi=100)
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[52]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[53]:



score.mean()


# Random Forest Classidfier

# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[55]:


randomclass_scores = []
for k in range(1,21):
    randc_classifier = RandomForestClassifier(n_estimators=i)
    score=cross_val_score(randc_classifier,X,y,cv=10)
    randomclass_scores.append(score.mean())


# In[57]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5,forward=True)
fig.savefig('test2png.png', dpi=100)
plt.plot([k for k in range(1, 21)], randomclass_scores, color = 'red')
for i in range(1,21):
    plt.text(i, randomclass_scores[i-1], (i, randomclass_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('Random forest Classifier scores for different K values')


# In[58]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[59]:



score.mean()


# In[ ]:




