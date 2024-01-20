#!/usr/bin/env python
# coding: utf-8

# In[12]:


import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


# In[13]:


wine=pd.read_csv("winequality-red.csv")


# In[14]:


wine.head()


# In[15]:


wine.isnull().sum


# In[16]:


wine.drop_duplicates()


# In[17]:


sns.heatmap(wine.corr(), annot = True, fmt = '.2f', center =0);


# In[18]:


X = wine.drop('quality', axis=1)
Y = wine['quality']


# In[19]:


#split the data into train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.4, random_state= 42)


# In[20]:


#create a randomforestclassifer
rf_model = RandomForestClassifier(random_state=42)


# In[21]:


# Define the hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = best_rf_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




