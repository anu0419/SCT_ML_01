#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[73]:


dataset = pd.read_csv("C:/Users/anasu/Downloads/trainn.csv")
df = pd.DataFrame(dataset)
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[74]:


dataset


# In[75]:


dataset.info()


# In[76]:


dataset.head(52)


# In[77]:


features = ["GrLivArea", "bedroomabvgr","fullbath"]


# In[78]:


x = dataset[features]
y = dataset["SalePrice"]


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=70)

x_train.shape, x_test.shape


# In[80]:


sns.pairplot(dataset)
plt.show()


# In[81]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[82]:


y_pred = model.predict(x_test)

comparison = pd.DataFrame({'Actual':y_test, 'predicted':y_pred})
comparison


# In[83]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[85]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

