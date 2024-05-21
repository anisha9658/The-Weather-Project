#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pandas as pd


# In[121]:


weather = pd.read_csv("local_weather.csv", index_col="DATE")


# In[122]:


weather.head()


# In[123]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[124]:


core_weather = weather[["PRCP","SNOW","SNWD","TMAX","TMIN"]].copy()


# In[125]:


core_weather.head()


# In[126]:


core_weather.columns = ["precip", "snow" , "snow_depth" , "temp_max", "temp_min"]


# In[127]:


core_weather.head()


# In[128]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[129]:


core_weather["snow"].value_counts()


# In[130]:


del core_weather["snow"]


# In[131]:


core_weather["snow_depth"].value_counts()


# In[132]:


del core_weather["snow_depth"]


# In[133]:


core_weather[pd.isnull(core_weather["precip"])]


# In[134]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[135]:


core_weather[pd.isnull(core_weather["temp_max"])]


# In[136]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[137]:


core_weather = core_weather.fillna(method = "ffill")


# In[138]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[139]:


core_weather.dtypes


# In[140]:


core_weather.index


# In[141]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[142]:


core_weather.index


# In[143]:


core_weather.apply(lambda x: (x==9999).sum())


# In[144]:


core_weather[["temp_max", "temp_min"]].plot()


# In[145]:


core_weather.index.year.value_counts().sort_index()


# In[146]:


core_weather["precip"].plot()


# In[147]:


core_weather.groupby(core_weather.index.year).sum()["precip"]


# In[148]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[149]:


core_weather


# In[152]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[153]:


core_weather


# In[155]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=1)


# In[160]:


train = core_weather.loc[:"2020-12-31"]
test = core_weather.loc["2021-01-01":]


# In[161]:


reg.fit(train[predictors],train["target"])


# In[162]:


predictions = reg.predict(test[predictors])


# In[163]:


from sklearn.metrics import mean_absolute_error


# In[164]:


mean_absolute_error(test["target"],predictions)


# In[166]:


combined = pd.concat([test["target"],pd.Series(predictions, index=test.index)], axis =1)
combined.columns = ["actual","predictions"]


# In[167]:


combined


# In[169]:


combined.plot()


# In[170]:


reg.coef_


# In[171]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors],train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"],predictions)
    combined = pd.concat([test["target"],pd.Series(predictions, index=test.index)], axis =1)
    combined.columns = ["actual","predictions"]
    return error, combined


# In[172]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()


# In[173]:


core_weather


# In[174]:


core_weather["month_day_max"] = core_weather["month_max"]/core_weather["temp_max"]


# In[176]:


core_weather["max_min"] = core_weather["temp_max"]/core_weather["temp_min"]


# In[177]:


core_weather


# In[178]:


predictors = ["precip","temp_max","temp_min","month_max","month_day_max","max_min"]


# In[179]:


core_weather = core_weather.iloc[30:,:].copy()


# In[180]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[182]:


error


# In[185]:


combined.plot()


# In[189]:


core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month).apply(lambda x: x.expanding(1).mean())


# In[190]:


core_weather


# In[192]:


core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year).apply(lambda x: x.expanding(1).mean())


# In[195]:


predictors = ["precip","temp_max","temp_min","month_max","month_day_max","max_min","day_of_year_avg","monthly_avg"]


# In[196]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[197]:


error


# In[198]:


reg.coef_


# In[199]:


core_weather.corr()["target"]


# In[ ]:




