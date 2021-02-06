#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


cars=pd.read_csv("E:/Decode_Lectures/Case Study/Case Study_6_ Predicting Price of Car/CarPrice_Assignment.csv")
cars


# In[3]:


cars.info()


# In[4]:


cars.dtypes


# In[5]:


cars.shape


# In[6]:


# I want all numeric columns
cars_numeric=cars.select_dtypes(include=["float64","int64"])
cars_numeric.head(5)


# In[11]:


#create a pairplot
import seaborn as sns
sns.pairplot(cars_numeric)
plt.show()


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


# To check correlation between columns
plt.figure(figsize=(16,8))
sns.heatmap(cars,annot=True)
plt.show()


# In[15]:


#I am fetching a column name "CarName" & first 30 rows
cars["CarName"][:30]
#First is company name then car Name Exm. "alfa-romero giulia" here "alfa-romero" is company name & "giulia" is car name
# I want only car names


# In[16]:


# I want only car names
carnames=cars["CarName"].apply(lambda x : x.split(" ")[0])
carnames.head(30)


# In[17]:


# I mace a separet column in Cars data as "carnames"
cars["Car_Company"]=cars["CarName"].apply(lambda x : x.split(" ")[0])
cars.head(30)


# In[18]:


cars["Car_Company"][:50]


# In[19]:


cars["Car_Company"][50:100]


# In[20]:


cars["Car_Company"][150:]


# spelling mistake "vm,vokswagen,toyouta,maxda,Nissan,porcshce

# In[ ]:


#make a correction on the spelling mistakes
#cars.loc[(cars["Car_Company"]=="vm")|(cars["Car_Company"]=="vokswagen"),"Car_Company"]="volkswagen"
#cars.loc[(cars["Car_Company"]=="toyouta"),"Car_Company"]="toyota"
#cars.loc[(cars["Car_Company"]=="maxda"),"Car_Company"]="maxza"
#cars.loc[(cars["Car_Company"]=="Nissan"),"Car_Company"]="nissan"
#cars.loc[(cars["Car_Company"]=="porcshce"),"Car_Company"]="porche"


# In[ ]:


#cars["Car_Company"][50:100]


# In[21]:


cars.loc[(cars["Car_Company"]=="vw")|(cars["Car_Company"]=="vokswagen"),"Car_Company"]="volkswagen"


# In[22]:


cars.loc[(cars["Car_Company"]=="toyouta"),"Car_Company"]="toyota"


# In[23]:


cars["Car_Company"][150:]


# In[24]:


cars.loc[(cars["Car_Company"]=="maxda"),"Car_Company"]="mazda"
cars.loc[(cars["Car_Company"]=="Nissan"),"Car_Company"]="nissan"


# In[25]:


cars["Car_Company"][50:100]


# In[26]:


cars.loc[(cars["Car_Company"]=="porcshce"),"Car_Company"]="porche"


# In[27]:


cars["Car_Company"].value_counts()


# In[28]:


#We drop "CarName"this column for analysis purpose
cars=cars.drop("CarName",axis=1)


# In[29]:


cars.info()


# In[30]:


cars.head()


# In[31]:


cars.columns


# In[32]:


cars["cylindernumber"].value_counts()


# In[33]:


cars["doornumber"].value_counts()


# In[34]:


# In cars data set there is 2 columns "doornumber" & "cylindernumber" we need to convert it from strings to number
# First we need to create a column
def num_map(x):
    return x.map({"four":4,"six":6,"five":5,"eight":8,"two":2,"twelve":12,"three":3})


# In[35]:


#Apply the Function "num_map" on column
cars[["cylindernumber","doornumber"]]=cars[["cylindernumber","doornumber"]].apply(num_map)


# In[36]:


#Ther are other String column We cant convert it but we can use as one-not-encording
cars.select_dtypes(include=["object"])


# In[37]:


cars_categorical=cars.select_dtypes(include=["object"])


# In[38]:


cars_categorical.head()


# In[39]:


#Create a dummi data set
cars_dummies=pd.get_dummies(cars_categorical,drop_first=True)


# In[40]:


cars_dummies.head()


# In[41]:


#I want to delete all cetagoricel data "cars_categorical" First we convert it into list the we drop it
cars=cars.drop(list(cars_categorical.columns),axis=1)
cars


# In[ ]:


#I want to cocant "Cars" data with "cars_dummies"


# In[42]:


cars.shape


# In[43]:


cars_dummies.shape


# In[44]:


cars=pd.concat([cars,cars_dummies],axis=1)


# In[45]:


cars.shape


# In[46]:


cars.head(5)


# In[47]:


cars.info()


# In[48]:


# We drop "car_ID" Beacuse it won't help in prediction     
cars=cars.drop("car_ID",axis=1)


# In[49]:


cars.head(5)


# # Modeling

# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


df_train,df_test=train_test_split(cars,train_size=0.7,test_size=0.3,random_state=100)


# In[52]:


from sklearn.preprocessing import StandardScaler


# In[53]:


scaler=StandardScaler()


# In[54]:


cars.columns


# In[55]:


#Creating a columnlinst which is add previously
varlist=['symboling','doornumber','carlength','carwidth','cylindernumber',
       'enginesize','boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg']


# In[56]:


df_train[varlist]=scaler.fit_transform(df_train[varlist])


# In[57]:


df_train.head()


# In[58]:


y_train=df_train.pop("price")


# In[59]:


y_train


# In[60]:


x_train=df_train


# In[61]:


from sklearn.linear_model import LinearRegression


# In[62]:


lm=LinearRegression()


# In[63]:


lm.fit(x_train,y_train)


# In[64]:


lm.coef_


# In[65]:


lm.intercept_


# In[66]:


from sklearn.feature_selection import RFE


# In[67]:


lm=LinearRegression()
rfe1=RFE(lm,15)
rfe1.fit(x_train,y_train)


# In[68]:


rfe1.ranking_


# In[69]:


rfe1.support_


# In[70]:


import statsmodels.api as sm


# In[71]:


#List of top 15 columns
col1=x_train.columns[rfe1.support_]


# In[72]:


x_train_rfe1=x_train[col1]


# In[73]:


x_train_rfe1=sm.add_constant(x_train_rfe1)


# In[75]:


x_train_rfe1.head(5)


# In[83]:


lm1=sm.OLS(y_train,x_train_rfe1).fit()


# In[84]:


lm1.summary()


# In[85]:


#Creating empty data frame
vif=pd.DataFrame()


# In[87]:


vif["Features"]=x_train_rfe1.columns
vif


# In[88]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[89]:


vif["VIF"]=[variance_inflation_factor(x_train_rfe1.values,i)for i in range(x_train_rfe1.shape[1])]


# In[95]:


#I am round off this value to 2 decimel 
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[96]:


lm=LinearRegression()
rfe2=RFE(lm,10)
rfe2.fit(x_train,y_train)


# In[97]:


col2=x_train.columns[rfe2.support_]
x_train_rfe2=x_train[col2]
x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# In[98]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[99]:


x_train_rfe2.drop("Car_Company_subaru",axis=1,inplace=True)


# In[ ]:


#x_train_rfe2.drop("carbody_hardtop",axis=1,inplace=True)


# In[100]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# In[105]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[112]:


x_train_rfe2.drop("const",axis=1,inplace=True)


# In[113]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# In[114]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[115]:


x_train_rfe2.drop("carbody_hardtop",axis=1,inplace=True)


# In[116]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# In[117]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[118]:


x_train_rfe2.drop("enginetype_ohcf",axis=1,inplace=True)


# In[119]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# In[120]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[121]:


x_train_rfe2.drop("const",axis=1,inplace=True)


# In[122]:


vif=pd.DataFrame()
vif["Features"]=x_train_rfe2.columns
vif["VIF"]=[variance_inflation_factor(x_train_rfe2.values,i)for i in range(x_train_rfe2.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF",ascending=False)
vif


# In[123]:


x_train_rfe2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train,x_train_rfe2).fit()
print(lm2.summary())


# # Making Prediction

# In[124]:


df_test[varlist]=scaler.transform(df_test[varlist])


# In[125]:


y_test=df_test.pop("price")


# In[126]:


x_test=df_test


# In[127]:


col2


# In[133]:


x_test_rfe2=x_test[col2]


# In[142]:


x_test_rfe2


# In[135]:


#x_test_rfe2=x_test_rfe2.drop(["enginetype_ohcf","carbody_hardtop","Car_Company_subaru"],axis=1)


# In[143]:


x_test_rfe2=sm.add_constant(x_test_rfe2)


# In[144]:


y_pred=lm2.predict(x_test_rfe2)


# In[145]:


plt.scatter(y_test,y_pred)
plt.show()


# In[146]:


from sklearn.metrics import r2_score


# In[147]:


r2_score(y_test,y_pred)


# In[148]:


cars[col2]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




