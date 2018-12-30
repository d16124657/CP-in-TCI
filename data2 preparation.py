#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#  loading into Panda dataframe
data2 = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/Telecom_customer churn.csv")
#looking at the features we have
#data2.head()


#data cleaning
data2=data2.drop(['crclscod','numbcars','lor','dwllsize','HHstatin','ownrent','dwlltype'], axis=1)
data2.replace('  ',np.nan)
data2['income']=data2['income'].replace(np.nan,"5.78")
data2['adults']=data2['adults'].replace(np.nan,"2.53")

data2['infobase']=data2['infobase'].replace('  ',np.nan)
data2['infobase']=data2['infobase'].replace(np.nan,'unknow')
data2['hnd_webcap']=data2['hnd_webcap'].replace('  ',np.nan)
data2['hnd_webcap']=data2['hnd_webcap'].replace(np.nan,'unknow')
data2['prizm_social_one']=data2['prizm_social_one'].replace('  ',np.nan)
data2['prizm_social_one']=data2['prizm_social_one'].replace(np.nan,'unknow')

data2=data2.dropna(axis=0, how='any')

#data2.to_csv('../data/a.csv')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
idColumn=['Customer_ID']
targetColumn=['churn']
a=['adults']
e=['ethnic']
ar=['area']
catagoricalColumns=data2.nunique()[data2.nunique() < 7].keys().tolist()
catagoricalColumns=[x for x in catagoricalColumns if x not in targetColumn + a]

#numerical columns
numColumns  = [x for x in data2 if x not in catagoricalColumns + targetColumn + idColumn +e +ar]

#Binary columns with 2 values
binColumns = data2.nunique()[data2.nunique() == 2].keys().tolist()
#Label encoding Binary columns
le = LabelEncoder()
for i in binColumns :
    data2[i] = le.fit_transform(data2[i])

#Columns more than 2 values
multiColumns = [i for i in catagoricalColumns if i not in binColumns]
#Duplicating columns for multi value columns
data2 = pd.get_dummies(data = data2,columns=multiColumns)
data2 = pd.get_dummies(data = data2,columns=e)
data2 = pd.get_dummies(data = data2,columns=ar)

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data2[numColumns])
scaled = pd.DataFrame(scaled,columns=numColumns)
print ("Rows     : " ,scaled.shape[0])
scaled=scaled.dropna(axis=0, how='any')
#print ("\nMissing values :  ", scaled.isnull().sum().values.sum())
print ("Rows     : " ,data2.shape[0])
data2_og = data2.copy()
data2=data2.drop(columns = numColumns,axis = 1)
scaled.to_csv('../data/s.csv')
data2.to_csv('../data/d.csv')
#dropping original values merging scaled values for numerical columns


data2=data2.merge(scaled,left_index=True,right_index=True,how = "left")
print ("\nMissing values :  ", data2.isnull().sum().values.sum())
#print(data2.head())
#print ("Rows     : " ,data2.shape[0])
#print ("Columns  : " ,data2.shape[1])
#print ("\nFeatures : \n" ,data2.columns.tolist())
#print ("\nMissing values :  ", data2.isnull().sum().values.sum())
#print ("\nUnique values :  \n",data2.nunique())
#print(data2.dtypes)

#data2=data2.dropna(axis=0, how='any')
print ("Rows     : " ,data2.shape[0])
#train,test = train_test_split(data2,test_size = .5 ,random_state = 1111)
#train.to_csv('../data/data2-train.csv')
#test.to_csv('../data/data2-test.csv')


