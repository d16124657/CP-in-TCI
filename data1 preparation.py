#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#  loading into Panda dataframe
data1 = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# looking at the features we have
#data1.head()
#data1.to_csv('../data/a.csv')

#print("Rows     : ", data1.shape[0])
#print("Columns  : ", data1.shape[1])
#print("\nFeatures : \n", data1.columns.tolist())
#print("\nMissing values :  ", data1.isnull().sum().values.sum())
#print("\nUnique values :  \n", data1.nunique())
#print(data1.dtypes)
#convert to float type
data1['TotalCharges'] = data1["TotalCharges"].replace(" ",np.nan)
#data1["TotalCharges"] = data1["TotalCharges"].astype(float)
#data1["TotalCharges"] = pd.to_numeric(data1.TotalCharges, errors='coerce')
#print(data1.isnull().sum())

# There are only 11 missing values, all of them for the TotalCharges column.
# This values are actually a blank space in the csv file and are exclusive for customers with zero tenure.
# It's possible to concluded that they are missing due to the fact that the customer never paied anything to the company.
# We will impute this missing values with zero:
data1['TotalCharges'] = data1['TotalCharges'].replace(np.nan, 0).astype(float)
print(data1.isnull().sum())

#print(data1.info())

#encoder = OneHotEncoder(sparse=False,dtype='object')
#encoder.fit_transform(data1[['gender','InternetService']])

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
idColumn=['customerID']
targetColumn=['Churn']
#categoricalColumns=["gender","SeniorCitizen","Partner","Dependents","PhoneService",
                   # "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                    #"DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                   # "Contract","PaperlessBilling","PaymentMethod"]
catagoricalColumns=data1.nunique()[data1.nunique() < 6].keys().tolist()
catagoricalColumns=[x for x in catagoricalColumns if x not in targetColumn]
#numerical columns
numColumns  = [x for x in data1 if x not in catagoricalColumns + targetColumn + idColumn]

#Binary columns with 2 values
binColumns = data1.nunique()[data1.nunique() == 2].keys().tolist()
#Label encoding Binary columns
le = LabelEncoder()
for i in binColumns :
    data1[i] = le.fit_transform(data1[i])

#Columns more than 2 values
multiColumns = [i for i in catagoricalColumns if i not in binColumns]
#Duplicating columns for multi value columns
data1 = pd.get_dummies(data = data1,columns=multiColumns)

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data1[numColumns])
scaled = pd.DataFrame(scaled,columns=numColumns)

#dropping original values merging scaled values for numerical columns
data1_og = data1.copy()
data1=data1.drop(columns = numColumns,axis = 1)
data1=data1.merge(scaled,left_index=True,right_index=True,how = "left")
print ("\nMissing values :  ", data1.isnull().sum().values.sum())
#print(data1.head())

#train,test = train_test_split(data1,test_size = .5 ,random_state = 111)
#train.to_csv('../data/data1-train.csv')
#test.to_csv('../data/data1-test.csv')
#data1["MultipleLines"]=data1["MultipleLines"].astype('int64')


