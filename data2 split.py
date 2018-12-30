import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#  loading into Panda dataframe
data2 = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/dd.csv")
print ("\nMissing values :  ", data2.isnull().sum().values.sum())
print ("Rows     : " ,data2.shape[0])
print(data2.dtypes)
#train,test = train_test_split(data2,test_size = .5 ,random_state = 111)
#train.to_csv('../data/data2-train.csv')
#test.to_csv('../data/data2-test.csv')
