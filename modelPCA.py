import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
train = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/data1-train.csv")
test = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/data1-test.csv")
df = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/fisherscore/data1trainFS.csv")
df = df.loc[df['rank']<41]
features = df['features'].tolist()
x_train=train[features]
y_train = train['Churn']
x_test = test[features]
y_test = test['Churn']
#print(features)
#print(x_train.head())

pca = PCA()
pca.fit(x_train)
x_train = pca.transform(x_train)
print (pca.explained_variance_ratio_.cumsum())
