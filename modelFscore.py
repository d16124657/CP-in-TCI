import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/data1-train.csv")
test = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/data1-test.csv")
df = pd.read_csv("D:/DIT-2016-2019/NEW/Dissertation/data/fisherscore/data1trainFS.csv")
df = df.loc[df['rank']<17]
features = df['features'].tolist()
x_train=train[features]
y_train = train['Churn']
x_test = test[features]
y_test = test['Churn']
#print(features)
#print(x_train.head())

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report,roc_curve
from sklearn import metrics

classifiers = [['DecisionTree :',DecisionTreeClassifier()],
               ['RandomForest :',RandomForestClassifier()],
               ['Naive Bayes :', GaussianNB()],
               ['KNeighbours :', KNeighborsClassifier()],
               ['SVM :', SVC()],
               ['Neural Network :', MLPClassifier()],
               ['LogisticRegression :', LogisticRegression()],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['BaggingClassifier :', BaggingClassifier()],
               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],
               ['XGB :', XGBClassifier()],
               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]
predictions_df = pd.DataFrame()
predictions_df['actual_labels'] = y_test

for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    predictions_df[name.strip(" :")] = predictions
    print(name, accuracy_score(y_test, predictions))
    rocauc = metrics.roc_auc_score(y_test, classifier.predict(x_test))
    print('ROC-AUC score :',rocauc)
    report = classification_report(y_test, classifier.predict(x_test))
    print(report)

