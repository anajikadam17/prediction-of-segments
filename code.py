# --------------
#import the packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
# code starts here
seg=pd.read_csv(path,encoding='latin1')
categorical=seg.select_dtypes(include='O')
numerical=seg.select_dtypes(include=np.number)
print(categorical.head())
print(numerical.head())
# en=OneHotEncoder()
# en.fit_transform(categorical)
# categorical=pd.get_dummies(categorical)
df=pd.concat([numerical,pd.get_dummies(categorical)],axis=1)



# --------------
# code starts here
from xgboost import XGBClassifier
X=df.drop(columns=['customer id','segments'])
y=df['segments']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True)
classifier=XGBClassifier(random_state=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
f1=f1_score(y_test,y_pred,average='macro')
xgb_cr=classification_report(y_pred,y_test)
print(xgb_cr)
# code ends here


# --------------
from sklearn.model_selection import GridSearchCV
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}
# code start here
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,n_jobs=-1,verbose=4)
grid_search.fit(X_train,y_train)
grid_predictions=grid_search.predict(X_test)
grid_f1=f1_score(y_test,grid_predictions,average='macro')
report=classification_report(grid_predictions,y_test)
print(report)
# code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
# code starts here
model=RandomForestClassifier(random_state=2)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
f1=f1_score(y_test,y_pred,average='macro')
report=classification_report(y_test,y_pred)
print(report)


