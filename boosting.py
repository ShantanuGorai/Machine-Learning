from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

df = sns.load_dataset('iris')
X = df.drop('species', axis = 1)
y = df['species']

le = LabelEncoder()

y_encoded = le.fit_transform(y)
X_test, X_train, y_test, y_train = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded, shuffle=True)

#ada boost

ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#gradient boosting

grad_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
grad_model.fit(X_train, y_train)
y_pred_grad = grad_model.predict(X_test)

accuracy_grad = accuracy_score(y_test, y_pred_grad)
print(accuracy_grad)

#xg boostig

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

