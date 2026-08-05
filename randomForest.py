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
from sklearn.ensemble import RandomForestClassifier

df = sns.load_dataset('iris')
X = df.drop('species', axis = 1)
y = df['species']

le = LabelEncoder()

y_encoded = le.fit_transform(y)
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_encoded, shuffle=True)

rf_model = RandomForestClassifier(
    n_estimators=100, #number of trees
    max_depth=None,#let trees grow fully
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)