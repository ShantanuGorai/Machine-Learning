import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

df = sns.load_dataset('iris')
print(df['species'].unique())

X = df.drop('species', axis = 1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, shuffle=True)

model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
print(model_knn.score(X_test, y_test))

model_svm = SVC(C = 30, kernel='rbf', gamma='auto')
model_svm.fit(X_train, y_train)
print(model_svm.score(X_test, y_test))

#Grid Search CV

classifier = GridSearchCV((model_svm), {
        'C' : [1, 10, 30],
        'kernel' : ['linear', 'rbf'],
    }, cv=5, return_train_score=False)

classifier.fit(X, y)
results = pd.DataFrame(classifier.cv_results_)

print(results[['param_C', 'param_kernel', 'mean_test_score']])

#Random search CV

classifier_random = RandomizedSearchCV(model_svm, {
    'C' : [1, 10, 30],
    'kernel' : ['linear', 'rbf'],
    },n_iter=4, cv=5, return_train_score=False)

classifier_random.fit(X, y)
results_random = pd.DataFrame(classifier_random.cv_results_)

print(results_random[['param_C', 'param_kernel', 'mean_test_score']])


