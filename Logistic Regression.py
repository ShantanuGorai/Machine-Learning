import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

sns.set(style="whitegrid")
df = sns.load_dataset("titanic")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)
print("\nInformation")
print(df.info())

print("\nStatistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

plt.figure(figsize=(5,4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='class', hue='survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df['age'])
plt.title("Age Boxplot")
plt.show()

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()


df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df.drop(columns=['deck'], inplace=True)

df.dropna(inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

features = [
    'pclass',
    'sex',
    'age',
    'sibsp',
    'parch',
    'fare',
    'embarked'
]

X = df[features]
y = df['survived']
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded Features")
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

print("ROC AUC Score:", auc)
coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coef = coef.sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance")
print(coef)

plt.figure(figsize=(8,6))

sns.barplot(
    data=coef,
    x='Coefficient',
    y='Feature'
)

plt.title("Logistic Regression Coefficients")

plt.show()
print("\nFirst 10 Predictions")

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison)