import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sns.set(style="whitegrid")
df = sns.load_dataset("tips")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)
print("\nInformation")
print(df.info())

print("\nStatistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())
plt.figure(figsize=(7,5))
sns.histplot(df["total_bill"], bins=25, kde=True)
plt.title("Distribution of Total Bill")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df["tip"], bins=25, kde=True)
plt.title("Distribution of Tip")
plt.show()
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)
plt.title("Total Bill vs Tip")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x="day", y="tip", data=df)
plt.title("Tip by Day")
plt.show()
sns.pairplot(df)
plt.show()
plt.figure(figsize=(6,5))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "category":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())
X = df.drop("tip", axis=1)
y = df["tip"]
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
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(7,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Tip")
plt.ylabel("Predicted Tip")
plt.title("Actual vs Predicted")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.show()


residuals = y_test - y_pred

plt.figure(figsize=(7,5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef = coef.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Importance")
print(coef)

plt.figure(figsize=(8,6))

sns.barplot(
    data=coef,
    x="Coefficient",
    y="Feature"
)

plt.title("Feature Coefficients")

plt.show()

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))