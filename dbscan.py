import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['dbscan_clusters'] = dbscan_labels
sns.scatterplot(
    x = df['Feature1'],
    y = df['Feature2'],
    hue = df['dbscan_clusters'],
    palette='tab10'
)

plt.show()