import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
    
plt.plot(K_range, inertia, marker='o')
plt.show()

kmeans_final = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df['clusters'] = cluster_labels
sns.scatterplot(
    x = df['Feature1'],
    y = df['Feature2'],
    hue = df['clusters'],
    palette='viridis'
)

plt.show()
