import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = make_blobs(n_samples=500, n_features=5, centers=3, cluster_std=1.5, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['label'] = y

sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='label', palette='Set2')
plt.show()
