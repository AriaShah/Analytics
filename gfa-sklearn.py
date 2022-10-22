
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import seaborn as sns; sns.set()
from kneed import DataGenerator, KneeLocator
import csv


df = pd.read_csv('C:/Users/HP/Desktop/ir2.csv')
X_weighted =df.loc[:,['Latitude','Longitude','kgg']]

lat_long = X_weighted[X_weighted.columns[0:2]]
pop_size = X_weighted[X_weighted.columns[2]]
sample_weight = pop_size


# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(lat_long, sample_weight = pop_size)
    sse.append(kmeans.inertia_)

# Elbow plot
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

kl.elbow

# Final kmeans with optimal k
kmeans = KMeans(n_clusters = kl.elbow, max_iter=1000, init ='k-means++')

lat_long = X_weighted[X_weighted.columns[0:2]]
pop_size = X_weighted[X_weighted.columns[2]]
weighted_kmeans_clusters = kmeans.fit(lat_long, sample_weight = pop_size) # Compute k-means clustering.
X_weighted['cluster_label'] = kmeans.predict(lat_long, sample_weight = pop_size)

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
print(centers)

labels = X_weighted['cluster_label'] # Labels of each point

X_weighted.plot.scatter(x = 'Latitude', y = 'Longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
plt.title('Clustering GPS Co-ordinates to Form Regions - Weighted',fontsize=18, fontweight='bold')
plt.show()
