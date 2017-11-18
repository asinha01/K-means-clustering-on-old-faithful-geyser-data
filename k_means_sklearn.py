import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'x': [3.600, 1.800, 2.283, 3.333, 2.883, 4.533, 1.950, 1.833, 4.700, 3.600, 1.600, 4.350, 3.917, 4.200, 1.750, 1.800, 4.700, 2.167, 4.800, 1.750],
    'y': [79, 54, 62, 74, 55, 85, 51, 54, 88, 85, 52, 85, 84, 78, 62, 51, 83, 52, 84, 47]
})

while True:
    try:
        n = int(raw_input("enter the number of clusters(maximum limit is 3 clusters):-\t"))
    except ValueError:
        print("Sorry, invalid entry. Retry by putting an integer between 1 and 3")
        continue
    else:
        break


kmeans = KMeans(n_clusters=n)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 5)
plt.ylim(0, 100)
plt.title("k-means clustering with " + str(n) + " clusters" )
plt.xlabel('Duration')
plt.ylabel('Wait')
plt.show()