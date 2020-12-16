import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

df = pd.read_csv("salience_csv/Landmarken_mit_Faktorenwerten.csv",
                 sep=";",
                 decimal=",")

salience_values = df["Salienz_gerundet"].to_numpy()
print("mean:", np.mean(salience_values), "median:", np.median(salience_values))
salience_values = salience_values.reshape(-1, 1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(salience_values)
print("K-means cluster centers for c=2:", kmeans.cluster_centers_)
print("mean:", (kmeans.cluster_centers_[0]+kmeans.cluster_centers_[1])/2)

class_0 = []
class_1 = []
for idx, label in enumerate(kmeans.labels_):
    if label == 0:
        class_0.append(salience_values[idx])
    else:
        class_1.append(salience_values[idx])
print("cluster 0 var:", np.var(class_0), "cluster 1 var:", np.var(class_1))
print("cluster 0 std:", np.std(class_0), "cluster 1 std:", np.std(class_1))

expectation_maximization = GaussianMixture(n_components=2, random_state=0).fit(salience_values)
print("EM mixture components centers for c=2:", expectation_maximization.means_)
print("EM mixture variances:", expectation_maximization.covariances_)
print("mean 0 neg:", expectation_maximization.means_[0]-expectation_maximization.covariances_[0][0],
      "mean 0 pos:", expectation_maximization.means_[0]+expectation_maximization.covariances_[0][0])
print("mean 1 neg:", expectation_maximization.means_[1]-expectation_maximization.covariances_[1][0],
      "mean 1 pos:", expectation_maximization.means_[1]+expectation_maximization.covariances_[1][0])
