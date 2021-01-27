import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from image_data_module import TrainTestData
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import math

data_module = TrainTestData()
(X_train, Y_train), (X_test, Y_test) = data_module.get_train_test_salience()

salience_values = np.concatenate((Y_train, Y_test))
counts = Counter(salience_values)
salience_values = []
for key in counts:
    appearances = int(counts[key]/3)
    for i in range(0, appearances):
        salience_values.append(key)
salience_values = np.asarray(salience_values)

print("Normalverteilungstest", stats.normaltest(salience_values))
'''
# Density Plot / Histogram of all salience values
ax = sns.distplot(salience_values, hist=True, kde=False,
                  bins=int(180/5), color='darkblue',
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 3})
ax.set(xlabel='Salienzwert', ylabel='Häufigkeit')
plt.show()
'''

print("shape:", salience_values.shape, "min:", np.min(salience_values),
      "max:", np.max(salience_values), "std:", np.std(salience_values),
      "mean:", np.mean(salience_values), "median:", np.median(salience_values))
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
print("cluster 0 var:", np.var(class_0), "cluster 0 std:", np.std(class_0))
print("cluster 1 var:", np.var(class_1), "cluster 1 std:", np.std(class_1))

print("mean 0:", kmeans.cluster_centers_[0]-np.std(class_0))
print("mean 1:", kmeans.cluster_centers_[1]-np.std(class_1))
print("SPLIT (KMEANS):", (kmeans.cluster_centers_[0]-np.std(class_0)+
                          kmeans.cluster_centers_[1]+np.std(class_1))/2)

print(5*"#", "EXPECTATION MAXIMIZATION", 5*"#")

expectation_maximization = GaussianMixture(n_components=2, random_state=0).fit(salience_values)
print("EM mixture components centers for c=2:", expectation_maximization.means_)
print("EM mixture variances:", expectation_maximization.covariances_)
print("EM std 0:", math.sqrt(expectation_maximization.covariances_[0][0]),
      "EM std 1:", math.sqrt(expectation_maximization.covariances_[1][0]))
print("mean 0:", expectation_maximization.means_[0]-math.sqrt(expectation_maximization.covariances_[0][0]))
print("mean 1:", expectation_maximization.means_[1]+math.sqrt(expectation_maximization.covariances_[1][0]))
print("SPLIT (EM):", (expectation_maximization.means_[0]-math.sqrt(expectation_maximization.covariances_[0][0])+
expectation_maximization.means_[1]+math.sqrt(expectation_maximization.covariances_[1][0]))/2)
