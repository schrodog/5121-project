# %%
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
import numpy as np

# %%

label = [0,1,2,3,5,7,9,13,16,18,24] #7
d = dict(zip(label, range(len(label))))

x = np.array([np.random.choice(label,3, replace=False) for _ in range(30)])

# %%
y=np.zeros([len(label), len(label)])

from itertools import combinations

for i in x:
  for j in list(combinations(i, 2)):
    y[d[j[0]],d[j[1]]] += 1
    y[d[j[1]],d[j[0]]] += 1

from sklearn.preprocessing import minmax_scale

print(x)
print(label)
print(y)

# y = minmax_scale(y)

# for i in range(len(y)):
#   y[i,i] = 1
  
# print(y)

# %%
SpectralClustering(3, affinity='precomputed').fit_predict(y)




# %%

SpectralClustering(3, affinity='precomputed').fit_predict(y)


# %%
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
X, _ = load_digits(return_X_y=True)
print(X.shape)

embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(X[:100])
print(X_transformed.shape)

# %%


# %%
mat = np.matrix([[1.,.1,.6,.4],[.1,1.,.1,.2],[.6,.1,1.,.7],[.4,.2,.7,1.]])
SpectralClustering(2, affinity='precomputed').fit_predict(mat)

# %%
eigen_values, eigen_vectors = np.linalg.eigh(mat)
KMeans(n_clusters=2, init='k-means++').fit_predict(eigen_vectors[:, 2:4])

# %%
DBSCAN(min_samples=1).fit_predict(mat)
















