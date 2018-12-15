# %%

from time import time
import numpy as np
from scipy import ndimage
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
% matplotlib inline

# %%

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

print(digits['DESCR'])

# %%
np.random.seed(0)

def nudge_images(X, y):
  shift = lambda x: ndimage.shift(x.reshape((8,8)), 
            .3 * np.random.normal(size=2), mode='constant').ravel()
  X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
  Y = np.concatenate([y, y], axis=0)
  return X,Y

X, y = nudge_images(X, y)

print(X,y)
# print(X.shape, y.shape)

# %%
def plot_clustering(X_red, labels, title=None):
  x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
  X_red = (X_red - x_min) / (x_max - x_min)

  plt.figure(figsize=(6,4))
  for i in range(X_red.shape[0]):
    plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
              color=plt.cm.nipy_spectral(labels[i] / 10.),
              fontdict={'weight': 'bold', 'size': 9})

  plt.xticks([])
  plt.yticks([])
  if title is not None:
    plt.title(title, size=17)
  plt.axis('off')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  # plt.savefig(fname=title+'.png')

# %%
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete', 'single'):
  clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
  t0 = time()
  clustering.fit(X_red)
  print("%s :\t%.2fs" %(linkage, time()-t0))
  plot_clustering(X_red, clustering.labels_, linkage+" linkage")

plt.show()

# %%

X_red

# %%

lis = [np.array([16,27]), np.array([12]), np.array([27,29])]
b = np.array([12,16,20,24,27,29])


np.stack([np.where(np.isin(b,i), 1, 0) for i in lis])





