# %%

import os
import numpy as np
import pandas as pd
from plotnine import *


# %%
data_file = os.getcwd()+'/anonymous-msweb.data'

attr_data = []
visit_data = {}
user = 0

# extract data
with open(data_file, 'r') as f:
  for line in f:
    letter = line[0]
    d = line.split(',')
    if letter == 'A':
      attr_data.append([int(d[1]), d[3][1:-1], d[4][1:-3] ])
    elif letter == 'C':
      user = int(d[2])
      visit_data[user] = []
    elif letter == 'V':
      visit_data[user].append(int(d[1]))

# %%

attr_pd = pd.DataFrame(
  attr_data,
  index = range(len(attr_data)),
  columns = ['id', 'title', 'url']
)

visit_pd = pd.Series(visit_data)

visit_pd

# %%

data = np.array([])
for key,val in visit_data.items():
  data = np.concatenate((data, val))

uniq_elem, counts = np.unique(data, return_counts=True)

df = pd.DataFrame({
  'visit': counts,
  'vroot': uniq_elem
})

# %%

gg = (ggplot(df)
  + geom_col(aes(x="vroot", y="visit"))
  + scale_x_continuous(breaks = range(1000,1301, 20))
)

print(gg)
gg.save()

# %%
top_vroot = uniq_elem[counts > 1000]

attr_pd.loc[attr_pd['id'].isin(top_vroot)]


# %%
# make one hot encoding
attr_list = np.sort(attr_pd['id'])

visit_bool_data = np.stack([np.where(np.isin(attr_list,i), 1, 0) for i in visit_pd])

visit_full_df = pd.DataFrame(
  visit_bool_data,
  columns=attr_list
)

# %%
from sklearn import manifold

print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2, n_jobs=3).fit_transform(visit_bool_data)
print("Done.")

X_red

# %%
(ggplot(pd.DataFrame({'x': X_red[:,0], 'y':X_red[:,1]} ))
  + geom_point(aes(x='x', y='y'))
)

# %%
from matplotlib import pyplot as plt

def plot_clustering(X_red, labels, title=None):
  x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
  X_red = (X_red - x_min) / (x_max - x_min)

  plt.figure(figsize=(6,4))
  # for i in range(X_red.shape[0]):
    # plt.text(X_red[i, 0], X_red[i, 1], str(1),
    #           color=plt.cm.nipy_spectral(labels[i] / 10.),
    #           fontdict={'weight': 'bold', 'size': 9})
  for i in range(X_red.shape[0]):
    plt.plot(X_red[i,0], X_red[i,1], 'ro', markersize=1, color=plt.cm.nipy_spectral(labels[i]/10.))
  plt.xticks([])
  plt.yticks([])
  if title is not None:
    plt.title(title, size=17)
  plt.axis('off')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  # plt.savefig(fname=title+'.png')  

from sklearn.cluster import AgglomerativeClustering
from time import time

linkage = 'ward'

clustering = AgglomerativeClustering(linkage=linkage, n_clusters=17)
t0 = time()
print('start')
clustering.fit(X_red)

print("%s :\t%.2fs" %(linkage, time()-t0))
plot_clustering(X_red, clustering.labels_, linkage+" linkage")

plt.show()

# %%
clustering.labels_ == 0












