# %%

import numpy as np
import pandas as pd
from plotnine import *
from apyori import apriori
from preprocessing import preprocess
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
import multiprocessing
from matplotlib import pyplot as plt
%matplotlib inline

visit_pd, visit_sel_df, target_attrs_df = preprocess()

# %%
part_df = visit_sel_df[:50000]

# %%

def getTrans(num_cluster):
  cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
  cluster_data = cluster.fit_predict(part_df)

  user_group = [part_df.index[cluster_data == i] for i in range(num_cluster)]
  transactions = [visit_pd[visit_pd.index.isin(user_group[i])].values for i in range(num_cluster)]

  dist = [np.count_nonzero(cluster.labels_ == i) for i in range(num_cluster)]
  return (num_cluster, transactions, dist)

pool = multiprocessing.Pool(1)
res = pool.map_async(getTrans, (i for i in range(3,30,3)) )
pool.close()
pool.join()

# %%
res.get()

# %%
def getApiori(arg):
  def getrule(nested_rules):
    a = [[item for sublist in i for item in sublist] for i in nested_rules]
    flat_list = [item for sublist in a for item in sublist]
    raw_rules = []
    for k in flat_list:
      if not k in raw_rules:
        raw_rules.append(k)
    return raw_rules

  support, conf, lis = arg
  l = []
  for item in lis:
    num_cluster, transactions, dist = item
    if num_cluster == 24:
      results = [list(apriori(transactions[i], min_support=support, min_confidence=conf, min_length=2, percent=True)) for i in range(num_cluster)]
      results = [i for i in results if i]
      flat_res = getrule(results)
      # if support==0.1 and conf==0.1:
      l.append((num_cluster, len(flat_res) )) #, dist) )
  return (("support", np.around(support,1),"conf", np.around(conf,1), l))

pool = multiprocessing.Pool(10)
api = pool.map_async(getApiori, ((i, j, res.get()) for i in np.linspace(0.1,0.9,9) \
                                  for j in np.linspace(0.1,0.9,9) ))
pool.close()
pool.join()
# %%
api.get()

# %% extract unique rules on 0.2,0.3
nested_rules = api.get()[11][4][0][1]

a = [[item for sublist in i for item in sublist] for i in nested_rules]
flat_list = [item for sublist in a for item in sublist]
raw_rules = []
for k in flat_list:
  if not k in raw_rules:
    raw_rules.append(k)

len(raw_rules)

# %%

# %%
target_attrs_df.loc[target_attrs_df.index.isin([1017, 1074, 1009])]

# %%
ps1 = []
for leng in range(len(d2)):
  for src in d2[leng]:
    for i in d2[leng+1:]:
      for j in i:
        if j==src:
          if not j in ps1:
            ps1.append(j)

for i in ps1:
  print(i)

# %% get clustering of features

# %%
from itertools import combinations

label = attr_pd.index #294
dicts = dict(zip(label, range(len(label))))
site_data = visit_pd.values

y = np.zeros([len(label), len(label)])

add = 0
for i in site_data:
  for j in list(combinations(i, 2)):
    add += 1
    y[dicts[j[0]], dicts[j[1]]] += 1
    y[dicts[j[1]], dicts[j[0]]] += 1

print(add)
print(y.shape)
print(~np.all(y == 0, axis=1))
# %%
# empty_col = np.where(np.sum(y, axis=0) == 0)[0]
exist_col = ~np.all(y == 0, axis=1)
y = y[exist_col]
y = y[:, exist_col]
label = label[exist_col]

print(y.shape, label.shape)



# %%
d1 = [1698, 2077, 1968, 7077, 2558, 1557, 2706, 1564, 1073, 1772, 920, 2049, 756, 1175, 536, 895, 2227]
d1.sort(reverse=True)
d2 = [4314,1040,682,648,616,876,536,895,602,1009,875,758,416,675,411,780,258,544,436,256,694,920,405,2049,408,314,755,318,472,449,312,354,511,2227,318,262,242,834,524,1772,348,395,580,205,313]
d2.sort(reverse=True)
dist17 = pd.DataFrame({
  'y': d1, 'x': list(range(1,len(d1)+1))
})
dist45 = pd.DataFrame({
  'y': d2, 'x': list(range(1,len(d2)+1))
})

# %%
gg = (ggplot(dist17)
  + geom_col(aes(x='x', y="y"))
  + scale_x_continuous(name='clusters')
  + scale_y_continuous(name='size')
  + ggtitle('cluster = 17')
)
print(gg)
gg.save('outputs/dist17.png')

# %%
gg = (ggplot(dist45)
  + geom_col(aes(x='x', y="y"))
  + scale_x_continuous(name='clusters')
  + scale_y_continuous(name='size')
  + ggtitle('cluster = 45')
)
print(gg)
gg.save('outputs/dist45.png')





























