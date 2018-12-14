# %%

import os
import numpy as np
import pandas as pd
from plotnine import *
from apyori import apriori


# %%
import socket
hostname = socket.gethostname()
hostname

# %%

data_file = os.getcwd()+'/data/anonymous-msweb.data'

attr_data, attr_id = [], []
visit_data = {}
user = 0

# extract data
with open(data_file, 'r') as f:
  for line in f:
    letter = line[0]
    d = line.split(',')
    if letter == 'A':
      attr_data.append([d[3][1:-1], d[4][1:-3] ])
      attr_id.append(int(d[1]))
    elif letter == 'C':
      user = int(d[2])
      visit_data[user] = []
    elif letter == 'V':
      visit_data[user].append(int(d[1]))
# %%
# create dataframe for attributes & visits
attr_pd = pd.DataFrame(
  attr_data,
  index = attr_id,
  columns = ['title', 'url']
)
visit_pd = pd.Series(visit_data)

# %%
df = visit_pd.apply(lambda x: len(x))
user_visit_df = pd.DataFrame(
  df, columns=['count']
)
user_visit_df
# %%

# gg = (ggplot(user_visit_df)
#   + aes(x='count')
#   + geom_histogram(binwidth=1)
#   # + geom_col(aes(x=user_visit_df[:100].index, y="count"))
#   + scale_x_continuous(breaks = range(0,20, 2))
#   # + theme(axis_text_x=element_text(rotation=40, ha="right"))
# )

# print(gg)
# gg.save('outputs/user_visit_count2.pdf')

# %%

# sum visits of all users
data = np.array([])
for key,val in visit_data.items():
  data = np.concatenate((data, val))

uniq_elem, counts = np.unique(data, return_counts=True)

stat_df = pd.DataFrame({
  'visit': counts,
  'vroot': uniq_elem
})


# %% show count of visits to same vroot
# gg = (ggplot(stat_df)
#   + aes(x='visit')
#   + geom_histogram(binwidth=25)
#   + scale_x_continuous(breaks = range(0, 12000, 500))
#   + theme(axis_text_x=element_text(rotation=40, ha="right"))
# )
# print(gg)
# gg.save('outputs/visit_count.pdf')

# %% show distribution of visits by user

# gg = (ggplot(stat_df)
#   + geom_col(aes(x="vroot", y="visit"))
#   + scale_x_continuous(breaks = range(1000,1301, 50))
#   + theme(axis_text_x=element_text(rotation=40, ha="right"))
# )

# print(gg)
# gg.save('outputs/visit_vroot.pdf')


# %% filter out vroot with only 1 visitor
one_vroot = uniq_elem[counts == 1]
multiple_vroot = uniq_elem[counts > 1]
target_attrs_df = attr_pd.loc[attr_pd.index.isin(multiple_vroot)]

# %% filter 1 user {41033} by attr 
dependent_df = visit_pd.apply(lambda x:  not(len(x) == 1 and \
  bool(set(x).intersection(one_vroot)) ))
visit_pd = visit_pd[dependent_df]

# %%
# make one hot encoding of hot movies
attr_list = np.sort(multiple_vroot)

visit_bool_data = np.stack([np.where(np.isin(attr_list,i), 1, 0) for i in visit_pd])

visit_full_df = pd.DataFrame(
  data=visit_bool_data,
  index=visit_pd.index,
  columns=attr_list
)

visit_full_df

# %%
# select rich features (264 -> 162) left
from sklearn.feature_selection import VarianceThreshold

percent = 0.999
sel = VarianceThreshold(percent * (1-percent))
visit_sel_value = sel.fit_transform(visit_full_df.values)

# %% row from 32710 -> 32608
# filtered all user without 1's

filters = np.where(np.sum(visit_sel_value, axis=1) == 0, False, True)

visit_sel_df = pd.DataFrame(
  visit_sel_value[filters],
  index = visit_full_df.index[filters]
)

visit_sel_df

# %%
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from apyori import apriori
%matplotlib inline

# %%
part_df = visit_sel_df[:20000]

# %%

# z = linkage(part_df, method='ward')

# fig, ax = plt.subplots(1,1, figsize=(50,20))
# ax.set_title("hierarchical clustering")
# ax.set_xlabel("distance")
# ax.set_ylabel("id")

# gg = dendrogram(z, labels=part_df.index, leaf_rotation=0, orientation='top', color_threshold=20, above_threshold_color='grey' )
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=15)
# plt.yticks(size=20)
# plt.savefig("outputs/dendro.png", format="png", dpi=80, bbox_inches='tight')

# %%

import sys
import multiprocessing

manager = multiprocessing.Manager()

# %%

def getTrans(num_cluster):
  cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
  cluster_data = cluster.fit_predict(part_df)

  user_group = [part_df.index[cluster_data == i] for i in range(num_cluster)]
  transactions = [visit_pd[visit_pd.index.isin(user_group[i])].values for i in range(num_cluster)]

  return (num_cluster, transactions)

pool = multiprocessing.Pool(4)
res = pool.map_async(getTrans, (i for i in range(25,26,5)) )
pool.close()
pool.join()

# %%
def getApiori(arg):
  support, conf, lis = arg
  l = []
  for item in lis:
    num_cluster, transactions = item
    results = [list(apriori(transactions[i], min_support=support, min_confidence=conf, min_length=2, percent=True)) for i in range(num_cluster)]
    results = [i for i in results if i]
    l.append((num_cluster, results) )
  return (("support", np.around(support,1),"conf", np.around(conf,1),l) )

pool = multiprocessing.Pool(10)
api = pool.map_async(getApiori, ((i, j, res.get()) for i in np.linspace(0.1,0.9,9) \
                                  for j in np.linspace(0.1,0.9,9) ))
pool.close()
pool.join()

# %%
api.get()

# %%
data = api.get()
d2 = [[item for sublist in i for item in sublist] for i in data[30][4][0][1]]
for i in d2:
  print(i)

# %%
# total 30 cluster, highest: 27

[len(i[4][0][1]) for i in api.get()]

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
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

N = 30
label_class = SpectralClustering(N, affinity='precomputed').fit_predict(y)

classify = [[] for _ in range(N)]
for i,j in zip(label_class, label):
  classify[i].append(j)

classify
# %%

# %%
d



