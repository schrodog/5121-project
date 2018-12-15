# %%
import numpy as np
import pandas as pd
import os
from plotnine import *
from sklearn.feature_selection import VarianceThreshold

def preprocess():
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
  # get count of site visting per user
  # df = visit_pd.apply(lambda x: len(x))
  # user_visit_df = pd.DataFrame(
  #   df, columns=['count']
  # )
  # %%

  # gg = (ggplot(user_visit_df)
  #   + aes(x='count')
  #   + geom_histogram(binwidth=1)
  #   # + geom_col(aes(x=user_visit_df[:100].index, y="count"))
  #   + coord_cartesian(xlim=(0,20))
  #   + scale_x_continuous(breaks = range(0,20, 1), name="visit count")
  #   + scale_y_continuous(name='num of users')
  #   # + theme(axis_text_x=element_text(rotation=40, ha="right"))
  # )
  # print(gg)
  # gg.save('outputs/user_visit_count.png')

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

  # np.count_nonzero(stat_df['visit'] < 5)
  # gg = (ggplot(stat_df)
  #   + aes(x='visit')
  #   + geom_histogram(binwidth=1)
  #   + coord_cartesian(xlim=(0,100))
  #   + scale_x_continuous(breaks = range(0, 100, 10), name='site visit count')
  #   + scale_y_continuous(name='num of sites')
  #   + theme(axis_text_x=element_text(rotation=40, ha="right"))
  # )
  # print(gg)

  # gg.save('outputs/site_visit_count.png')

  # %% show distribution of visits by user

  # gg = (ggplot(stat_df)
  #   + geom_col(aes(x="vroot", y="visit"))
  #   + scale_x_continuous(breaks = range(1000,1301, 50), name='visit count')
  #   + scale_y_continuous(name='num of users')
  #   + theme(axis_text_x=element_text(rotation=40, ha="right"))
  # )
  # print(gg)
  # gg.save('outputs/visit_vroot.pdf')



  # %% filter out vroot with only 1 visitor
  one_vroot = uniq_elem[counts <= 4]
  multiple_vroot = uniq_elem[counts > 4]
  target_attrs_df = attr_pd.loc[attr_pd.index.isin(multiple_vroot)]

  # %% filter 10 users
  dependent_df = visit_pd.apply(lambda x:  not(len(x) == 1 and \
    bool(set(x).intersection(one_vroot)) ))
  visit_pd = visit_pd[dependent_df]

  # %%
  # make one hot encoding of hot sites
  attr_list = np.sort(multiple_vroot)

  visit_bool_data = np.stack([np.where(np.isin(attr_list,i), 1, 0) for i in visit_pd])

  visit_full_df = pd.DataFrame(
    data=visit_bool_data,
    index=visit_pd.index,
    columns=attr_list
  )

  # %%
  # select rich features (231 -> 162) left

  percent = 0.999
  sel = VarianceThreshold(percent * (1-percent))
  visit_sel_value = sel.fit_transform(visit_full_df.values)

  # %% row from 32700 -> 32608
  # filtered all user without 1's

  filters = np.where(np.sum(visit_sel_value, axis=1) == 0, False, True)

  visit_sel_df = pd.DataFrame(
    visit_sel_value[filters],
    index = visit_full_df.index[filters],
    columns=attr_list[sel.get_support()]
  )

  # one-hot encoded, description of labels
  return (visit_pd, visit_sel_df, target_attrs_df)











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

# from sklearn.manifold import MDS
# embedding = MDS(n_components=2)
# x_transformed = embedding.fit_transform(visit_sel_df.values[:40000])

# x_transformed.shape

# %%
# from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

# kmeans = KMeans(n_clusters=30, random_state=5, n_jobs=-1).fit(x_transformed)

# [np.count_nonzero(kmeans.labels_ == i) for i in range(30)]





# %%
# from sklearn.cluster import SpectralClustering, KMeans, DBSCAN

# N = 30
# label_class = SpectralClustering(N, affinity='precomputed').fit_predict(y)

# classify = [[] for _ in range(N)]
# for i,j in zip(label_class, label):
#   classify[i].append(j)

# [len(i) for i in classify]
# # classify
# # %%

# from sklearn.manifold import MDS
# embedding = MDS(n_components=100, dissimilarity='precomputed')
# x_transformed = embedding.fit_transform(y)

# # %%
# x_transformed.shape


# # %%
# kmeans = KMeans(n_clusters=10, random_state=5, n_jobs=-1).fit(x_transformed)
# kmeans.labels_
# # kmeans = SpectralClustering(20).fit(x_transformed)
# # np.count_nonzero(kmeans.labels_ == 3)


