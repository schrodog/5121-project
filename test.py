# %%

import os
import numpy as np
import pandas as pd
from plotnine import *

# %%

data_file = os.getcwd()+'/anonymous-msweb.data'

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

gg = (ggplot(user_visit_df)
  + aes(x='count')
  + geom_histogram(binwidth=1)
  # + geom_col(aes(x=user_visit_df[:100].index, y="count"))
  + scale_x_continuous(breaks = range(0,20, 2))
  # + theme(axis_text_x=element_text(rotation=40, ha="right"))
)

print(gg)
gg.save('user_visit_count.pdf')

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

# %%
stat_df.plot(kind='bar')

# %%
stat_df['vroot'] = CategoricalDtype(stat_df['vroot'], categories=stat_df['visit'], ordered=True)

# %% show count of visits to same vroot
gg = (ggplot(stat_df)
  + aes(x='visit')
  + geom_histogram(binwidth=25)
  + scale_x_continuous(breaks = range(0, 12000, 500))
  + theme(axis_text_x=element_text(rotation=40, ha="right"))
)
print(gg)
gg.save('visit_count.pdf')

# %% show distribution of visits by user

gg = (ggplot(stat_df)
  + geom_col(aes(x="vroot", y="visit"))
  + scale_x_continuous(breaks = range(1000,1301, 50))
  + theme(axis_text_x=element_text(rotation=40, ha="right"))
)

print(gg)
gg.save('visit_vroot.pdf')

# %%
stat_df

# %%
# investigate users
# visit_pd[[10076, 10001, 10099, 10003, 10006, 10089, 10002, 10100]]
visit_pd[41033]

# %%
# attr_pd[attr_pd['id'].isin([1205,1001,1043,1004,1000,1001,1002,1003,1018])]
attr_pd[attr_pd['id'].isin([1008,1007,1037,1009,1032,1017,1034,1018])]


# %% filter out vroot with only 1 visitor
one_vroot = uniq_elem[counts == 1]
multiple_vroot = uniq_elem[counts > 5]
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
multiple_vroot, len(multiple_vroot)

# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
%matplotlib tk

# %%
part_df = visit_full_df[:100]
z = linkage(part_df, method='ward')
z

# %%
fig, ax = plt.subplots(1,1, figsize=(50,20))
ax.set_title("hierarchical clustering")
ax.set_xlabel("distance")
ax.set_ylabel("id")

dendrogram(z, labels=part_df.index, leaf_rotation=0, orientation='top', color_threshold=20, above_threshold_color='grey' )
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=15)
plt.yticks(size=20)
plt.savefig("dendro.png", format="png", dpi=80, bbox_inches='tight')

# %%
z = None

