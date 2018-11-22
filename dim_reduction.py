# %%
visit_bool_data

# %%
from sklearn import manifold

print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=220, n_jobs=3).fit_transform(visit_bool_data)
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

# too time consuming to reduce dimensionality
linkage = 'ward'
clustering = AgglomerativeClustering(linkage=linkage, n_clusters=17)
t0 = time()
print('start')
clustering.fit(X_red)

print("%s :\t%.2fs" %(linkage, time()-t0))
plot_clustering(X_red, clustering.labels_, linkage+" linkage")

plt.show()

clustering.labels_ == 0




