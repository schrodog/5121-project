# %%
from apyori import apriori
import numpy as np

transaction = np.array([
  [1,3,4],
  [2,3,5],
  [1,2,3,5],
  [2,5]
])

results = list(apriori(transaction, min_support=0.5, min_confidence=0.5))
results



