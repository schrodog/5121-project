# %%
from apyori import apriori
import numpy as np

# %%

transactions = np.array([
  [1,2],
  [1,3],
  [2,3,4],
  [1,2,3,4]
])

result = list(apriori(transactions, min_support=0.4, min_confidence=0.1, min_length=2)) 
result













