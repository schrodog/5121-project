# %%
from apyori import apriori
import numpy as np
import dill

# %%

transactions = np.array([
  [1,2],
  [1,3],
  [2,3,4],
  [1,2,3,4],
  [1,2,4]
])

result = np.array(list(apriori(transactions, min_support=3, min_confidence=0, min_length=2, percent=False)))
print(result)

# %%
dill.dump_session('temp_storage.db')



# %%
dill.load_session('temp_storage.db')
result






