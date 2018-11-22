# %%
import pandas as pd
import numpy as np
from plotnine import *
from plotnine.data import mpg

# %%
mpg.head(15)

# %%
(mpg['manufacturer']
  .value_counts(sort=False)
  .plot.barh()
  .set_title('Number of Car by Make')
)

# %%
(ggplot(mpg)
  + aes(x='manufacturer')
  + geom_bar(size=20)
  + coord_flip()
  + ggtitle('num of cars')
)

# %%
(ggplot(mpg)
  + aes(x='cty')
  + geom_histogram(binwidth=4)
)









