# %%
import numpy as np
import pandas as pd
import os

def preprocess():
  # %%
  data_file = os.getcwd()+'/data/anonymous-msweb.test'

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

  
  
  
  
  
  
  
  
