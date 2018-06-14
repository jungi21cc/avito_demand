import re
import gc
from datetime import datetime

import numpy as np
import pandas as pd

df_train = pd.read_csv("../train.csv")
periods_train = pd.read_csv('../periods_train.csv')
periods_train = periods_train.drop(['activation_date'], axis=1)

gc.collect()

merge_periods_train = pd.merge(df_train, periods_train, how="left", on='item_id')
merge_periods_train.to_csv("../merged_train.csv")
