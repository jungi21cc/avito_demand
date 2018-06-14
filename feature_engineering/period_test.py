import re
import gc
from datetime import datetime

import numpy as np
import pandas as pd

df_test = pd.read_csv("../test.csv")
periods_test = pd.read_csv('../periods_test.csv')
periods_test = periods_test.drop(['activation_date'], axis=1)

gc.collect()

merge_periods_test = pd.merge(df_test, periods_test, how="left", on='item_id')
merge_periods_test.to_csv("../merged_test.csv")

