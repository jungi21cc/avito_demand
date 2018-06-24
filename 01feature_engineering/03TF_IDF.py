import numpy as np
import pandas as pd
from scipy import sparse

train = pd.read_csv('../train1.csv')
test = pd.read_csv('../test1.csv')



sparse.save_npz("X.npz", X)
X = sparse.load_npz("X.npz")


sparse.save_npz("testing.npz", testing)
testing = sparse.load_npz("testing.npz")
