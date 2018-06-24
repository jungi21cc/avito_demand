import pickle
import pandas as pd
import numpy as np

pickle.dump(model, open("pima.pickle.dat", "wb"))
 
model = pickle.load(open("pima.pickle.dat", "rb"))

model.predict()