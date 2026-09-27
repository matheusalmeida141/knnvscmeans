import skfuzzy as fuzz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df = pd.read
#%%
alldata = X_train

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, 3, 2, error=0.005, maxiter=1000, init=None)