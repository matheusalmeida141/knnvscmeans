# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
# %%

centers = [[1, 1], [-1, -1], [1, -1]]
X, Y= make_blobs(
    n_samples=100000, centers=centers, cluster_std=0.24, random_state=21
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X)
X= X.rename(columns={0:"x_0", 1:"x_1"})
X
# %%
plt.figure(dpi=400)
sns.scatterplot(X, x="x_0", y="x_1")
plt.grid(True)
plt.title("Dados Sinteticos")

# %%
X_train = X.copy()
X_train.isna().sum().sort_values(ascending = False)

#%%
#Construindo missing

quantidade = int(X_train.shape[0] * 10/100)
index = X_train.sample(n = quantidade,random_state=32).index

X_train.loc[index, "x_1"] = np.nan

X_train

# %%

X_train.isna().sum()
#%%
X_train["median"] = X_train["x_1"]
X_train["media"]  = X_train["x_1"]
X_train["max"]    = X_train["x_1"]
X_train["min"]    = X_train["x_1"]


# %%
#       inputacao por max, min, media e median
X_train["median"] = X_train["median"].fillna(np.median(X_train["x_1"].dropna()))
X_train["media"]  = X_train["media"].fillna( np.mean(  X_train["x_1"]))
X_train["max"]    = X_train["max"].fillna(   np.max(   X_train["x_1"]))
X_train["min"]    = X_train["min"].fillna(   np.min(   X_train["x_1"]))
X_train.isna().sum().sort_values(ascending = False)

# %%
plt.figure(figsize=(8,5), dpi=400)

plt.scatter(
    X_train["x_0"],
    X_train["x_1"],
    color = "blue",
    label="Original",
    s=30,
    alpha=0.2
)

plt.scatter(
    X_train[X_train.index.isin(index)]["x_0"],
    X_train[X_train.index.isin(index)]["median"],
    color = "orange",
    label="Mediana",
    s=30,
    alpha=1

)

plt.scatter(
    X_train[X_train.index.isin(index)]["x_0"],
    X_train[X_train.index.isin(index)]["media"],
    color = "red",
    label="Media",
    s=30,
    alpha=1

)
plt.scatter(
    X_train[X_train.index.isin(index)]["x_0"],
    X_train[X_train.index.isin(index)]["max"],
    color = "green",
    label="Max",
    s=30,
    alpha=1
)

plt.scatter(
    X_train[X_train.index.isin(index)]["x_0"],
    X_train[X_train.index.isin(index)]["min"],
    color = "green",
    label="Min",
    s=30,
    alpha=1
)

plt.grid()
plt.title("Inputaçoes vs Original")
plt.legend()
plt.show()

# %%
X_train.to_csv("data/train1.csv", index=False)
X.to_csv("data/original1.csv", index=False)
# %%
