#%%
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("data/train1.csv")
X = pd.read_csv("data/original1.csv")
X_train = df[["x_0", "x_1"]]
index = X_train[~(X_train["x_1"] > -120)].index

#%%
error = 0
for i in range(1, 1000):
    imputer = KNNImputer(n_neighbors=i)
    result = imputer.fit_transform(X_train)
    error = (np.sqrt(np.sum(np.pow(X["x_1"] - result[:,1], 2))/X.shape[0]))
    print(error)
    with open("knn.csv", mode="a") as fl:
        fl.write(f"{i},{error},knn\n")

# %%

sns.scatterplot(X, x="x_0", y="x_1",legend="auto")
sns.scatterplot(df[X_train.index.isin(index)], x="x_0", y="KNN", legend="auto")
plt.legend(["Original", "KNN Input"])
plt.title("KNN Imputer ")
# %%
np.sqrt(np.sum(np.pow(X["x_1"] - df["KNN"], 2))/X.shape[0])
# %%


#     imputer = KNNImputer(n_neighbors=2)
#     result = imputer.fit_transform(X_train)

#     X_aux= pd.DataFrame({
#         "x_2": result[:,0],
#         "KNN": result[:,1]
        
#     })
#     # X_train = X.rename(columns={0:"x_0", 1:"x_1"})
#     df = pd.concat([X_train,X_aux],axis=1)
#     del X_aux
#     df.drop(columns=["x_2"])
# %%
