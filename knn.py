#%%
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
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
#%%

#%%
# # 
# # %%
# result = pd.DataFrame({"x_0": result[:,0],
#                         "x_1":result[:,1]})
# # %%
# plt.figure(dpi=400)
# # Plota os dados originais
# plt.scatter(x =  X.loc[index,"x_0"], y=X.loc[index,"x_1"], color="red", label="Original")
# # Plota os dados imputados
# plt.scatter(
#     result.loc[index, "x_0"],
#     result.loc[index, "x_1"],
#     color="blue",
#     label="Imputado",
# )
# # %%
# # np.sqrt(np.sum(np.pow(X["x_1"] - df["KNN"], 2))/X.shape[0])
# # %%


# #     imputer = KNNImputer(n_neighbors=2)
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
