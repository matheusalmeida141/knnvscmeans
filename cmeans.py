# %%
from lib.fuzzy import FCMImputer
import pandas as pd
import numpy as np
df = pd.read_csv("./data/train1.csv")
df.head()


#%%
df_train = df[["x_0", "x_1"]].copy()
X = pd.read_csv("data/original1.csv")
index = df_train[df_train["x_1"].isna()].index
index

#%%
error = 0
for iter in range(1, 1001, 10):
    imp = FCMImputer(n_clusters=3, m=2.0, max_iter=iter, tol=1e-1000)
    imp_fit = imp.fit_transform(df_train)
    df_imputado = pd.DataFrame({
        "x_0": imp_fit[:,0],
        "x_1": imp_fit[:,1] 
    })
    error = (np.sqrt(np.sum(np.pow(X["x_1"] - df_imputado["x_1"], 2))/X.shape[0]))
    print(iter,error)
    with open("knn.csv", mode="a") as fl:
        fl.write(f"{iter},{error},fuzzy\n")
#%%
# df_original = pd.read_csv("./data/original1.csv")
# df_original["x_1"] - df_imputado["x_1"]
#%%
# import matplotlib.pyplot as plt

# plt.figure(dpi=400)
# # Plota os dados originais
# plt.scatter(df_original.loc[index,"x_0"], df_original.loc[index,"x_1"], color="red", label="Original")
# plt.title(f"Original e C-Means iteração: {inte}")
# # Plota os dados imputados
# plt.scatter(
#     df_imputado.loc[index, "x_0"],
#     df_imputado.loc[index, "x_1"],
#     color="blue",
#     label="Imputado",
# )

# # Adiciona os rótulos dos eixos e exibe a legenda
# plt.xlabel("x_0")
# plt.ylabel("x_1")
# plt.legend()

# plt.show()