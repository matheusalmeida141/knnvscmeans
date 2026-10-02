# %%
import numpy as np
import pandas as pd
from lib.fuzzy import FCMImputer

df = pd.read_csv("./data/train1.csv")
X = pd.read_csv("data/original1.csv")

df_train = df[["x_0", "x_1"]].copy()
idx_missing = df_train["x_1"].isna()

# Limpa o arquivo de saída antes de iniciar
with open("fuzzy_results.csv", mode="w") as fl:
  fl.write("iter,rmse,method\n")

for iter_max in range(1, 101, 2):  # Passo menor para ver a convergência
  imp = FCMImputer(n_clusters=3, m=2.0, max_iter=iter_max, random_state=42)
  imp_fit = imp.fit_transform(df_train)

  df_imputado = pd.DataFrame({"x_0": imp_fit[:, 0], "x_1": imp_fit[:, 1]})

  # RMSE apenas nas posições faltantes
  rmse = np.sqrt(
      np.mean((X.loc[idx_missing, "x_1"] - df_imputado.loc[idx_missing, "x_1"]) ** 2)
  )

  with open("fuzzy_results.csv", mode="a") as fl:
    fl.write(f"{iter_max},{rmse},fuzzy\n")