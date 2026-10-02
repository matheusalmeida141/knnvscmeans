#%%
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import time

t1 = time.time()


# 1. Carregamento dos dados
df = pd.read_csv("data/train1.csv")
X_true = pd.read_csv("data/original1.csv")
X_train = df[["x_0", "x_1"]].copy()

# 2. Identificação precisa das posições faltantes (NaN) em x_1
idx_missing = X_train["x_1"].isna()

# 3. Definição do limite de K (não pode exceder o total de amostras válidas)
n_amostras_validas = X_train["x_1"].dropna().shape[0]
max_k = min(100, n_amostras_validas)  # Limite seguro para evitar estouro

# 4. Inicialização do arquivo CSV (limpa dados de execuções anteriores)
with open("knn.csv", mode="w") as fl:
    fl.write("k,rmse,method\n")

# 5. Loop de imputação e avaliação
with open("knn.csv", mode="a") as fl:
    for k in range(1, max_k):
        imputer = KNNImputer(n_neighbors=k)
        result = imputer.fit_transform(X_train)
        
        # RMSE calculado SOMENTE nas posições que eram originalmente NaN
        val_real = X_true.loc[idx_missing, "x_1"]
        val_imp = result[idx_missing, 1]
        rmse = np.sqrt(np.mean((val_real - val_imp) ** 2))
        
        fl.write(f"{k},{rmse:.6f},knn\n")
        print(f"K = {k:3d} | RMSE = {rmse:.6f}")

print("Tempo de execucao ", (time.time() - t1))