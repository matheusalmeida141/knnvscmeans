"""
Imputação de dados faltantes com Fuzzy C-Means (FCM).
Estratégia: Optimal Completion Strategy (Hathaway & Bezdek, 2001).
Dependência: apenas numpy.
"""
import numpy as np


class FCMImputer:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters      # c: número de clusters
        self.m = m                        # fuzzificador (> 1)
        self.max_iter = max_iter          # máximo de iterações
        self.tol = tol                    # critério de convergência
        self.random_state = random_state

    # ---------- passos internos do FCM ----------
    def _atualiza_centros(self, X, U):
        Um = U ** self.m                                   # (n, c)
        return (Um.T @ X) / Um.sum(axis=0)[:, None]        # (c, d)

    def _atualiza_pertinencias(self, X, V):
        # distâncias euclidianas de cada ponto a cada centro: (n, c)
        D = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)
        D = np.fmax(D, 1e-12)                              # evita divisão por zero
        expoente = 2.0 / (self.m - 1.0)
        # u_ik = 1 / soma_j (d_ik / d_ij)^expoente
        razao = (D[:, :, None] / D[:, None, :]) ** expoente
        return 1.0 / razao.sum(axis=2)

    # ---------- método principal ----------
    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        mask = np.isnan(X)                                 # True onde falta valor
        if not mask.any():
            return X

        # 1) normalização (média e desvio ignorando NaN)
        media = np.nanmean(X, axis=0)
        desvio = np.nanstd(X, axis=0)
        desvio[desvio == 0] = 1.0
        Xn = (X - media) / desvio

        # 2) preenchimento inicial: média da coluna (= 0 após normalizar)
        Xn[mask] = 0.0

        # 3) inicializa matriz de pertinência aleatória (linhas somam 1)
        rng = np.random.default_rng(self.random_state)
        U = rng.random((Xn.shape[0], self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)

        self.historico_ = []
        for it in range(self.max_iter):
            V = self._atualiza_centros(Xn, U)              # centros
            U_novo = self._atualiza_pertinencias(Xn, V)    # pertinências

            # 4) reimputa apenas as posições faltantes:
            #    x_ij = sum_k u_ik^m * v_kj / sum_k u_ik^m
            Um = U_novo ** self.m
            estimativa = (Um @ V) / Um.sum(axis=1, keepdims=True)
            Xn[mask] = estimativa[mask]

            # 5) convergência
            delta = np.abs(U_novo - U).max()
            self.historico_.append(delta)
            U = U_novo
            if delta < self.tol:
                break

        self.centros_ = V * desvio + media                 # volta à escala original
        self.pertinencias_ = U
        self.n_iter_ = it + 1
        return Xn * desvio + media                         # desnormaliza


