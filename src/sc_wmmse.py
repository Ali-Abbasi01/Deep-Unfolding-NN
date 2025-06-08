# WMMSE algorithm for single cell

import torch

class WMMSE_alg_sc():
    def __init__(self, K, n_tx, n_rx, H, PT, sig_k, d, alpha, max_iter_alg, tol_alg):
        self.K = K
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.H = H
        self.PT = PT
        self.sig_k = sig_k
        self.d = d
        self.alpha = alpha
        self.max_iter_alg = max_iter_alg
        self.tol_alg = tol_alg

    def algorithm(self, V_init):
        def update_U(V):
            U = {}
            for k in range(self.K):
                term1 = (self.sig_k[k] / self.PT) * (sum([torch.trace(V[str(k)] @ V[str(k)].conj().T) for k in range(self.K)])) * torch.eye(self.n_rx[k], dtype=torch.cdouble)
                term2 = sum([(self.H[str(k)] @ V[str(l)] @ V[str(l)].conj().T @ self.H[str(k)].conj().T)for l in range(self.K)])
                A = term1 + term2
                U[str(k)] = torch.linalg.inv(A) @ self.H[str(k)] @ V[str(k)]
            return U
        
        def update_W(U, V):
            W = {}
            for k in range(self.K):
                E = torch.eye(self.d[k], dtype=torch.cdouble) - U[str(k)].conj().T @ self.H[str(k)] @ V[str(k)]
                W[str(k)] = torch.linalg.inv(E)
            return W

        def update_V(U, W):
            V = {}
            for k in range(self.K):
                term1 = (sum([((self.sig_k[k] / self.PT) * torch.trace(self.alpha[k] * U[str(k)] @ W[str(k)] @ U[str(k)].conj().T)) for k in range(self.K)])) * torch.eye(self.n_tx, dtype=torch.cdouble)
                term2 = sum([(self.alpha[k] * self.H[str(k)].conj().T @ U[str(k)] @ W[str(k)] @ U[str(k)].conj().T @ self.H[str(k)]) for k in range(self.K)])
                B = term1 + term2
                V[str(k)] = self.alpha[k] * torch.linalg.inv(B) @ self.H[str(k)].conj().T @ U[str(k)] @ W[str(k)]
            V_proj = proj_power(V)
            return V_proj
        
        def proj_power(V):
            alph = torch.sqrt(torch.tensor(self.PT)) / torch.sqrt(torch.tensor(sum([torch.trace(V[str(k)] @ V[str(k)].conj().T) for k in range(self.K)])))
            V_proj = {str(k): alph * V[str(k)] for k in range(self.K)}
            return V_proj
        
        # Keep record of V, U, and W
        V_l = []
        U_l = []
        W_l = []

        # The algorithm
        U = update_U(V_init)
        W = update_W(U, V_init)
        V = update_V(U, W)
        V_l.append(V)
        U_l.append(U)
        W_l.append(W)
        for _ in range(self.max_iter_alg):
            W_prm = W
            U = update_U(V)
            W = update_W(U, V)
            V = update_V(U, W)
            V_l.append(V)
            U_l.append(U)
            W_l.append(W)

            # Check for convergance
            val1 = 0
            for k in range(self.K):
                val1 += torch.log2(torch.linalg.det(W[str(k)]))

            val2 = 0
            for k in range(self.K):
                val2 += torch.log2(torch.linalg.det(W_prm[str(k)]))

            if torch.abs(val1 - val2) <= self.tol_alg:
                break
        
        return V_l, U_l, W_l