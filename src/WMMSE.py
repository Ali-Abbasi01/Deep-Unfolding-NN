# The WMMSE algorithm

import torch

K = 3  # number of cells
I_k = [2, 3, 1]  # number of users in each cell
n_tx = [4, 2, 3]  # number of antennas at each transmitter
n_rx = [[2, 3], [1, 2, 2], [4]]  # number of antennas at each user in each cell
P_k = [5, 6, 7]
sig_i_k = [[.2, .3], [.1, .2, .2], [.4]]
d = [[5, 5], [5, 5, 5], [5]]
alpha = [[1, 1], [1, 1, 1], [1]]

# Initialize channel dictionary
H = {}
for k in range(K):  # transmitter cell index
    H[k] = {}
    for l in range(K):  # receiver cell index
        for i in range(I_k[l]):  # user index in cell l
            tx_ant = n_tx[k]
            rx_ant = n_rx[l][i]
            # Channel from transmitter k to user (l, i)
            H[k][(l, i)] = torch.randn(rx_ant, tx_ant, dtype=torch.cfloat)

class WMMSE_alg():
    def __init__(self, K, I_k, n_tx, n_rx, H, P_k, sig_i_k, d, alpha):
        self.K = K
        self.I_k = I_k
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.H = H
        self.P_k = P_k
        self.sig_i_k = sig_i_k
        self.d = d
        self.alpha = alpha

    def algorithm(self):
        def update_U(V):
            U = {}
            for k in range(self.K):
                U[k] = {}
                for i in range(self.I_k[k]):
                    A = 0
                    for j in range(self.K):
                        a = 0
                        for l in range(self.I_k[j]):
                            a += self.H[j][(k, i)] @ V[j][l] @ V[j][l].conj().T @ self.H[j][(k, i)].conj().T
                        A += a
                    A = A + self.sig_i_k[k][i] @ torch.eye(self.n_rx[k][i])
                    U[k][i] = torch.linalg.inv(A) @ self.H[k][(k, i)] @ V[k][i]


        def update_W(U, V):
            W = {}
            for k in range(self.K):
                W[k] = {}
                for i in range(self.I_k[k]):
                    E = torch.eye(self.d[k][i]) - U[k][i].conj().T @ self.H[k][(i, k)] @ V[k][i]
                    W[k][i] = torch.linalg.inv(E)


        def update_V(U, W):
            V = {}
            for k in range(self.K):
                V[k] = {}
                for i in range(self.I_k[k]):
                    B = 0
                    for j in range(self.K):
                        b = 0
                        for l in range(self.I_k[j]):
                            b += self.alpha[j][l] * self.H[k][(j, l)] @ U[j][l] @ W[j][l] @ U[j][l].conj().T @ self.H[k][(j, l)]
                        B += b
                    B = B + mu_star[k] @ torch.eye(self.n_rx[k][i])
                    V[k][i] = self.alpha[k][i] * torch.linalg.inv(B) @ self.H[k][(k, i)].conj().T @ U[k][i] @ W[k][i]
        

        def calc_mu(U, W):
            def lhs(mu):
                return torch.sum(phi_diag / (lambda_diag + mu)**2)

            # Initialize bounds
            mu_low = torch.tensor(0.0, dtype=phi_diag.dtype, device=phi_diag.device)
            mu_high = torch.tensor(1.0, dtype=phi_diag.dtype, device=phi_diag.device)

            # Expand upper bound until lhs(mu_high) <= Pk
            while lhs(mu_high) > Pk:
                mu_high *= 2

            # Bisection search
            for _ in range(max_iter):
                mu_mid = (mu_low + mu_high) / 2
                val = lhs(mu_mid)

                if torch.abs(val - Pk) < tol:
                    break
                elif val > Pk:
                    mu_low = mu_mid
                else:
                    mu_high = mu_mid

            return mu_mid


        # Initialize V

        # The algorithm
        for :
        

        return V


