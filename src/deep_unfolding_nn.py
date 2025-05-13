import torch
import torch.nn as nn

# Define the setup
class setup():
    def __init__(self, n_tx, n_rx, num_streams, num_users, P):
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.d = num_streams
        self.K = num_users
        self.P = P


# Define the layer
class Layer(nn.Module):
    def __init__(self, setup):
        super().__init__()
        self.setup = setup
        X_U_dict = {}
        Y_U_dict = {}
        Z_U_dict = {}
        O_U_dict = {}
        X_W_dict = {}
        Y_W_dict = {}
        Z_W_dict = {}
        X_V_dict = {}
        Y_V_dict = {}
        Z_V_dict = {}
        O_V_dict = {}
        for i in range(self.setup.K):
            X_U_dict[i] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            Y_U_dict[i] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            Z_U_dict[i] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            O_U_dict[i] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.d[i]))
            X_W_dict[i] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            Y_W_dict[i] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            Z_W_dict[i] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            X_V_dict[i] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            Y_V_dict[i] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            Z_V_dict[i] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            O_V_dict[i] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.d[i]))
        self.X_U = nn.ParameterDict(X_U_dict)
        self.Y_U = nn.ParameterDict(Y_U_dict)
        self.Z_U = nn.ParameterDict(Z_U_dict)
        self.O_U = nn.ParameterDict(O_U_dict)
        self.X_W = nn.ParameterDict(X_W_dict)
        self.Y_W = nn.ParameterDict(Y_W_dict)
        self.Z_W = nn.ParameterDict(Z_W_dict)
        self.X_V = nn.ParameterDict(X_V_dict)
        self.Y_V = nn.ParameterDict(Y_V_dict)
        self.Z_V = nn.ParameterDict(Z_V_dict)
        self.O_V = nn.ParameterDict(O_V_dict)

    def forward(self, V, H):

        # A^+ operator
        def plus(A):
            diag_inv = 1.0 / A.diagonal(dim1=-2, dim2=-1)
            A_plus = torch.zeros_like(A)
            A_plus.diagonal(dim1=-2, dim2=-1).copy_(diag_inv)
            return A_plus

        def proj():
            return

        num_samples = len[H]

        # Calculate A
        A = {}
        for i in range(num_samples):
            A[i] = {}
            for j in range(self.setup.K):
                s = 0
                for k in range(self.setup.K):
                    s += torch.tensor(V[i][k].conj().T @ V[i][k])
                ey = (1/self.setup.P) * s * torch.eye(self.setup.n_rx[j], dtype=torch.cfloat)
                s = 0
                for k in range(self.setup.K):
                    s += V[i][k] @ V[i][k].conj().T
                A[i][j] = eye + H.iloc[i, j] @ s @ H.iloc[i, j].conj().T

        # Calculate U
        U = {}
        for i in range(num_samples):
            U[i] = {}
            for j in range(self.setup.K):
                A_inv = plus(A[i][j]) @ self.X_U[j] + A[i][j] @ self.Y_U[j] + self.Z_U[j]
                U[i][j] = A_inv @ H.iloc[i, j] @ V[i][j] + self.O_U[j]

        # Calclate E
        E = {}
        for i in range(num_samples):
            E[i] = {}
            for j in range(self.setup.K):
                E[i][j] = torch.eye(self.setup.d[j], dtype=torch.cfloat) - U[i][j].conj().T @ H[i][j] @ V[i][j]
        
        # Calculate W
        W = {}
        for i in range(num_samples):
            W[i] = {}
            for j in range(self.setup.K):
                W[i][j] = plus(E[i][j]) @ self.X_W[j] + E[i][j] @ self.Y_W[j] + self.Z_W[j]

        # Calculate B
        B = {}
        for i in range(num_samples):
            B[i] = {}
            for j in range(self.setup.K):
                s = 0
                for k in range(self.setup.K):
                    s += torch.trace(U[i][k] @ W[i][k] @ U[i][k].conj().T)
                ey = (1/self.setup.P) * s * torch.eye(self.setup.n_tx, dtype=torch.cfloat)
                s = 0
                for k in range(self.setup.K):
                    s += H.iloc[i, k].conj().T @ U[i][k] @ W[i][k] @ U[i][k].conj().T @ H.iloc[i, k]

        # Calculate V
        V = {}
        for i in range(num_samples):
            V[i] = {}
            for j in range(self.setup.K): 
                B_inv = plus(B[i][j]) @ self.X_V[j] + B[i][j] @ self.Y_V[j] + self.Z_V[j]
                V[i][j] = B_inv @ H.iloc[i, j] @ U[i][j] @ W[i][j] + self.O_V[j]