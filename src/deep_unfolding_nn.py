# Implement the deep-unfolding NN architecture
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Any
from torch.utils.data import DataLoader, TensorDataset

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
            X_U_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            Y_U_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            Z_U_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.n_rx[i]))
            O_U_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_rx[i], self.setup.d[i]))
            X_W_dict[str(i)] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            Y_W_dict[str(i)] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            Z_W_dict[str(i)] = nn.Parameter(torch.randn(self.setup.d[i], self.setup.d[i]))
            X_V_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            Y_V_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            Z_V_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.n_tx))
            O_V_dict[str(i)] = nn.Parameter(torch.randn(self.setup.n_tx, self.setup.d[i]))
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

        def proj_power(V):
            for i in range(num_samples):
                s = 0
                for j in range(self.setup.K):
                    s += torch.trace(V[str(i)][str(j)] @ V[str(i)][str(j)].conj().T)
                for j in range(self.setup.K):
                    V[str(i)][str(j)] = torch.sqrt(self.setup.P/s) * V[str(i)][str(j)]
            return V

        num_samples = len(H)

        # Calculate A
        A = {}
        for i in range(num_samples):
            A[str(i)] = {}
            for j in range(self.setup.K):
                s = 0
                for k in range(self.setup.K):
                    s += torch.trace(V[str(i)][str(k)].conj().T @ V[str(i)][str(k)])
                ey = (1/self.setup.P) * s * torch.eye(self.setup.n_rx[j], dtype=torch.cfloat)
                s = 0
                for k in range(self.setup.K):
                    s += V[str(i)][str(k)] @ V[str(i)][str(k)].conj().T
                A[str(i)][str(j)] = ey + H.iloc[i, j] @ s @ H.iloc[i, j].conj().T

        # Calculate U
        U = {}
        for i in range(num_samples):
            U[str(i)] = {}
            for j in range(self.setup.K):
                A_inv = plus(A[str(i)][str(j)]) @ self.X_U[str(j)].to(torch.cfloat) + A[str(i)][str(j)] @ self.Y_U[str(j)].to(torch.cfloat) + self.Z_U[str(j)].to(torch.cfloat)
                U[str(i)][str(j)] = A_inv @ H.iloc[i, j] @ V[str(i)][str(j)] + self.O_U[str(j)].to(torch.cfloat)

        # Calclate E
        E = {}
        for i in range(num_samples):
            E[str(i)] = {}
            for j in range(self.setup.K):
                E[str(i)][str(j)] = torch.eye(self.setup.d[j], dtype=torch.cfloat) - U[str(i)][str(j)].conj().T @ H.iloc[i, j] @ V[str(i)][str(j)]
        
        # Calculate W
        W = {}
        for i in range(num_samples):
            W[str(i)] = {}
            for j in range(self.setup.K):
                W[str(i)][str(j)] = plus(E[str(i)][str(j)]) @ self.X_W[str(j)].to(torch.cfloat) + E[str(i)][str(j)] @ self.Y_W[str(j)].to(torch.cfloat) + self.Z_W[str(j)].to(torch.cfloat)

        # Calculate B
        B = {}
        for i in range(num_samples):
            s = 0
            for k in range(self.setup.K):
                s += torch.trace(U[str(i)][str(k)] @ W[str(i)][str(k)] @ U[str(i)][str(k)].conj().T)
            ey = (1/self.setup.P) * s * torch.eye(self.setup.n_tx, dtype=torch.cfloat)
            s = 0
            for k in range(self.setup.K):
                s += H.iloc[i, k].conj().T @ U[str(i)][str(k)] @ W[str(i)][str(k)] @ U[str(i)][str(k)].conj().T @ H.iloc[i, k]
            B[str(i)] = ey + s
                

        # Calculate V
        V = {}
        for i in range(num_samples):
            V[str(i)] = {}
            for j in range(self.setup.K): 
                B_inv = plus(B[str(i)]) @ self.X_V[str(j)].to(torch.cfloat) + B[str(i)] @ self.Y_V[str(j)].to(torch.cfloat) + self.Z_V[str(j)].to(torch.cfloat)
                V[str(i)][str(j)] = B_inv @ H.iloc[i, j].conj().T @ U[str(i)][str(j)] @ W[str(i)][str(j)] + self.O_V[str(j)].to(torch.cfloat)

        # Project V
        V = proj_power(V)

        return V

# Define the deep unfolding NN
class DUNN(nn.Module):
    def __init__(self, num_layers, setup):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(setup)
            for _ in range(num_layers)
        ])

    def forward(self, V, H):
        for layer in self.layers:
            V = layer(V, H)
        return V

# Train the model
class Trainer:
    def __init__(self, model: DUNN, setup, lr: float = 1e-3):
        self.setup = setup
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, H_df, loss_fn, num_epochs, batch_size):

        # Function for initializing V
        def init_V(H_df, setup):
            num_samples = H_df.shape[0]
            K = setup.K
            V_dict = {}

            for sample_idx in range(num_samples):
                V_sample = {}
                H_sample = [H_df.iloc[sample_idx, k] for k in range(K)]

                for k in range(K):
                    # Create interference channel matrix for all users ≠ k
                    H_interference = torch.cat([H_sample[j] for j in range(K) if j != k], dim=0)

                    # Compute null space of interference channel using SVD
                    _, S, Vh = torch.linalg.svd(H_interference)
                    rank = (S > 1e-6).sum().item()
                    null_space = Vh[rank:].conj().T  # shape: [n_tx, nullity]

                    # Choose as many columns as the number of streams we want (≤ nullity)
                    d_k = min(setup.n_rx[k], null_space.shape[1])
                    V_k = null_space[:, :d_k]

                    V_sample[str(k)] = V_k

                V_dict[str(sample_idx)] = V_sample
            return V_dict

        def proj_power(V):
            for i in range(len(V)):
                s = 0
                for j in range(self.setup.K):
                    s += torch.trace(V[str(i)][str(j)] @ V[str(i)][str(j)].conj().T)
                for j in range(self.setup.K):
                    V[str(i)][str(j)] = torch.sqrt(self.setup.P/s) * V[str(i)][str(j)]
            return V

        def shuffle_and_batch(df, batch_size):
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            return [df_shuffled[i:i + batch_size] for i in range(0, len(df_shuffled), batch_size)]

        loader = shuffle_and_batch(H_df, batch_size)

        for _ in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for H_batch in loader:
                # H_batch = H_batch.to(torch.complex64)
                self.opt.zero_grad()
                # Initialize V
                V0 = init_V(H_batch, self.setup)
                V0 = proj_power(V0)
                V_pred = self.model(V0, H_batch)
                loss = loss_fn(H_batch, V_pred, self.setup.P)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
        return total_loss / len(loader)