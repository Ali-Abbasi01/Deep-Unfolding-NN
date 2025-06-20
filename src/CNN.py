# The CNN model
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
import importlib

import src.utils
importlib.reload(src.utils)
from src.utils import sum_rate_loss_BC

# Define the CNN model
class ChannelCNN(nn.Module):
    def __init__(self, setup):
        super(ChannelCNN, self).__init__()
        self.K = setup.K
        self.n_rx = setup.n_rx
        self.n_tx = setup.n_tx
        self.d = setup.d
        self.conv1 = nn.Conv2d(in_channels=2*self.K, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * self.n_rx * self.n_tx, 128)
        self.fc2 = nn.Linear(128, 2*self.K*self.d*self.n_tx)  # Output size matches the beamforming matrix

    def df_to_tensor(self, x_df):
        # Coverts dataframe to tensor, also real and imaginary parts of tensor entries are seperated.
        batch_size, K = x_df.shape
        N_r, N_t = x_df.iloc[0, 0].shape
        real_parts = torch.empty((batch_size, K, N_r, N_t))
        imag_parts = torch.empty((batch_size, K, N_r, N_t))

        for i in range(batch_size):
            for k in range(K):
                H = x_df.iloc[i, k]  # complex matrix
                real_parts[i, k] = H.real
                imag_parts[i, k] = H.imag

        # Concatenate along channel dimension → shape: (batch_size, 2K, N_r, N_t)
        x_tensor = torch.cat([real_parts, imag_parts], dim=1)
        return x_tensor

    def df_to_dict(self, df: pd.DataFrame) -> dict:
        # Converts dataframe to dictionary
        result = {}
        for row_idx in range(len(df)):
            row_dict = {}
            for col_idx in range(len(df.columns)):
                row_dict[str(col_idx)] = df.iloc[row_idx, col_idx].to(torch.cdouble)
            result[str(row_idx)] = row_dict
        return result

    def tensor_to_dict(self, tensor4d: torch.Tensor) -> dict:
        # Converts tensor to dictionary
        nested_dict = {}
        for i in range(tensor4d.size(0)):
            inner_dict = {}
            for j in range(tensor4d.size(1)):
                inner_dict[str(j)] = tensor4d[i, j].to(torch.cdouble)
            nested_dict[str(i)] = inner_dict
        return nested_dict

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 2*self.K, self.n_tx, self.d)  # Reshape to match the beamforming matrix
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

# Defien the training procedure
class Trainer():
    def __init__(self, setup, model):
        self.setup = setup
        self.model = model
    
    # Supervised training
    def train_supervised(self, dataset, num_epochs, batch_size, lr=0.001):
        criterion = nn.MSELoss()  # Using Mean Squared Error loss for regression
        optimizer = optim.Adam(self.model.parameters(), lr)

        def split_df_batches(df: pd.DataFrame, d: int, batch_size: int):
            # splits dataframe into batches. Returns a list of tuples, the first entry of which are the channels and the second entry are target beamforming matrices.
            result = []
            num_rows = len(df)
            for i in range(0, num_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                first_part = batch.iloc[:, :d]
                second_part = batch.iloc[:, d:]
                result.append((first_part.reset_index(drop=True), second_part.reset_index(drop=True)))          
            return result
        
        loader = split_df_batches(df=dataset, d=self.setup.K, batch_size=batch_size)

        for epoch in range(num_epochs):
            self.model.train()
            for inputs, targets in loader:
                inputs_tensor = self.model.df_to_tensor(inputs)
                targets_tensor = self.model.df_to_tensor(targets)
                optimizer.zero_grad()
                outputs = self.model(inputs_tensor)
                loss = criterion(outputs, targets_tensor)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Unsupervised training
    def train_unsupervised(self, dataset, num_epochs, batch_size, lr=0.001, penalty_coef=10):

        def srate_penalty_obj(H, V, PT, penalty_coef):
            # Calculates the loss which includes average sum rate and average penalty.
            K = H.shape[1]
            num_samples = H.shape[0]
            H = H.applymap(lambda x: x.to(torch.cdouble))
            s_rate_avg = sum_rate_loss_BC(H, V, PT)
            pens = []
            for b in range(num_samples):
                s_trace = 0
                for k in range(K):
                    s_trace += torch.trace(V[str(b)][str(k)] @ V[str(b)][str(k)].conj().T).real
                # print(s_trace)
                # pen = penalty_coef * ((s_trace - PT).clamp(min=0) ** 2).mean()
                pen = penalty_coef * ((s_trace - PT).clamp(min=0) ** 2)
                # print(pen)
                pens.append(pen)
            loss = (-1)*s_rate_avg + sum(pens)/len(pens)
            return loss
        
        def split_df_batches(df: pd.DataFrame, d: int, batch_size: int):
            result = []
            num_rows = len(df)
            for i in range(0, num_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                first_part = batch.iloc[:, :d]
                second_part = batch.iloc[:, d:]
                result.append((first_part.reset_index(drop=True), second_part.reset_index(drop=True)))          
            return result

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        loader = split_df_batches(df=dataset, d=self.setup.K, batch_size=batch_size)

        for epoch in range(num_epochs):
            self.model.train()
            for inputs, targets in loader:
                # targets = self.model.df_to_dict(targets)
                optimizer.zero_grad()
                inputs_tensor = self.model.df_to_tensor(inputs)
                outputs = self.model(inputs_tensor)
                outputs_dict = self.model.tensor_to_dict(outputs)
                loss = srate_penalty_obj(inputs, outputs_dict, self.setup.PT, penalty_coef)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        print("Training complete!")