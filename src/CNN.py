import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
from src.utils import calculate_sum_rate

# Define the CNN model
class ChannelCNN(nn.Module):
    def __init__(self, K, Nr, Nt):
        super(ChannelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*K, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * Nr * Nt, 128)
        self.fc2 = nn.Linear(128, 2*K*Nr*Nt)  # Output size matches the beamforming matrix

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 2*K, Nr, Nt)  # Reshape to match the target beamforming matrix
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def train_supervised(self):
        criterion = nn.MSELoss()  # Using Mean Squared Error loss for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        print("Training complete!")

    def train_unsupervised(self):
        def srate_penalty_obj(H, V, alpha, P_max, penalty_coef):

            # Bring H back to the dictionary form
            K = int(H.shape[1]/2)
            B, twoK, Nr, Nt = X.shape
            assert twoK == 2*K, f"Expected 2*K={2*K} channels, got {twoK}"
            
            channel_dicts = []
            for i in range(B):
                sample_dict = {}
                for k in range(K):
                    real = H[i, 2*k    , :, :]   # [Nr, Nt]
                    imag = H[i, 2*k + 1, :, :]   # [Nr, Nt]
                    Hk   = torch.complex(real, imag)
                    sample_dict[k] = Hk
                channel_dicts.append(sample_dict)
            H = channel_dicts

            # Bring V back to the dictionary form
            K = int(V.shape[1]/2)
            B, twoK, Nr, Nt = V.shape
            assert twoK == 2*K, f"Expected 2*K={2*K} channels, got {twoK}"
            
            V_dicts = []
            for i in range(B):
                sample_dict = {}
                for k in range(K):
                    real = V[i, 2*k    , :, :]   # [Nr, Nt]
                    imag = V[i, 2*k + 1, :, :]   # [Nr, Nt]
                    Vk   = torch.complex(real, imag)
                    sample_dict[k] = Vk
                V_dicts.append(sample_dict)
            V = V_dicts

            # Calculate the sum rate
            s_rate = []
            for b in range(B):
                s_rate.append(calculate_sum_rate(H[b], V[b], alpha, sig))
            
            # Calculate penalties
            pens = []
            for b in range(B):
                s_trace = 0
                for k in range(K):
                    s_trace += torch.linalg.trace(V[b][k] @ V[b][k].conj().T)
                pen = penalty_coeff * ((s_trace - P_max).clamp(min=0) ** 2).mean()
                pens.append(pen)

            #total objective
            loss = ((-1)*s_rate + pens).mean()
            return loss

        criterion = srate_penalty_obj()  # Using Mean Squared Error loss for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        print("Training complete!")