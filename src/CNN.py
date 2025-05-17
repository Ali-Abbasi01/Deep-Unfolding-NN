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

class Trainer():
    def __init__(self, setup, model):
        self.setup = setup
        self.model = model
    
    def train_supervised(self, H_df, num_epochs, batch_size, lr=0.001):
        criterion = nn.MSELoss()  # Using Mean Squared Error loss for regression
        optimizer = optim.Adam(self.model.parameters(), lr)
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')