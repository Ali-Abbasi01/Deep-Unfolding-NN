import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A, B: (..., M, K) and (..., K, N) complex
    return torch.matmul(A, B)


def hermitian(X: torch.Tensor) -> torch.Tensor:
    # Conjugate transpose
    return X.mH


def sum_rate_objective(H: torch.Tensor,
                       V: torch.Tensor,
                       weights: torch.Tensor = None,
                       P_max: float = None,
                       penalty_coeff: float = 1.0) -> torch.Tensor:
    """
    H: [B, K, Nr, Nt] complex
    V: [B, K, Nt, Nr] complex
    weights: [B, K] real optional
    P_max: float, total power limit
    penalty_coeff: float
    returns: [scalar], the negative sum-rate + penalty
    """
    B, K, Nr, Nt = H.shape
    device = H.device

    # Compute total transmit covariance and power
    VV = torch.zeros((B, Nt, Nt), dtype=torch.complex64, device=device)
    trace_V = torch.zeros((B,), dtype=torch.float32, device=device)
    for k in range(K):
        Vk = V[:, k]  # [B, Nt, Nr]
        VkVk = complex_matmul(Vk, hermitian(Vk))  # [B, Nt, Nt]
        VV += VkVk
        trace_V += VkVk.real.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

    rate = torch.zeros((B,), dtype=torch.float32, device=device)
    I_Nr = torch.eye(Nr, dtype=torch.complex64, device=device).unsqueeze(0)

    for k in range(K):
        Hk = H[:, k]  # [B, Nr, Nt]
        Vk = V[:, k]  # [B, Nt, Nr]
        # Interference-plus-noise
        JK = trace_V.view(B, 1, 1) * I_Nr + complex_matmul(Hk, complex_matmul(VV - complex_matmul(Vk, hermitian(Vk)), hermitian(Hk)))
        # Desired signal
        Sig = complex_matmul(Hk, Vk)
        SigCov = complex_matmul(Sig, hermitian(Sig))  # [B, Nr, Nr]
        # Rate per user: log det(I + SigCov JK^{-1})
        JK_inv = torch.linalg.inv(JK)
        arg = I_Nr + complex_matmul(SigCov, JK_inv)
        det = torch.linalg.det(arg).real.clamp(min=1e-12)
        rate_k = torch.log(det)
        if weights is not None:
            rate_k = rate_k * weights[:, k]
        rate += rate_k

    # Loss = -sum_rate + penalty
    loss = -rate.mean()
    if P_max is not None:
        penalty = penalty_coeff * ((trace_V - P_max).clamp(min=0) ** 2).mean()
        loss = loss + penalty
    return loss


class CNN(nn.Module):
    def __init__(self, Nr: int, Nt: int, K: int):
        super().__init__()
        self.Nr, self.Nt, self.K = Nr, Nt, K
        # CNN for channel matrix
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.act3 = nn.LeakyReLU()

        flat_size = 32 * (Nr * K) * (Nr * K)
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc2 = nn.Linear(1024, 2 * K * Nt * Nr)  # real+imag

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, 2, Nr*K, Nr*K]
        B = X.shape[0]
        # Combine real+imag as channels
        Xc = X[:, 0:1]  # use real part only or combine? we can stack
        # For simplicity, process magnitude or real
        x = self.conv1(Xc)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Output beamforming V
        V_real = x[:, :self.K * self.Nt * self.Nr]
        V_imag = x[:, self.K * self.Nt * self.Nr:]
        V_real = V_real.view(B, self.K, self.Nt, self.Nr)
        V_imag = V_imag.view(B, self.K, self.Nt, self.Nr)
        V = torch.complex(V_real, V_imag)
        return V


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 P_max: float = None,
                 penalty_coeff: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.P_max = P_max
        self.penalty_coeff = penalty_coeff

    def train_supervised(self,
                         dataloader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         epochs: int = 10):
        self.model.train()
        mse = nn.MSELoss()
        for epoch in range(epochs):
            total_loss = 0
            for X, V_true in dataloader:
                X, V_true = X.to(self.device), V_true.to(self.device)
                optimizer.zero_grad()
                V_pred = self.model(X)
                loss = mse(V_pred.real, V_true.real) + mse(V_pred.imag, V_true.imag)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Supervised Loss: {total_loss/len(dataloader):.4f}")

    def train_unsupervised(self,
                           dataloader: torch.utils.data.DataLoader,
                           optimizer: torch.optim.Optimizer,
                           epochs: int = 10,
                           weights: torch.Tensor = None):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X, H in dataloader:
                X, H = X.to(self.device), H.to(self.device)
                optimizer.zero_grad()
                V_pred = self.model(X)
                loss = sum_rate_objective(H, V_pred, weights, self.P_max, self.penalty_coeff)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Unsupervised Loss: {total_loss/len(dataloader):.4f}")
