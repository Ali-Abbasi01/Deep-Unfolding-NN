import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

############################################################
#  Complex helpers                                         #
############################################################


def complex_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex matrix multiplication that works for batched tensors.
    Both inputs: (..., M, K) @ (..., K, N) -> (..., M, N)"""
    return torch.matmul(a, b)


def hermitian(a: torch.Tensor) -> torch.Tensor:
    """Hermitian (conjugate transpose) for last two dims."""
    return a.transpose(-2, -1).conj()


def proj_power(v: torch.Tensor, p_total: float) -> torch.Tensor:
    """Project precoding matrix to satisfy total power constraint."""
    power = torch.sum(torch.real(v * v.conj()))
    scale = torch.sqrt(p_total / (power + 1e-12))
    return v * scale

############################################################
#  Building blocks                                          #
############################################################

class ParamBlock(nn.Module):
    """Parameter block that holds X, Y, Z for a single iteration.
    X, Y, Z are learnable complex matrices (broadcast across batch).
    Dimensions:
        Nt: transmit antenna count
        d : stream/user dimension (columns of V)
    """
    def __init__(self, Nt: int, d: int):
        super().__init__()
        # initialize with small random values
        self.X = nn.Parameter(0.1 * torch.randn(Nt, Nt, dtype=torch.cfloat))
        self.Y = nn.Parameter(0.1 * torch.randn(Nt, Nt, dtype=torch.cfloat))
        self.Z = nn.Parameter(0.1 * torch.randn(Nt, Nt, dtype=torch.cfloat))

    def forward(self, A: torch.Tensor, H: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute U_k or W_k like expression:
        out = (A^+ X + A Y + Z) H V
        Args:
            A : (..., Nt, Nt) complex tensor (e.g., A_k)
            H : (..., Nr, Nt) complex tensor
            V : (..., Nt, d ) complex tensor
        Returns: (..., Nt, d) complex tensor
        """
        term = complex_mm(hermitian(A), self.X) + complex_mm(A, self.Y) + self.Z
        HV = complex_mm(H, V)  # (..., Nr, d)
        out = complex_mm(term, complex_mm(H, V))  # (..., Nt, d)
        return out

############################################################
#  Iteration block                                          #
############################################################

class IterationBlock(nn.Module):
    """One unfolded iteration consisting of F, G, J as in the diagram."""
    def __init__(self, Nt: int, d: int, power: float):
        super().__init__()
        self.power = power
        self.F = ParamBlock(Nt, d)
        self.G = ParamBlock(Nt, d)
        self.J = ParamBlock(Nt, d)

    def forward(self, A: torch.Tensor, E: torch.Tensor, H: torch.Tensor, V_prev: torch.Tensor):
        # F-stage
        U = self.F(A, H, V_prev)
        # G-stage
        W = self.G(E, H, U)
        # J-stage produces next V
        V = self.J(A, H, W)
        # enforce power constraint
        V = proj_power(V, self.power)
        return V, U, W

############################################################
#  Full network                                             #
############################################################

class NovelBeamformerNet(nn.Module):
    """Main network wrapping L iteration blocks.

    Args:
        Nt: transmit antennas
        d : streams per user (columns of V)
        L : number of unfolded iterations
        power: total transmit power budget
    """
    def __init__(self, Nt: int, d: int, L: int, power: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList([IterationBlock(Nt, d, power) for _ in range(L)])
        self.L = L
        self.power = power

    def forward(self, H: torch.Tensor, V0: Optional[torch.Tensor] = None):
        """Forward pass.
        Args:
            H: (..., Nr, Nt) complex channel matrix
            V0: optional initial precoder (..., Nt, d). If None, use isotropic.
        Returns:
            V_L: final precoder (..., Nt, d)
        """
        batch_shape = H.shape[:-2]
        Nt = H.size(-1)
        d = self.blocks[0].F.X.size(-1)
        Nr = H.size(-2)

        if V0 is None:
            V0 = torch.randn(*batch_shape, Nt, d, dtype=torch.cfloat)
            V0 = proj_power(V0, self.power)

        V = V0
        # dummy placeholders for A and E matrices – in practice these should be
        # provided or computed outside; here we assume identity for illustration.
        A = torch.eye(Nt, dtype=torch.cfloat, device=H.device).expand(*batch_shape, Nt, Nt)
        E = torch.eye(Nt, dtype=torch.cfloat, device=H.device).expand(*batch_shape, Nt, Nt)

        for blk in self.blocks:
            V, _, _ = blk(A, E, H, V)
        return V

############################################################
#  Simple trainer                                           #
############################################################

class Trainer:
    """Utility class for training the novel beamformer network with an arbitrary loss."""
    def __init__(self, model: NovelBeamformerNet, lr: float = 1e-3):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader, loss_fn):
        self.model.train()
        total_loss = 0.0
        for H_batch in loader:
            H_batch = H_batch.to(torch.complex64)
            self.opt.zero_grad()
            V_pred = self.model(H_batch)
            loss = loss_fn(H_batch, V_pred)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        return total_loss / len(loader)

############################################################
#  Example loss: negative sum‑rate (unsupervised)           #
############################################################

def sum_rate_loss(H: torch.Tensor, V: torch.Tensor, noise_var: float = 1.0):
    """Negative sum‑rate for single‑user (treating each column as a stream)."""
    # Y = H V, Nr×d
    Y = complex_mm(H, V)
    Nr, d = Y.shape[-2:]
    eye = torch.eye(d, dtype=torch.cfloat, device=Y.device)
    C = eye + (1.0 / noise_var) * complex_mm(hermitian(Y), Y)
    rate = torch.logdet(C).real
    return -rate.mean()
