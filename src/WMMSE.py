# The WMMSE algorithm

import torch

K = 3  # number of cells
I_k = [2, 3, 1]  # number of users in each cell
n_tx = [4, 2, 3]  # number of antennas at each transmitter
n_rx = [[2, 3], [1, 2, 2], [4]]  # number of antennas at each user in each cell

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
    def __init__(self, K, I_k, n_tx, n_rx, H):
        self.K = K
        self.I_k = I_k
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.H = H

    def algorithm(self):

