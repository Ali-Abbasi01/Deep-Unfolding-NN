import os
import sys
import json
import torch
import pandas as pd
import importlib
# Get the current working directory
scripts_dir = os.getcwd()
# Go up one level
project_root = os.path.abspath(os.path.join(scripts_dir, '..'))
sys.path.append(project_root)

import src.WMMSE
importlib.reload(src.WMMSE)
from src.WMMSE import WMMSE_alg

import src.utils
importlib.reload(src.utils)
from src.utils import calculate_sum_rate

def generate_random_mimo_channels(n_rx, n_tx, dtype=torch.complex64):
    real_part = torch.randn(num_samples, n_rx, n_tx)
    imag_part = torch.randn(num_samples, n_rx, n_tx)
    channels = torch.complex(real_part, imag_part).to(dtype)
    return channels

def main(num_samples, n_T, n_R, Pt, num_users, d, WMMSE_iter=20):
    # d: Number of data streams

    data_records = []
    for i in range(num_samples):
        H = {}
        H[0] = {}
        for k in range(num_users):
            H[0][(0, k)] = generate_random_mimo_channels(n_R, n_T, dtype=torch.complex64)
        
        V_l = []
        U_l = []
        W_l = []
        sum_rate_l = []
        wmmse = WMMSE_alg(K=1, I_k=[num_users], n_tx=[n_T], n_rx=[[n_R]*num_users], H=H, P_k=[Pt], sig_i_k=[[1]*num_users],
                          d=d, alpha=[[1]*num_users], max_iter_mu=1000, tol_mu=0.001, max_iter_alg=1000, tol_alg=0.001)
        for _ in range(WMMSE_iter):
            V, U, W = wmmse.algorithm()
            V_l.append(V)
            U_l.append(U)
            W_l.append(W)
            sum_rate = calculate_sum_rate(H, V)
            sum_rate_l.append(sum_rate)
        idx = sum_rate_l.index(max(sum_rate_l))
        V = list(V_l[idx][0].values())
        U = list(U_l[idx][0].values())
        W = list(W_l[idx][0].values())
        sum_rate = sum_rate_l[idx]
        H = list(H[0].values())

        H_serial = [
            {"real": m.real.tolist(), "imag": m.imag.tolist()}
            for m in H
        ]
        V_serial = [
            {"real": m.real.tolist(), "imag": m.imag.tolist()}
            for m in V
        ]
        U_serial = [
            {"real": m.real.tolist(), "imag": m.imag.tolist()}
            for m in U
        ]
        W_serial = [
            {"real": m.real.tolist(), "imag": m.imag.tolist()}
            for m in W
        ]

        # Append one record (row)
        data_records.append({
            "H": H_serial,
            "V": V_serial,
            "U": U_serial,
            "W": W_serial,
            "sum_rate": sum_rate
        })

    with open("synthesized_data.json", "w") as f:
        json.dump(data_records, f, indent=2)

num_samples = 1000
n_T = 4
n_R = 4
Pt = 4

if __name__ == "__main__":
    main(num_samples, n_T, n_R, Pt)
 