import torch

def calculate_rate(H, sigma):
    n_r = H.shape[0]
    I = torch.eye(n_r, dtype=H.dtype)
    M = I + H @ sigma @ H.conj().T
    rate = torch.log2(torch.det(M))
    return rate

def calculate_sum_rate(H, V, alpha, sig):
    sum_rate = 0
    for k in range(len(H)):
        for i in range(len(H[k])):
            Nr = H[0][(k, i)]
            S = torch.zeros((Nr, Nr))
            for j in range(len(H)):
                for l in range(len(H[k])):
                    if (j, l) == (k, i):
                        continue
                    S += H[j][(k, i)] @ V[j][l] @ V[j][l].conj().T @ H[j][(k, i)].conj().T
            S += sig[k][i] * torch.eye(Nr)
            tmp = torch.eye(Nr) + H[k][(k, i)] @ V[k][i] @ V[k][i].conj().T @ H[k][(k, i)].conj().T @ torch.linalg.inv(S)
            R = torch.log2(torch.linalg.det(tmp))
            sum_rate += alpha[k][i] * R

    return sum_rate