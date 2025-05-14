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

def sum_rate_loss_BC(H, V, PT):
    # Recieves H as a dataframe of num_samples rows and num_users columns, V as a dict of num_samples keys, the value of each is a dict of num_users keys and their corresponding V as values. Outputs the average of sum rate across samples.
    num_samples = len(H)
    K = H.shape[1]
    n_tx = H.iloc[0, 0].shape[1]
    n_rx = [H.iloc[0, i].shape[0] for i in range(K)]
    s_rate = []
    for i in range(num_samples):
        s = 0
        for j in range(K):
            ey = torch.eye(n_rx[j], dtype=torch.cfloat)
            Omeg1 = 0
            for k in range(K):
                if k == j: pass
                else:
                    Omeg1 += V[i][k] @ V[i][k].conj().T
            Omeg1 = H.iloc[i, j] @ Omeg1 @ H.iloc[i, j].conj().T
            Omeg2 = 0
            for k in range(K):
                Omeg2 += torch.trace(V[i][k] @ V[i][k].conj().T)
            Omeg2 = (1/PT) * Omeg2 * ey
            Omeg = torch.linalg.inv(Omeg1 + Omeg2)
            tmp = ey + H.iloc[i, j] @ V[i][j] @ V[i][j].conj().T @ H.iloc[i, j].conj().T @ Omeg
            rate = (1/torch.log(torch.tensor(2.0))) * torch.logdet(tmp)
            s += rate
        s_rate.append(s)
    return torch.tensor(s_rate).mean()