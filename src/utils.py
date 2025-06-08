import torch

def calculate_rate(H, sigma):
    n_r = H.shape[0]
    I = torch.eye(n_r, dtype=H.dtype)
    M = I + H @ sigma @ H.conj().T
    rate = torch.log2(torch.det(M))
    return rate

def calculate_sum_rate(H, I_k, V, alpha, sig):
    sum_rate = 0
    for k in range(len(H)):
        for i in range(I_k[k]):
            Nr = H[0][(k, i)].shape[0]
            S = torch.zeros((Nr, Nr)).to(torch.cfloat)
            for j in range(len(H)):
                for l in range(I_k[k]):
                    if (j, l) == (k, i):
                        continue
                    S += H[j][(k, i)].to(torch.cfloat) @ V[j][l].to(torch.cfloat) @ V[j][l].conj().T.to(torch.cfloat) @ H[j][(k, i)].conj().T.to(torch.cfloat)
            S += sig[k][i] * torch.eye(Nr).to(torch.cfloat)
            tmp = torch.eye(Nr).to(torch.cfloat) + H[k][(k, i)].to(torch.cfloat) @ V[k][i].to(torch.cfloat) @ V[k][i].conj().T.to(torch.cfloat) @ H[k][(k, i)].conj().T.to(torch.cfloat) @ torch.linalg.inv(S)
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
                    Omeg1 += V[str(i)][str(k)] @ V[str(i)][str(k)].conj().T
            Omeg1 = H.iloc[i, j] @ Omeg1 @ H.iloc[i, j].conj().T
            Omeg2 = 0
            for k in range(K):
                Omeg2 += torch.trace(V[str(i)][str(k)] @ V[str(i)][str(k)].conj().T)
            Omeg2 = (1/PT) * Omeg2 * ey
            Omeg = torch.linalg.inv(Omeg1 + Omeg2)
            tmp = ey + H.iloc[i, j] @ V[str(i)][str(j)] @ V[str(i)][str(j)].conj().T @ H.iloc[i, j].conj().T @ Omeg
            rate = (1/torch.log(torch.tensor(2.0))) * torch.logdet(tmp)
            s += rate
        s_rate.append(s)
    return (sum(s_rate)/len(s_rate)).real

def calculate_sum_rate_sc(H, V, alpha, sig):
    # Calculate sum rate for single cell
    sum_rate = 0
    for k in range(len(H)):
        Nr = H[str(k)].shape[0]
        # Calculate Omega
        S = 0
        for l in range(len(H)):
            if l == k: pass
            else:
                S += H[str(k)] @ V[str(l)] @ V[str(l)].conj().T @ H[str(k)].conj().T
        S += sig[k] * torch.eye(Nr, dtype=torch.cdouble)
        tmp = torch.eye(Nr, dtype=torch.cdouble) + H[str(k)] @ V[str(k)] @ V[str(k)].conj().T @ H[str(k)].conj().T @ torch.linalg.inv(S)
        R = torch.log2(torch.linalg.det(tmp))
        sum_rate += alpha[k] * R
    return sum_rate