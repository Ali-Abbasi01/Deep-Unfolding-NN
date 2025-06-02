# The dual-link algorithm

import torch

class dual_link_alg():

    def __init__(self, H, w, PT, tol_alg, max_iter_alg, data_streams):
        self.H = H
        self.L = len(H)
        self.PT = PT
        self.w = w
        self.tol_alg = tol_alg
        self.max_iter_alg = max_iter_alg
        self.d = data_streams

    def algorithm(self, sigma):

        def update_omega(sigma):
            omega = []
            for l in range(self.L):
                s = sum([self.H[l][k] @ sigma[k] @ (self.H[l][k].conj().T) for k in range(self.L) if k!=l])
                omega.append(torch.eye(self.H[l][0].shape[0]) + s)
            return omega

        def update_sigma_hat(omega, sigma):
            sigma_hat = []
            den = 0
            for l in range(self.L):
                den += self.w[l] * torch.trace(torch.linalg.inv(omega[l]) - torch.linalg.inv(omega[l] + (self.H[l][l] @ sigma[l] @ self.H[l][l].conj().T)))
            
            for l in range(self.L):
                num = torch.linalg.inv(omega[l]) - torch.linalg.inv(omega[l] + self.H[l][l] @ sigma[l] @ self.H[l][l].conj().T)
                sigma_hat.append(self.PT * self.w[l] * (num/den))

            return sigma_hat

        def update_omega_hat(sigma_hat):
            omega_hat = []
            for l in range(self.L):
                s = sum([self.H[k][l].conj().T @ sigma_hat[k] @ self.H[k][l] for k in range(self.L) if k!=l])
                omega_hat.append(torch.eye(self.H[0][l].shape[1]) + s)
            return omega_hat

        def update_sigma(omega_hat, sigma_hat):
            sigma = []
            den = 0
            for l in range(self.L):
                den += self.w[l] * torch.trace(torch.linalg.inv(omega_hat[l]) - torch.linalg.inv(omega_hat[l] + (self.H[l][l].conj().T @ sigma_hat[l] @ self.H[l][l])))
            
            for l in range(self.L):
                num = torch.linalg.inv(omega_hat[l]) - torch.linalg.inv(omega_hat[l] + self.H[l][l].conj().T @ sigma_hat[l] @ self.H[l][l])
                sigma.append(self.PT * self.w[l] * (num/den))

            return sigma

        def sum_rate(omega, sigma):
            s_rate = 0
            for l in range(self.L):
                s_rate += self.w[l] * torch.log2(torch.linalg.det(torch.eye(self.H[l][0].shape[0]) + self.H[l][l] @ sigma[l] @ self.H[l][l].conj().T @ torch.linalg.inv(omega[l])))
            return s_rate

        

        # # Initialize Sigma

        # sigma_init = []
        # for l in range(self.L):
        #     V = torch.rand(self.H[0][l].shape[1], self.d[l], dtype=torch.cdouble)
        #     sigma_init.append(V @ V.conj().T)

        # ss = 0
        # for l in range(self.L):
        #     ss += torch.trace(sigma_init[l])
        # sigma_init = [sigma_init[l] * (self.PT/ss.real) for l in range(self.L)]

        # sigma = sigma_init

        # The algorithm
        R_l = []
        sigma_l = []
        omega = update_omega(sigma)
        R = sum_rate(omega, sigma)
        R_l.append(R)
        # for _ in range(self.max_iter_alg):
        while True:
            R_p = R
            sigma_hat = update_sigma_hat(omega, sigma)
            omega_hat = update_omega_hat(sigma_hat)
            sigma = update_sigma(omega_hat, sigma_hat)
            omega = update_omega(sigma)
            R = sum_rate(omega, sigma)
            R_l.append(R)
            sigma_l.append(sigma)

            # Check for convergence
            if torch.abs(R - R_p) <= self.tol_alg:
                break

        return R_l, sigma_l