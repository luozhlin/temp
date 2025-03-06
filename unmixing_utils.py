import torch as th
import numpy as np
from functools import partial
from torch.nn.functional import conv1d, pad
from pysptools.abundance_maps.amaps import FCLS
from pysptools.abundance_maps.amaps import NNLS

_H_cache = None

def cal_gradient(W, H, Y, bar_alpha, alpha, var, t, mask, type="denoise", __cache_H=False):
    H = (H + 1)/2
    W = W[:, 0]  # (R, 1 , L) ->  (R, L)
    W = (W + 1)/2
    B, R, Hh, Ww = H.shape
    H = H.reshape(R, -1)  # (R, N)
    H = H.T  # (N, R)
    Y = Y.T  # (L, N) -> (N, L)

    global _H_cache

    if type == "denoise":
        HT = H.T  
        # 对W的梯度

        ## data fidelity
        A = Y - H @ W  # 4096, 224
        grad_W = HT @ (Y - H @ W)  # 3,224
        B = H @ grad_W  # 4096,224
        delta_W = (A.T @ B).trace()/ ((B.T @ B).trace()+1e-5)
        delta_W = delta_W * np.sqrt(bar_alpha)/2

        ## ED loss
        # eta1 = 1/2
        # grad_ed = eta1 *2 * (W - th.mean(W, dim=0, keepdim=True))


        # 对H的梯度
        grad_H = (H @ W - Y) @ W.T  # 4096,3
        C = grad_H @ W  # 4096,224 
        delta_H = (A.T @ C).trace() / ((C.T @ C).trace() + 1e-5)
        delta_H = delta_H * np.sqrt(bar_alpha) / 2
        #delta_H = th.tensor(1)
    else:
        raise NotImplementedError
    assert not th.isnan(delta_W),th.isnan(delta_H)
    grad_W = grad_W[:, None] * delta_W  #- grad_ed[:, None]
    grad_H = grad_H[:, None] * delta_H
    grad_H = grad_H.permute(1,2,0).reshape(1, R, Hh, Ww)

    log = {'res': th.norm(A).item(), "H": H.cpu().numpy(), "W": W.cpu().numpy()}

    return grad_W, grad_H, log


def cal_conditional_gradient_W(W, Y, bar_alpha, alpha, var, t, mask, type="dps", __cache_H=False):
    W = W[:, 0]
    W = (W + 1)/2
    if mask is None:
        W_masked = W
        Y_masked = Y
    else:
        W_masked = W[:, ~mask]
        Y_masked = Y

    H = solve_H(Y_masked, W_masked, t, __cache_H=__cache_H)
    N, R = H.shape
    if type == "dmps":
        grad = H.T @ th.inverse(var*th.eye(N, device=H.device)+(1-bar_alpha)/bar_alpha * H@H.T)/np.sqrt(bar_alpha) @ (Y - H@W) * (1-alpha)/np.sqrt(alpha)/2
        A = Y_masked - H @ W_masked
        delta = th.tensor(1.75)
    elif type == "diffun":
        HT = H.T
        grad = HT @ (Y_masked - H @ W_masked)
        A = Y_masked - H @ W_masked
        B = H @ grad
        delta = (A.T @ B).trace()/ ((B.T @ B).trace()+1e-5)
        delta = delta* np.sqrt(bar_alpha)/2
    else:
        raise NotImplementedError
    assert not th.isnan(delta)
    grad = grad[:, None] * delta
    H = H[:, None]
    log = {'res': th.norm(A).item(), "H": H.cpu().numpy(), "W": W.cpu().numpy()}
    return grad, H, log


def solve_H(Y, W, t, __cache_H=True):
    global _H_cache
    R = W.shape[0]
    if t % 5 == 0 or _H_cache is None or __cache_H is False:
        H = FCLS(Y, W) #.cpu.numpy
        _H_cache = H
    else:
        H = _H_cache
    return th.from_numpy(H).float()


def denoising_fn(sigma=1):
    blur = partial(
        conv1d,
        bias=None,
        stride=1,
        padding=0,
    )

    k = 7
    gaussian = np.exp(-((np.arange(k)-k//2)**2/(2*sigma**2)))
    def filter(W):
        weight = th.from_numpy(gaussian)
        weight = weight/th.sum(weight)
        weight = weight[None, None].to(W.device).float()
        W = pad(W, (k//2, k//2), mode="reflect")
        return blur(W, weight)

    return filter


def SAD(s1, s2):
    return np.arccos(np.sum(s1*s2)/(np.linalg.norm(s1+1e-9)*np.linalg.norm(s2+1e-9)))

def SAD_matrix(A, B=None):
    if B is None:
        B = A
    N = A.shape[0]
    M = B.shape[0]
    dist_matrix = np.zeros([N, M], dtype=np.float32)
    for i in range(N):
        for j in range(M):
            dist_matrix[i, j] = SAD(A[i], B[j])

    return dist_matrix

def NSAD(A, n, B=None):
    sad_matrix = SAD_matrix(A, B)
    n_sad = np.sort(sad_matrix, axis=1)[:, n]
    return n_sad


def sample_from(A, S, n, thre=0.1):
    R1 = len(S)
    used_idx = []
    idx = np.arange(len(A))
    np.random.shuffle(idx)
    if R1 < 1:
        S = A[[idx[0]]]
        used_idx.append(idx[0])
        R1 = 1
    R2 = n - R1
    for i in idx:
        if len(S) > R2:
            break
        sad = np.arccos(S @ A[i].T / np.linalg.norm(S+1e-9, axis=1)/np.linalg.norm(A[i]+1e-9)).min()
        # sad = np.max(np.abs(S - A[i]), axis=1).min()
        if sad > thre:
            S = np.concatenate([S , A[[i]]], axis=0)
            used_idx.append(i)
    return S, used_idx


class UnmixingUtils:
    def __init__(self, A, S):
        self.A = A
        self.S = S
        pass

    def hyperSAD(self, A_est):
        Rt = self.A.shape[1]
        Re = A_est.shape[1]
        P = np.zeros([Rt, Re])
        for i in range(Rt):
            d = np.arccos(np.clip(A_est.T @ self.A[:, i] / np.linalg.norm(A_est, axis=0)/np.linalg.norm(self.A[:, i]), -1, 1))
            P[i, np.argmin(d)] = 1

        Ap = A_est @ P.T
        dist = np.zeros(Rt)
        for i in range(Rt):
            dist[i] = np.arccos(np.clip(Ap.T[i] @ self.A[:, i] / np.linalg.norm(Ap.T[i], axis=0)/np.linalg.norm(self.A[:, i]), -1, 1))

        mean_dist = np.mean(np.sort(dist)[:A_est.shape[1]])
        return dist, mean_dist, P

    def hyperRMSE(self, S_est, P):
        # print(P) (6,3)
        N = np.size(self.S, 0)
        print("N", N)
        Sp = S_est.T @ P.T
        # Sp = Sp / np.sum(Sp, axis=1, keepdims=True)
        rmse = self.S - Sp
        rmse = rmse * rmse
        rmse = (np.sqrt(np.sum(rmse, 0) / N))
        # print(rmse)
        armse = np.mean(np.sort(rmse)[:Sp.shape[1]])
        return rmse, armse


def analyse(W, dir):
    from guided_diffusion.spectral_datasets import _list_spectral_files, load_spectral_from_txts
    path = np.array(_list_spectral_files(dir))
    A = load_spectral_from_txts(path, False)
    A = A[~np.any(np.isnan(A), axis=1)]
    names = []
    for wi in W:
        sad = np.arccos(np.clip(wi @ A.T / np.linalg.norm(A.T, axis=0)/np.linalg.norm(wi), -1, 1))
        idx = np.argmin(sad)
        name = path[idx].split("/")[-1]
        names.append(name)
    return names

# -*- coding: utf-8 -*-
import sys
import scipy as sp
import scipy.linalg as splin


#############################################
# Internal functions
#############################################

def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape  # L number of bands (channels), N number of pixels
    [p, N] = x.shape  # p number of endmembers (reduced dimension)

    P_y = sp.sum(Y ** 2) / float(N)
    P_x = sp.sum(x ** 2) / float(N) + sp.sum(r_m ** 2)
    snr_est = 10 * sp.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est


def vca(Y, R, verbose=True, snr_input=0):
    # Vertex Component Analysis
    #
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    #
    #

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if (R < 0 or R > L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = sp.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :R]  # computes the R-projection matrix
        x_p = sp.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y, y_m, x_p);

        # if verbose:
        #     print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        # if verbose:
        #     print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * sp.log10(R)

    #############################################
    # Choosing Projective Projection or
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            # print("... Select proj. to R-1")

            d = R - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = sp.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :d]  # computes the p-projection matrix
                x_p = sp.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = sp.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x ** 2, axis=0)) ** 0.5
            y = sp.vstack((x, c * sp.ones((1, N))))
    else:
        # if verbose:
        #     print("... Select the projective proj.")

        d = R
        Ud = splin.svd(sp.dot(Y, Y.T) / float(N))[0][:, :d]  # computes the p-projection matrix

        x_p = sp.dot(Ud.T, Y)
        Yp = sp.dot(Ud, x_p[:d, :])  # again in dimension L (note that x_p has no null mean)

        x = sp.dot(Ud.T, Y)
        u = sp.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / sp.dot(u.T, x)

    #############################################
    # VCA algorithm
    #############################################



    indice = sp.zeros((R), dtype=int)
    A = sp.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1);
        f = w - sp.dot(A, sp.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = sp.dot(f.T, y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]
 

    return Ae, indice, Yp
