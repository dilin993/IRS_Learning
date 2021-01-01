import numpy as np
from scipy.stats import ncx2

from nodes.node import Node
from scipy.linalg import dft

from itertools import combinations_with_replacement

def db2lin(a):
    return np.power(10, a/10)


def distance(node1, node2):
    return np.linalg.norm(node1.pos - node2.pos)


def lin2db(a):
    return 10 * np.log10(a)


def lin2dbm(a):
    return lin2db(a) + 30


def dft_codebook(mh, mv):
    dft_mh = dft(mh)
    dft_mv = dft(mv)
    return np.kron(dft_mh, dft_mv).T


def discrete_codebook(mh, mv, levels):
    n = mh*mv
    angles = np.linspace(0, np.pi, levels)
    shifts = np.exp(-1j * angles)
    return np.array(list(combinations_with_replacement(shifts, n)))


def calculate_rician_outage_probability(h, sigma_sqr, sigma_n_sqr, gamma_th, p_sig, antnum):
    lambda1 = np.square(np.linalg.norm(h))/sigma_sqr
    a_sqr = 2 * lambda1
    b_sqr = 2 * sigma_n_sqr * gamma_th / (sigma_sqr * p_sig)
    return ncx2.cdf(b_sqr, 2 * antnum, a_sqr)


def calculate_sumrate(h, sigma_n_sqr, p_sig):
    return np.log2(1 + p_sig * np.square(np.linalg.norm(h)) / sigma_n_sqr)


def generate_array_response(n_x, n_y, n_z, delta, theta, phi):
    kd = 2 * np.pi * delta
    nx_ind = np.arange(n_x)
    ny_ind = np.arange(n_y)
    nz_ind = np.arange(n_z)
    nxx_ind = np.tile(nx_ind, (1, n_y * n_z))
    nyy_ind = np.tile(np.reshape(np.tile(ny_ind, (n_x, 1)), (1, n_x * n_y)), (1, n_z))
    nzz_ind = np.reshape(np.tile(nz_ind, (n_x * n_y, 1)), (1, n_x * n_y * n_z))
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    gamma = nxx_ind * gamma_x + nyy_ind * gamma_y + nzz_ind * gamma_z
    return np.exp(gamma)


def path_loss_los(d, fc):
    pl_db = 32.4 + 21 * np.log10(d) + 20 * np.log10(fc)
    return db2lin(-pl_db)

