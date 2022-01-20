import numpy as np
import torch
from spectral import spectral_decomp_sym, spectral_decomp_skew_sym
from tree_sampling import tree_sampling_dpp


def decompose_proposal_dpp(B1, B2, C):
    # decompose for proposal symmetric DPP with L_hat = Z_hat @ X_hat @ Z_hat.T
    eigen_vals1, eigen_vecs1 = spectral_decomp_sym(B1)
    eigen_vals2, eigen_vecs2 = spectral_decomp_skew_sym(B2, C)
    sigmas = torch.diagonal(eigen_vals2, offset=1)[::2].abs()
    sigmas_duplicated = sigmas.repeat(2, 1).T.reshape(-1).abs()
    X_hat = torch.cat((eigen_vals1, sigmas_duplicated))
    Z_hat = torch.hstack((eigen_vecs1, eigen_vecs2))

    # re-orthonormalize
    eigen_vals, eigen_vecs = spectral_decomp_sym(Z_hat * X_hat.sqrt())
    return eigen_vals, eigen_vecs, sigmas


def rejection_sampling_ndpp(tree, eigen_vecs, eigen_vals, get_det_L, get_det_Lhat, rng=None):
    num_rejections = 0
    while (1):
        sample = tree_sampling_dpp(tree, eigen_vecs, eigen_vals, rng)
        rand_num = rng.rand() if rng else np.random.rand()

        if rand_num < get_det_L(sample) / get_det_Lhat(sample):
            break
        num_rejections += 1
    return sample, num_rejections
