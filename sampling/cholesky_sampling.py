import numpy as np
import torch


def cholesky_based_sampling(B, C=None, rng=None):
    n, k = B.shape
    if C is not None:
        assert C.shape[0] == k and C.shape[1] == k
    else:
        C = torch.eye(k)
    rand_nums = rng.rand(n) if rng else np.random.rand(n)

    Y = []
    Q_Y = C
    for j in range(n):
        prob_j = B[j, :].dot((Q_Y @ B[j, :][:, None]).flatten())
        if rand_nums[j] < prob_j:
            norm_const = prob_j
            Y.append(j)
        else:
            norm_const = prob_j - 1

        a1 = Q_Y @ B[j, :][:, None]
        a2 = B[j, :][None] @ Q_Y
        Q_Y = Q_Y - np.outer(a1, a2) / norm_const

    return Y