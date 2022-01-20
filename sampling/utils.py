import numpy as np


def compute_total_variation(x, y):
    assert len(x) == len(y)
    if type(x) == dict and type(y) == dict:
        assert sorted(x.keys()) == sorted(y.keys())
        return np.sum([abs(x_ - y[k_]) for k_, x_ in x.items()]) / 2.0
    else:
        raise NotImplementedError


def get_num_points_in_clusters(n, k):
    lam = 5
    num_points_in_clusters = np.random.poisson(lam, (k,))
    num_points_in_clusters = np.round(num_points_in_clusters * (n / sum(num_points_in_clusters)))
    diff = n - sum(num_points_in_clusters)
    idx = np.argmax(num_points_in_clusters)
    num_points_in_clusters[idx] = num_points_in_clusters[idx] + diff
    return num_points_in_clusters


def get_nonuniform_cluster_matrix(n, m, k):
    B = np.zeros((n, m))
    while (1):
        num_points_in_clusters = get_num_points_in_clusters(n, k)
        if np.min(num_points_in_clusters) > 0:
            break

    mean_scaling = 1 / np.sqrt(m)
    indices = np.array([0] + np.cumsum(num_points_in_clusters).tolist())

    for i in range(k):
        idx = np.arange(indices[i], indices[i + 1])
        points_per_cluster = len(idx)
        B[idx.astype(int),:] = np.random.multivariate_normal(\
            mean_scaling * np.random.randn(m),\
            1/points_per_cluster * 0.1 * np.eye(m),\
            (points_per_cluster,))

    return B[np.random.permutation(n)]