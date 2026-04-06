import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn import decomposition
from scipy.cluster.hierarchy import ClusterWarning, linkage, cut_tree
from scipy.spatial.distance import squareform
from isosplit6 import isosplit6

from ..schema import SortingBatch, SortingParameters

warnings.filterwarnings("ignore", category=ClusterWarning)


class Isosplit6Clustering(nn.Module):
    """Cluster PCA features using the isosplit6 subdivision method.

    This module encapsulates the CPU boundary: features are moved to CPU,
    clustered via the C++ isosplit6 library + scipy hierarchical merging,
    and the resulting labels are returned as a tensor on the original device.

    Reads:  batch.features
    Writes: batch.labels
    """

    def __init__(self, params: SortingParameters):
        super().__init__()
        self.npca_per_subdivision = params.npca_per_subdivision

    def forward(self, batch: SortingBatch) -> SortingBatch:
        features = batch.features
        assert features is not None

        device = features.device
        features_np = features.cpu().numpy()

        labels_np = _isosplit6_subdivision_method(
            features_np,
            npca_per_subdivision=self.npca_per_subdivision,
        )

        batch.labels = torch.as_tensor(labels_np, dtype=torch.int32, device=device)
        return batch


def _compute_pca_features_cpu(X: np.ndarray, *, npca: int) -> np.ndarray:
    """CPU PCA using sklearn to match the original pipeline exactly."""
    L, D = X.shape
    k = min(npca, L, D)
    if L == 0 or D == 0:
        return np.zeros((0, k), dtype=np.float32)
    pca = decomposition.PCA(n_components=k, random_state=0)
    return pca.fit_transform(X)


def _isosplit6_subdivision_method(
    X: np.ndarray,
    *,
    npca_per_subdivision: int,
    inds: np.ndarray = None,
) -> np.ndarray:
    """Recursive isosplit6 subdivision — runs entirely on CPU/numpy."""
    if inds is not None:
        X_sub = X[inds]
    else:
        X_sub = X

    L = X_sub.shape[0]
    if L == 0:
        return np.zeros((0,), dtype=np.int32)

    features = _compute_pca_features_cpu(X_sub, npca=npca_per_subdivision)
    labels = isosplit6(features)

    K = int(np.max(labels)) if len(labels) > 0 else 0
    if K <= 1:
        return labels

    # Compute centroids and pairwise distances
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(1, K + 1):
        centroids[k - 1] = np.median(X_sub[labels == k], axis=0)
    X_sub = None  # free memory

    dists = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    dists = squareform(dists)

    # Hierarchical split into two groups
    Z = linkage(dists, method="single", metric="euclidean")
    clusters0 = cut_tree(Z, n_clusters=2)
    cluster_inds_1 = np.where(clusters0 == 0)[0] + 1
    cluster_inds_2 = np.where(clusters0 == 1)[0] + 1

    inds1 = np.where(np.isin(labels, cluster_inds_1))[0]
    inds2 = np.where(np.isin(labels, cluster_inds_2))[0]

    if inds is not None:
        inds1_b = inds[inds1]
        inds2_b = inds[inds2]
    else:
        inds1_b = inds1
        inds2_b = inds2

    labels1 = _isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds1_b)
    labels2 = _isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds2_b)

    K1 = int(np.max(labels1))
    ret_labels = np.zeros(L, dtype=np.int32)
    ret_labels[inds1] = labels1
    ret_labels[inds2] = labels2 + K1
    return ret_labels
