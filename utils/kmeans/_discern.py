"""
Torch-based K-Means
by Ali Hassani

DISCERN initializer
"""

import numpy as np
import torch
from .utils import self_similarity_matrix, row_norm, torch_unravel_index


def discern(x, n_clusters=None, max_n_clusters=None, x_norm=None):
    """
    DISCERN initialization

    Parameters
    ----------
    x : torch.Tensor of shape (n_training_samples, n_features)
    n_clusters : int or NoneType
        Estimates the number of clusters if set to None

    max_n_clusters : int or NoneType, default=None
        Defaults to n_training_samples / 2 if None

    x_norm : torch.Tensor of shape (n_training_samples, n_features) or NoneType

    Returns
    -------
    centroids : torch.Tensor of shape (n_clusters, n_features)
    """
    max_n_clusters = max_n_clusters if max_n_clusters is not None else int(x.size(0) / 2) + 1

    if x_norm is None:
        x_norm = row_norm(x)
    similarity_matrix = self_similarity_matrix(x_norm, True)

    centroid_idx_0, centroid_idx_1 = torch_unravel_index(int(torch.argmin(similarity_matrix)), similarity_matrix.shape)
    centroid_idx = [centroid_idx_0, centroid_idx_1]

    remaining = [y for y in range(0, len(similarity_matrix)) if y not in centroid_idx]

    similarity_submatrix = similarity_matrix[centroid_idx, :][:, remaining]

    ctr = 2
    max_n_clusters = len(similarity_matrix) if max_n_clusters is None else max_n_clusters
    find_n_clusters = n_clusters is None or n_clusters < 2

    membership_values = None if not find_n_clusters else np.zeros(max_n_clusters + 1, dtype=float)

    while len(remaining) > 1 and ctr <= max_n_clusters:
        if n_clusters is not None and 1 < n_clusters <= len(centroid_idx):
            break

        min_vector, max_vector = torch.min(similarity_submatrix, dim=0).values, \
                                 torch.max(similarity_submatrix, dim=0).values
        diff_vector = max_vector - min_vector
        membership_vector = torch.square(max_vector) * min_vector * diff_vector
        min_idx = int(torch.argmin(membership_vector).item())
        membership_vector, min_idx, min_value = membership_vector, min_idx, float(membership_vector[min_idx].data)

        new_centroid_idx = remaining[min_idx]
        if find_n_clusters:
            membership_values[ctr] = min_value
        centroid_idx.append(new_centroid_idx)
        remaining.remove(new_centroid_idx)
        similarity_submatrix = similarity_matrix[centroid_idx, :][:, remaining]
        ctr += 1

    if find_n_clusters:
        membership_values = membership_values[:ctr]
        # TODO: torch implementation
        rx = range(0, len(membership_values))
        dy = np.gradient(membership_values, rx)
        d2y = np.gradient(dy, rx)
        kappa = (d2y / ((1 + (dy ** 2)) ** (3 / 2)))
        predicted_n_clusters = int(np.argmin(kappa))
        n_clusters = max(predicted_n_clusters, 2)

    centroid_idx = centroid_idx[:n_clusters]

    return x[centroid_idx, :]
