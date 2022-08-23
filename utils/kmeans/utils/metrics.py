"""
Torch-based K-Means
by Ali Hassani

Metric utils
"""

import torch
from .norms import squared_norm, row_norm
from .validations import distance_validation, similarity_validation


def distance_matrix(x, y, x_norm=None, y_norm=None):
    """
    Returns the pairwise distance matrix between the two input 2d tensors.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    y : torch.Tensor of shape (p, m)
    x_norm : torch.Tensor of shape (n, ) or NoneType
    y_norm : torch.Tensor of shape (p, ) or NoneType

    Returns
    -------
    distance_matrix : torch.Tensor of shape (n, p)
    """
    x_norm = squared_norm(x) if x_norm is None else x_norm
    y_norm = squared_norm(y).T if y_norm is None else y_norm.T
    mat = x_norm + y_norm - 2.0 * torch.mm(x, y.T)
    return distance_validation(mat)


def self_distance_matrix(x):
    """
    Returns the self distance matrix of the input 2d tensor.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    distance_matrix : torch.Tensor of shape (n, n)
    """
    return distance_validation(((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2))


def similarity_matrix(x, y, pre_normalized=False):
    """
    Returns the pairwise similarity matrix between the two input 2d tensors.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    y : torch.Tensor of shape (p, m)
    pre_normalized : bool, default=False
        Whether the inputs are already row-normalized

    Returns
    -------
    similarity_matrix : torch.Tensor of shape (n, p)
    """
    if pre_normalized:
        return similarity_validation((x.matmul(y.T)))
    return similarity_validation((row_norm(x).matmul(row_norm(y).T)))


def self_similarity_matrix(x, pre_normalized=False):
    """
    Returns the self similarity matrix of the input 2d tensor.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)
    pre_normalized : bool, default=False
        Whether the input is already row-normalized

    Returns
    -------
    similarity_matrix : torch.Tensor of shape (n, n)
    """
    if pre_normalized:
        return similarity_validation((1 + x.matmul(x.T)) / 2)
    x_normalized = row_norm(x)
    return similarity_validation((1 + x_normalized.matmul(x_normalized.T)) / 2)
