"""
Torch-based K-Means
by Ali Hassani

Validation utils
"""
import torch


def distance_validation(x):
    """
    Clamps the distance matrix to prevent invalid values.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    x_out : torch.Tensor
    """
    return torch.clamp_min(x, 0.0)


def similarity_validation(x):
    """
    Clamps the similarity matrix to prevent invalid values.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    x_out : torch.Tensor
    """
    return torch.clamp(x, 0.0, 1.0)
