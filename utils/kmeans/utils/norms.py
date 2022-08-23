"""
Torch-based K-Means
by Ali Hassani

Norm utils
"""

from torch.nn import functional as F


def squared_norm(x):
    """
    Computes and returns the squared norm of the input 2d tensor on dimension 1.
    Useful for computing euclidean distance matrix.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    x_squared_norm : torch.Tensor of shape (n, )
    """
    return (x ** 2).sum(1).view(-1, 1)


def row_norm(x):
    """
    Computes and returns the row-normalized version of the input 2d tensor.
    Useful for computing cosine similarity matrix.

    Parameters
    ----------
    x : torch.Tensor of shape (n, m)

    Returns
    -------
    x_normalized : torch.Tensor of shape (n, m)
    """
    return F.normalize(x, p=2, dim=1)
