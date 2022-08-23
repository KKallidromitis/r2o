"""
Torch-based K-Means
by Ali Hassani

Misc tools
"""
import torch


def torch_unravel_index(index, shape):
    """
    Unravel index for torch tensors
    By ModarTensai -- PyTorch Forums
    https://discuss.pytorch.org/u/ModarTensai

    Parameters
    ---------
    index : int
    shape : tuple

    Returns
    -------
    index : tuple
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
