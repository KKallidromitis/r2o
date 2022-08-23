"""
Torch-based K-Means
by Ali Hassani

Common util functions
"""

from .metrics import distance_matrix, self_distance_matrix, similarity_matrix, self_similarity_matrix
from .norms import row_norm, squared_norm
from .tools import torch_unravel_index
