"""
Torch-based K-Means
by Ali Hassani

K-Means
"""

import random
import numpy as np
import torch
from .utils import distance_matrix, similarity_matrix, squared_norm, row_norm
from ._kmeanspp import k_means_pp
from ._discern import discern


class _BaseKMeans:
    """
    Base K-Means : DO NOT USE DIRECTLY

    Parameters
    ----------
    n_clusters : int or NoneType
        The number of clusters or `K`. Set to None ONLY when init = 'discern'.

    init : 'random', 'k-means++', 'discern', callable or torch.Tensor of shape (n_clusters, n_features)
        Tensor of the initial centroid coordinates, one of the pre-defined methods {'random', 'k-means++',
        'discern'} or callable taking the training data as input and returning the centroid coordinates.

    n_init : int, default=10
        Ignored (Number of initializations).
        NOTE: not yet supported.

    max_iter : int, default=200
        Maximum K-Means iterations.

    metric : 'default' or callable, default='default'
        Distance metric when similarity_based=False and similarity metric otherwise. Default is 'default'
        which uses L2 distance and cosine similarity as the distance and similarity metrics respectively.
        WARNING: This metric does not apply to the pre-defined initialization methods (K-Means++ and DISCERN).
        The callable metrics should take in two tensors of shapes (n, d) and (m, d) and return a tensor of
        shape (n, m).

    similarity_based : bool, default=False
        Whether the metric is a similarity metric or not.

    eps : float, default=1e-6
        Threshold for early stopping.

    Attributes
    ----------
    labels_ : torch.Tensor of shape (n_training_samples,)
        Training cluster assignments

    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Final centroid coordinates

    inertia_ : float
        Sum of squared errors when not similarity_based and sum of similarities when similarity_based

    n_iter_ : int
        The number of training iterations
    """
    def __init__(self, n_clusters=None, init='k-means++', n_init=10, max_iter=200, metric='default',
                 similarity_based=False, eps=1e-6):
        self.n_clusters = n_clusters
        self.init_method = init if type(init) is str or callable(init) else 'k-means++'
        self.cluster_centers_ = init if type(init) is torch.Tensor else None
        self.max_iter = max_iter
        self.metric = metric if callable(metric) else 'default'
        self.similarity_based = similarity_based
        self.eps = eps

        self.center_norm = None
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0

    def _normalize(self, x):
        return row_norm(x) if self.similarity_based else squared_norm(x)

    def _assign(self, x, x_norm=None):
        """
        Takes a set of samples and assigns them to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        if self.similarity_based:
            return self._similarity_based_assignment(x, x_norm)
        return self._distance_based_assignment(x, x_norm)

    def _distance_based_assignment(self, x, x_norm=None):
        """
        Takes a set of samples and assigns them using the metric to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or NoneType

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        if callable(self.metric):
            dist = self.metric(x, self.cluster_centers_)
        else:
            dist = distance_matrix(x, self.cluster_centers_, x_norm=x_norm, y_norm=self.center_norm)
        return torch.argmin(dist, dim=1), torch.sum(torch.min(dist, dim=1).values)

    def _similarity_based_assignment(self, x, x_norm):
        """
        Takes a set of samples and assigns them using the metric to the clusters w.r.t the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        if callable(self.metric):
            dist = self.metric(x, self.cluster_centers_)
        else:
            dist = similarity_matrix(x_norm if x_norm is not None else self._normalize(x),
                                     self.center_norm if self.center_norm is not None \
                                         else self._normalize(self.cluster_centers_),
                                     pre_normalized=True)
        return torch.argmax(dist, dim=1), torch.sum(torch.max(dist, dim=1).values)


class KMeans(_BaseKMeans):
    """
    K-Means
    """
    def _initialize(self, x, x_norm):
        """
        Initializes the centroid coordinates.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        if callable(self.init_method):
            self.cluster_centers_ = self.init_method(x)
            self.n_clusters = self.cluster_centers_.size(0)
        elif self.init_method == 'k-means++':
            self._initialize_kpp(x, x_norm)
        elif self.init_method == 'discern':
            self._initialize_discern(x, x_norm)
        elif self.init_method == 'random':
            self._initialize_random(x)
        else:
            raise NotImplementedError("Initialization `{}` not supported.".format(self.cluster_centers_))
        self.center_norm = self._normalize(self.cluster_centers_)
        return self

    def _initialize_kpp(self, x, x_norm):
        """
        Initializes the centroid coordinates using K-Means++.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        if type(self.n_clusters) is not int:
            raise NotImplementedError("K-Means++ expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        self.cluster_centers_ = k_means_pp(x, n_clusters=self.n_clusters,
                                           x_norm=x_norm if not self.similarity_based else None)
        return self

    def _initialize_discern(self, x, x_norm):
        """
        Initializes the centroid coordinates using DISCERN.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
        x_norm : torch.Tensor of shape (n_samples, ) or shape (n_samples, n_features), or NoneType

        Returns
        -------
        self
        """
        self.cluster_centers_ = discern(x, n_clusters=self.n_clusters,
                                        x_norm=x_norm if self.similarity_based else None)
        self.n_clusters = self.cluster_centers_.size(0)
        return self

    def _initialize_random(self, x):
        """
        Initializes the centroid coordinates by randomly selecting from the training samples.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        if type(self.n_clusters) is not int:
            raise NotImplementedError("Randomized K-Means expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        self.cluster_centers_ = x[random.sample(range(x.size(0)), self.n_clusters), :]
        return self

    def fit(self, x):
        """
        Initializes and fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        x_norm = self._normalize(x)
        self._initialize(x, x_norm)
        self.inertia_ = None
        for itr in range(self.max_iter):
            labels, inertia = self._assign(x, x_norm)
            if self.inertia_ is not None and abs(self.inertia_ - inertia) < self.eps:
                self.labels_ = labels
                self.inertia_ = inertia
                break
            self.labels_ = labels
            self.inertia_ = inertia
            for c in range(self.n_clusters):
                idx = torch.where(labels == c)[0]
                self.cluster_centers_[c, :] = torch.mean(torch.index_select(x, 0, idx), dim=0)
            self.center_norm = self._normalize(self.cluster_centers_)
        self.n_iter_ = itr + 1
        return self

    def transform(self, x):
        """
        Assigns the samples given to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        labels, _ = self._assign(x, self._normalize(x))
        return labels

    def fit_transform(self, x):
        """
        Fits the centroids using the samples given w.r.t the metric, returns the final assignments.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        self.fit(x)
        return self.labels_
