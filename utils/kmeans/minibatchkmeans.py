"""
Torch-based K-Means
by Ali Hassani

MiniBatch K-Means

Sculley, David. "Web-scale k-means clustering." Proceedings of the 19th international conference on
World wide web. 2010. Manuscript available at: https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

NOTE: In the original mini-batch K-Means paper by David Sculley, the SGD updates were applied per sample
in a mini-batch. In this implementation, in order to make things faster, the updates are applied per
mini-batch by taking a sum over the mini-batch samples within the class, similar to the original K-Means
algorithm. Another difference to the original method in the paper is that one mini-batch is generated
randomly at each iteration by sampling the original dataset, while in this implementation, the entire
dataset is shuffled, split into mini-batches all of which are processed at each iteration.
"""
import numpy as np
import torch
from .kmeans import _BaseKMeans
from ._kmeanspp import k_means_pp


class MiniBatchKMeans(_BaseKMeans):
    """
    Mini Batch K-Means

    Parameters
    ----------
    n_clusters : int
        The number of clusters or `K`.

    init : 'random', 'k-means++' or torch.Tensor of shape (n_clusters, n_features)
        Tensor of the initial centroid coordinates, one of the pre-defined methods {'random'}.

    n_init : int, default=10
        Ignored (Number of initializations).
        NOTE: not yet supported.

    max_iter : int, default=200
        Maximum K-Means iterations.

    metric : 'default' or callable, default='default'
        Distance metric when similarity_based=False and similarity metric otherwise. Default is 'default'
        which uses L2 distance and cosine similarity as the distance and similarity metrics respectively.
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
        init = init if type(init) is torch.Tensor else 'k-means++'
        super(MiniBatchKMeans, self).__init__(n_clusters=n_clusters, init=init, max_iter=max_iter,
                                              metric=metric, similarity_based=similarity_based, eps=eps)

    def _initialize(self, dataloader):
        """
        Initializes the centroid coordinates.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        self
        """
        self.labels_ = None
        self.inertia_ = 0
        self.n_iter_ = 0
        if self.init_method == 'k-means++':
            self._initialize_kpp(dataloader)
        elif self.init_method == 'random':
            self._initialize_random(dataloader)
        else:
            raise NotImplementedError("Initialization `{}` not supported.".format(self.cluster_centers_))
        return self

    def _initialize_random(self, dataloader):
        """
        Initializes the centroid coordinates by randomly selecting from the training samples.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        self
        """
        if type(self.n_clusters) is not int:
            raise NotImplementedError("Randomized K-Means expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        # TODO: Better implementation of random initialization
        self.cluster_centers_, self.center_norm = dataloader.dataset.random_sample(self.n_clusters)
        self.center_norm = self._normalize(self.cluster_centers_)
        return self

    def _initialize_kpp(self, dataloader):
        """
        Initializes the centroid coordinates using K-Means++.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        self
        """
        # TODO: Mini-batch K-Means++
        if type(self.n_clusters) is not int:
            raise NotImplementedError("K-Means++ expects the number of clusters, given {}.".format(type(
                self.n_clusters)))
        x, x_norm = next(iter(dataloader))
        self.cluster_centers_ = k_means_pp(x, n_clusters=self.n_clusters,
                                           x_norm=x_norm if not self.similarity_based else None)
        return self

    def fit(self, dataloader):
        """
        Initializes and fits the centroids using the samples given w.r.t the metric.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        self
        """
        self._initialize(dataloader)
        self.inertia_ = None
        self.cluster_counts = np.zeros(self.n_clusters, dtype=int)
        for itr in range(self.max_iter):
            inertia = self._fit_iter(dataloader)
            if self.inertia_ is not None and abs(self.inertia_ - inertia) < self.eps:
                self.inertia_ = inertia
                break
        self.n_iter_ = itr + 1
        return self

    def _fit_iter(self, dataloader):
        """
        Performs one iteration of the mini-batch K-Means and updates the centroids.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        self
        """
        # TODO: Cleaner and faster implementation
        inertia_ = 0
        for x, x_norm in dataloader:
            labels, inertia = self._assign(x, x_norm)
            inertia_ += inertia
            for c in range(self.n_clusters):
                idx = torch.where(labels == c)[0]
                self.cluster_counts[c] += len(idx)
                if len(idx) > 0:
                    lr = 1 / self.cluster_counts[c]
                    self.cluster_centers_[c, :] = ((1 - lr) * self.cluster_centers_[c, :]) + \
                                                  (lr * torch.sum(torch.index_select(x, 0, idx), dim=0))
            self.center_norm = self._normalize(self.cluster_centers_)
        return inertia_

    def transform(self, dataloader):
        """
        Assigns the samples in the dataloader given to the clusters w.r.t the centroid coordinates
        and metric.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        label_list = []
        for x, x_norm in dataloader:
            labels, _ = self._assign(x, x_norm)
            label_list.append(labels)
        return torch.cat(label_list)

    def transform_tensor(self, x):
        """
        Assigns the samples given to the clusters w.r.t the centroid coordinates and metric.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        labels, _ = self._assign(x)
        return labels

    def fit_transform(self, dataloader):
        """
        Fits the centroids using the samples given w.r.t the metric, returns the final assignments.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader[KMeansDataset]

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
        """
        self.fit(dataloader)
        return self.labels_
