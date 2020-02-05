import numpy as np
from scipy.sparse import eye as speye
from scipy.sparse.linalg import cg

import networkx as nx


class GraphThresholdActiveLearner(object):

    def __init__(self, graph, tau, gamma, lamda=1e-3, epsilon=1e-6, alpha=None, use_tau_offset=True):
        """

        :param graph:
        :type graph: nx.Graph
        :param tau:
        :type tau: float
        :param gamma:
        :type gamma: float
        :param lamda:
        :type lamda: float
        :param epsilon:
        :type epsilon: float
        :param alpha:
        :type alpha: float
        """

        self.graph = graph
        self.tau = tau
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        if alpha is None:
            self.alpha = self.epsilon
        else:
            self.alpha = alpha
        self.use_tau_offset = use_tau_offset

        self.V = nx.linalg.laplacian_matrix(self.graph)
        self.V += speye(self.graph.number_of_nodes()) * self.lamda
        self.x = np.zeros(self.graph.number_of_nodes())
        self.n = np.zeros_like(self.x)
        self.mu_hat = np.ones_like(self.x) * self.tau

    def get_next_location(self):
        """

        :return:
        :rtype: int
        """

        delta = abs(self.mu_hat - self.tau) + self.epsilon
        return np.argmin(delta * np.sqrt(self.n + self.alpha))

    def update(self, i, x):
        """

        :param i:
        :type i: int
        :param x:
        :type x: float
        """

        if self.use_tau_offset:
            self.x[i] += (x - self.tau) / self.gamma
        else:
            self.x[i] += x / self.gamma

        self.n[i] += 1
        self.V[i, i] += 1 / self.gamma

        if self.use_tau_offset:
            self.mu_hat, info = cg(self.V, self.x, self.mu_hat - self.tau)
            self.mu_hat += self.tau
        else:
            self.mu_hat, info = cg(self.V, self.x, self.mu_hat)









