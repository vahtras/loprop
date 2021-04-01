import abc

import numpy as np


class Normalizer(abc.ABC):

    def __init__(self, overlap, threshold=1e-7):
        self.overlap = overlap
        self.threshold = threshold

    @abc.abstractmethod
    def normalize(self, b):
        "Implements truncate and normalize"


class Lowdin(Normalizer):

    def normalize(self, vectors):
        inverse_sqrt = self.transformer(vectors)
        return vectors @ inverse_sqrt

    def transformer(self, vectors):
        overlap = vectors.T @ self.overlap @ vectors
        l, T = np.linalg.eigh(overlap)
        mask = l > self.threshold
        inverse_sqrt = T[:, mask] * np.sqrt(1 / l[mask]) @ T[:, mask].T
        return inverse_sqrt


class GramSchmidt(Normalizer):

    def normalize(self, vectors):
        """
        Return Gram-Schmidt normalized basis
        """

        new = vectors[:, :1] / np.sqrt(vectors[:, 0] @ self.overlap @ vectors[:, 0])

        for column in vectors.T[1:]:
            column = column - new @ (new.T @ self.overlap @ column)
            norm = np.sqrt(column @ self.overlap @ column)
            if norm > self.threshold:
                column /= norm
                new = np.append(new, column.reshape((len(column), 1)), axis=1)
        return new

    def transformer(self, vectors):
        on_vectors = self.normalize(vectors)
        return np.linalg.inv(on_vectors.T @ self.overlap @ vectors)


class QR(Normalizer):

    def normalize(self, basis):

        q, r = np.linalg.qr(basis)
        mask = np.max(np.abs(r), axis=1) > self.threshold
        return q[:, mask]


def triangular_symmetric(A):
    indices = np.tril_indices(A.shape[0])
    return .5 * (A + A.T)[indices]


def upper_triangular_symmetric(A):
    indices = np.triu_indices(A.shape[0])
    return .5 * (A + A.T)[indices]
