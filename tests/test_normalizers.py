import numpy as np
import numpy.testing as npt
from hypothesis import given
import hypothesis.strategies as st

from loprop.linalg import GramSchmidt, Lowdin, triangular_symmetric


@given(st.integers(min_value=2, max_value=100))
def test_gram_schmidt(n):

    # Unitary plus random noise
    np.random.seed(0)
    S = np.eye(n) + .1 * np.random.random((n, n))
    S = S.T @ S

    gs = GramSchmidt(S, threshold=1e-5)

    v = gs.normalize(S)
    npt.assert_allclose(v.T @ S @ v, np.eye(v.shape[1]), atol=1e-7)

    t = gs.transformer(S)
    npt.assert_allclose(S @ t, v)


@given(n=st.integers(min_value=2, max_value=100), m=st.integers(min_value=2, max_value=100))
def test_lowdin(n, m):

    if m > n:
        n, m = m, n

    # Unitary plus random noise
    np.random.seed(0)
    S = np.eye(n) + .1 * np.random.random((n, n))
    S = S.T @ S

    pol = Lowdin(S, threshold=1e-5)

    vectors = np.eye(n, m) + .1 * np.random.random((n, m))

    v = pol.normalize(vectors)

    npt.assert_allclose(v.T @ S @ v, np.eye(m), atol=1e-7)

    t = pol.transformer(vectors)
    npt.assert_allclose(vectors @ t, v)


def test_pack_triangular():
    A = np.array([[1, 2], [3, 4]]) 
    np.testing.assert_allclose(
        triangular_symmetric(A),
        [1, 2.5, 4.0]
    )
