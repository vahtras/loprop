#!/usr/bin/env python
"""
Loprop model implementation (J. Chem. Phys. 121, 4494 (2004))
"""

import abc
import builtins
from collections import defaultdict
from functools import reduce, partial
import sys
from typing import List

import numpy

from util import full, blocked, subblocked
from util.full import Matrix

AU2ANG = 0.5291772108
ANG2AU = 1.0 / AU2ANG
DEFAULT_CUTOFF = 1.6

# Bragg-Slater radii () converted from Angstrom to Bohr
RBS = [
    -1,
    0.25, 0.25,                                                  # H-He
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,              # Li-Ne
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.00,              # Na-Ar
    2.20, 1.80,                                                  # K-Ca
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,  # Sc-Zn
    1.30, 1.25, 1.15, 1.15, 1.15, 1.15,                          # Ga-Kr
    2.35, 2.00,                                                  # Rb-Sr
    1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,  # Y-Cd
    1.55, 1.45, 1.45, 1.40, 1.40, 1.40,                          # In-Xe
    2.6, 2.15,                                                   # Cs-Ba
    1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                    # La-
    1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,                    # -Yb
    1.75, 1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,  # Lu-Hg
    1.90, 1.80, 1.60, 1.90, 1.90, 1.90,                          # Tl-Rn
    2.6, 2.15,                                                   # Fr-Ra
    1.95, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75,                    # Ac-
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75                     # -No
]

RBS = numpy.array(RBS)*ANG2AU

bond_co = defaultdict(lambda: DEFAULT_CUTOFF)
bond_co.update(
    {
        (1, 1): 1.2,
        (1, 6): 1.2,
        (6, 1): 1.2,
        (1, 7): 1.2,
        (7, 1): 1.2,
        (1, 8): 1.2,
        (8, 1): 1.2,
        (6, 6): 1.6,
        (6, 7): 1.6,
        (7, 6): 1.6,
        (6, 8): 1.6,
        (8, 6): 1.6,
        (7, 7): 1.6,
        (7, 8): 1.6,
        (8, 7): 1.6,
        (8, 8): 1.6,
    }
)


def symmetrize_first_beta(beta):
    # naive solution, transforms matrix
    # B[(x,y,z)][(xx, xy, xz, yy, yz, zz)] into array
    # Symmtrized UT array
    # B[ (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz) ]

    new = full.matrix(10)

    new[0] = beta[0, 0]
    new[1] = (beta[0, 1] + beta[1, 0]) / 2
    new[2] = (beta[0, 2] + beta[2, 0]) / 2
    new[3] = (beta[0, 3] + beta[1, 1]) / 2
    new[4] = (beta[0, 4] + beta[1, 2] + beta[2, 1]) / 3
    new[5] = (beta[0, 5] + beta[2, 2]) / 2
    new[6] = beta[1, 3]
    new[7] = (beta[1, 4] + beta[2, 3]) / 2
    new[8] = (beta[1, 5] + beta[2, 4]) / 2
    new[9] = beta[2, 5]

    return new


def penalty_function(alpha=2):
    """Returns function object """

    def pf(Za, Ra, Zb, Rb):
        """Inverse half of penalty function defined in Gagliardi"""

        from math import exp

        ra = RBS[int(round(Za))]
        rb = RBS[int(round(Zb))]

        xa, ya, za = Ra
        xb, yb, zb = Rb
        rab2 = (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2

        f = 0.5 * exp(-alpha * (rab2 / (ra + rb) ** 2))
        return f

    return pf


def pairs(n):
    """
    Generate index pairs for triangular packed matrices up to n
    """
    ij = 0
    for i in range(n):
        for j in range(i + 1):
            yield (ij, i, j)
            ij += 1


def shift_function(F: numpy.ndarray) -> float:
    """
    Returns value twice max value of F

    Arguments:

    :param F: Matrix-like
    :type F: numpy.ndarray

    :returns: max(abs(F))
    :rtype: float
    """

    return 2 * numpy.max(numpy.abs(F))


def header(string, output=None):
    """
    Pretty print header

    Arguments:

    :param string: header text
    :type string: str

    :returns: None

    >>> header(title)

    -----
    title
    -----
    >>>
    """
    print = builtins.print
    if output is not None:
        print = partial(builtins.print, file=output)

    border = "-" * len(string)
    print("\n%s\n%s\n%s" % (border, string, border))


def output_beta(beta, dip=None, fmt="%12.6f"):
    """Repeated output format for b(x; yz)"""
    print("Hyperpolarizability")
    print("beta(:, xx xy xz yy yz zz)")
    print("--------------------------")
    print("beta(x, *) " + (6 * fmt) % tuple(beta[0, :]))
    print("beta(y, *) " + (6 * fmt) % tuple(beta[1, :]))
    print("beta(z, *) " + (6 * fmt) % tuple(beta[2, :]))
    betakk = beta[:, 0] + beta[:, 3] + beta[:, 5]
    print("beta(:, kk)" + (3 * fmt) % tuple(betakk))
    if dip is not None:
        betapar = 0.2 * (betakk & dip) / dip.norm2()
        print("beta//dip  " + (fmt) % betapar)


class LoPropTransformer:
    """
    Class whose instance provide the transformation matrix between 
    an atomic-orbital basis and its LoProp basis, which defined by the overlap matrix
    and occupied/contracted per atom

    Arguments:

    :param S: overlap matrix
    :type S: numpy.ndarray

    :param cpa: contracted per atom
    :type cpa: list[int]

    :param opa: occupied per atom
    :type opa: list[int]
    """

    def __init__(self, S: numpy.ndarray, cpa: List[int], opa: List[int]):
        self.S = S
        self.set_cpa(cpa)
        self.set_opa(opa)
        self._T = None

    def set_cpa(self, cpa: List[int]):
        """
        Sets contracted per atom

        Arguments:

        :param cpa: number of contracted basis functions per atom
        :type cpa: list[int]
        """

        self.assert_pos_ints(cpa)
        self.cpa = tuple(cpa)
        self.noa = len(self.cpa)

    def set_opa(self, opa: List[List[int]]):
        """
        Sets occupied  per atom

        Arguments:

        :param opa: list of occupied basis functions per atom
        :type opa: list[list[int]]
        """
        for a in opa:
            self.assert_nonneg_ints(a)
        self.opa = tuple(tuple(o for o in a) for a in opa)

    @staticmethod
    def assert_pos_ints(arr: List[int]):
        """
        Assert that we have an array of positive ints

        Arguments:

        :param arr: arr
        :type arr: list[int]
        """

        for a in arr:
            assert isinstance(a, (int, numpy.intc, numpy.int32, numpy.int64)) and a > 0

    @staticmethod
    def assert_nonneg_ints(arr: List[int]):
        """
        Assert that we have an array of non-negative ints

        Arguments:

        :param arr: arr
        :type arr: list[int]
        """

        for a in arr:
            assert isinstance(a, (int, numpy.intc, numpy.int32, numpy.int64)) and a >= 0

    @property
    def T(self):
        """
        Returns the LoProp transformation matrix generated according to the
        following steps

        Given atomic overlap matrix:

        1. Gram-Scmidt orthogonalize in each atomic block
        2. Lowdin orthogonalize subspaces

           * Lowdin orthogonalize occupied subspace
           * Lowdin orthogonalize virtual subspace
        3. project occupied out of virtual
        4. Lowdin orthogonalize virtual

        :returns: T
        :rtype: numpy.ndarray
        """

        if self._T is not None:
            return self._T

        S = self.S

        T1 = self.gram_schmidt_atomic_blocks(S)
        S1 = T1.T @ S @ T1

        P = self.permute_to_occupied_virtual()
        S1P = P.T @ S1 @ P

        T2 = self.lowdin_occupied_virtual(S1P)
        S2 = T2.T @ S1P @ T2

        T3 = self.project_occupied_from_virtual(S2)
        S3 = T3.T @ S2 @ T3

        T4 = self.lowdin_virtual(S3)
        S4 = T4.T @ S3 @ T4

        #
        # permute back to original basis
        #
        S4 = P @ S4 @ P.T
        #
        # Return total transformation
        #
        T = T1 @ P @ T2 @ T3 @ T4 @ P.T
        self._T = T
        return self._T

    def gram_schmidt_atomic_blocks(self, S):
        """
        Orthogonalize within atomic blocks

        Arguments:

        :param S: overlap matrix
        :type S: numpy.ndarray

        :returns: Transformation matrix for GS orthonormalization
        :rtype: numpy.ndarray
        """

        cpa = self.cpa
        opa = self.opa
        #
        # 1. orthogonalize in each atomic block
        #
        nbf = sum(cpa)
        #
        # obtain atomic blocking
        #
        noa = len(cpa)
        nocc = 0
        for at in range(noa):
            nocc += len(opa[at])
        Ubl = full.unit(nbf).subblocked((nbf,), cpa)
        #
        # Diagonalize atom-wise
        #
        T1 = blocked.BlockDiagonalMatrix(cpa, cpa)
        for at in range(noa):
            T1.subblock[at] = Ubl.subblock[0][at].GST(S)
        T1 = T1.unblock()
        return T1

    def permute_to_occupied_virtual(self):
        """
        Reorder AO-basis to occupied-virtual order in two steps

        Arguments:

        :returns: permutation matrix
        :rtype: numpy.ndarray
        """
        return self.P1() @ self.P2()

    def P1(self):
        """
        Builds a permutation matrix such that,
        Within an atomic sub-block, the orbitals are permuted
        to have the occupied listed first

        >>> [o1a1, v1a1, o2a1, v2a2, o1a2, v1a2] @ P1
        [o1a1, o2a1, v1a1, v2a2, o1a2, v1a2]

        Arguments:

        :returns: Permutation matrix
        :rtype: numpy.ndarray
        """

        P1 = subblocked.matrix(self.cpa, self.cpa)
        for at in range(self.noa):
            P1.subblock[at][at][:, :] = full.permute(self.opa[at], self.cpa[at])
        P1 = P1.unblock()
        return P1

    def P2(self):
        """
        Builds a permutation matrix to follow the P1 permutaion
        such that the result is occupied virtual order

        >>> [o1a1, o2a1, v1a1, v2a2, o1a2, v1a2] @ P2
        [o1a1, o2a1, o1a2, v1a1, v2a2, v1a2]

        Arguments:

        :returns: Permutation matrix
        :rtype: numpy.mdarray
        """

        vpa = []
        adim = []
        for at in range(self.noa):
            vpa.append(self.cpa[at] - len(self.opa[at]))
            adim.append(len(self.opa[at]))
            adim.append(vpa[at])
        #
        # dimensions for permuted basis
        #
        pdim = []
        for at in range(self.noa):
            pdim.append(len(self.opa[at]))
        for at in range(self.noa):
            pdim.append(vpa[at])

        P2 = subblocked.matrix(adim, pdim)
        for i in range(0, len(adim), 2):
            P2.subblock[i][i // 2] = full.unit(adim[i])
        for i in range(1, len(adim), 2):
            P2.subblock[i][self.noa + (i - 1) // 2] = full.unit(adim[i])
        P2 = P2.unblock()
        return P2

    def lowdin_occupied_virtual(self, S_ov):
        """
        Given overlap in occupied-virtual order basis,
        Lőwdin orthonormalize occupied and virtual
        blocks separately, returning corresponding
        transformation matrix

        (S_oo⁻¹⁄² 0        )
        (0        S_vv⁻¹⁄² )


        Arguments:

        :param S_ov: overlap matrix in occupied-virtual order
        :type S_ov: numpy.ndarray

        :returns: transformation matrix
        :rtype: numpy.ndrarray
        """

        vpa = [c - len(o) for c, o in zip(self.cpa, self.opa)]
        nocc = sum(len(occ) for occ in self.opa)
        nvirt = sum(vpa)
        ov_dim = (nocc, nvirt)
        S_ov_bl = S_ov.block(ov_dim, ov_dim)
        T_bl = S_ov_bl.invsqrt()
        T = T_bl.unblock()
        return T

    def project_occupied_from_virtual(self, S_ov):
        """
        Project occupied out of virtual

        Arguments:

        :param S_ov: overlap matrix in occupied-virtual order
        :type S_ov: numpy.ndarray

        :returns: transformation matrix
        :rtype: numpy.ndrarray
        """

        vpa = [c - len(o) for c, o in zip(self.cpa, self.opa)]
        nocc = sum(len(occ) for occ in self.opa)
        nvirt = sum(vpa)
        occdim = (nocc, nvirt)

        S_ov_sb = S_ov.subblocked(occdim, occdim)
        nbf = S_ov.shape[0]

        T_sb = full.unit(nbf).subblocked(occdim, occdim)
        T_sb.subblock[0][1] = -S_ov_sb.subblock[0][1]
        T = T_sb.unblock()
        return T

    def lowdin_virtual(self, S_ov):
        """

        Lowdin orthogonalize virtual

        Arguments:

        :param S_ov: overlap in occupied-virtual order
        :type S_ov: numpy.ndarray

        :returns: transformation matrix
        :rtype: numpy.ndarray
        """
        vpa = [c - len(o) for c, o in zip(self.cpa, self.opa)]
        nocc = sum(len(occ) for occ in self.opa)
        occdim = (nocc, sum(vpa))
        T4b = blocked.unit(occdim)
        S3b = S_ov.block(occdim, occdim)
        T4b.subblock[1] = S3b.subblock[1].invsqrt()
        T4 = T4b.unblock()
        return T4


class MolFrag(abc.ABC):
    """
    An instance of the MolFrag class is created and populated with
    data from a Dalton/VeloxChem interface files
    """

    def __init__(
        self,
        tmpdir,
        max_l=0,
        pol=0,
        freqs=None,
        pf=penalty_function,
        sf=shift_function,
        gc=None,
        damping=False,
        real_pol=False,
        imag_pol=False,
        **kwargs
    ):
        """
        Constructur of MolFrag class objects
        input: tmpdir, scratch directory of Dalton/VeloxChem calculation
        """
        self.max_l = max_l
        self.pol = pol
        self.tmpdir = tmpdir
        if freqs is None:
            self.freqs = (0,)
            self.nfreqs = 1
        else:
            self.freqs = freqs
            self.nfreqs = len(freqs)
        self.rfreqs = range(self.nfreqs)
        self.damping = damping
        self._real_pol = real_pol
        self._imag_pol = imag_pol

        self.pf = pf
        self.sf = sf
        self.gc = gc

        self._T = None
        self._D = None
        self._Dk = None
        self._D2k = None
        self._x = None

        self._Rc = None
        self._Qab = None
        self._Da = None
        self._Dab = None
        self._Dsym = None
        self._QUab = None
        self._QUa = None
        self._QUsym = None
        self._QUN = None
        self._QUc = None
        self._dQa = None
        self._d2Qa = None
        self._dQab = None
        self._d2Qab = None
        self._Fab = None
        self._la = None
        self._l2a = None
        self._Aab = None
        self._Bab = None
        self._Am = None
        self._Bm = None
        self._dAab = None
        self._dBab = None

    @property
    def real_pol(self):
        return self._real_pol

    @property
    def imag_pol(self):
        return self._imag_pol

    def set_real_pol(self):
        self._real_pol = True
        self._imag_pol = False

    def set_imag_pol(self):
        self._real_pol = False
        self._imag_pol = True

    @abc.abstractmethod
    def get_basis_info(self):
        """
        Obtain basis set info

        Sets
            self.nbf: number of basis functions
            self.cpa: number of contraced/atom
            self.opa: lists of occupied/atom
            self.noa: number of atoms
        """

    @abc.abstractmethod
    def get_molecule_info(self):
        """
        Molecular info: nuclear charges, coordinates
        """

    @property
    def Rc(self):
        """
        Form Rc  molecular gauge origin, default nuclear center of charge
        """
        if self._Rc is not None:
            return self._Rc
        if self.gc is None:
            self._Rc = self.Z @ self.R * (1 / self.Z.sum())
        else:
            self._Rc = numpy.array(self.gc).view(full.matrix)
        return self._Rc

    @abc.abstractmethod
    def S(self):
        """
        Get overlap matrix in AO basis

        """

    @property
    def D(self):
        """
        Density from SIRIFC in blocked loprop basis
        """
        if self._D is not None:
            return self._D

        D = self.get_density_matrix()
        Ti = self.T.I
        self._D = (Ti @ D @ Ti.T).subblocked(self.cpa, self.cpa)
        return self._D

    @abc.abstractmethod
    def get_density_matrix(self):
        ...

    @abc.abstractmethod
    def get_dipole_matrices(self):
        ...

    @property
    def T(self):
        """
        Generate loprop transformation matrix according to the following steps
        Given atomic overlap matrix:
        1. orthogonalize in each atomic block
        2. a) Lowdin orthogonalize occupied subspace
           b) Lowdin orthogonalize virtual subspace
        3. project occupied out of virtual
        4. Lowdin orthogonalize virtual
        Input: overlap S (matrix)
               contracted per atom (list)
               occupied per atom (nested list)
        Returns: transformation matrix T
                 such that T+ST = 1 (unit) """

        if self._T is None:
            self._T = LoPropTransformer(self.S(), self.cpa, self.opa).T
        return self._T

    @property
    def Qab(self):
        """ set charge/atom property"""
        if self._Qab is not None:
            return self._Qab

        D = self.D

        noa = self.noa
        _Qab = full.matrix((noa, noa))
        for a in range(noa):
            _Qab[a, a] = -D.subblock[a][a].tr()
        self._Qab = _Qab
        return self._Qab

    @property
    def Qa(self):
        return self.Qab.diagonal()

    @property
    def Dab(self):
        """
        Returns localized atom and bond dipole moments
        d_ab = -<a|x|b> - Qa R_a delta(a, b)
        """

        if self._Dab is not None:
            return self._Dab

        x = self.x
        D = self.D
        Rab = self.Rab
        Qab = self.Qab

        noa = self.noa
        _Dab = full.matrix((3, noa, noa))
        for i in range(3):
            for a in range(noa):
                for b in range(noa):
                    _Dab[i, a, b] = (
                        -(x[i].subblock[a][b] & D.subblock[a][b])
                        - Qab[a, b] * Rab[a, b, i]
                    )

        self._Dab = _Dab
        return self._Dab

    @property
    def Da(self):
        """
        Sum up dipole bond contributions to atom
        """

        if self._Da is not None:
            return self._Da

        Dab = self.Dab
        self._Da = Dab.sum(axis=2).view(full.matrix)
        return self._Da

    @property
    def Dsym(self):
        """
        Symmetrize density contributions from atom pairs
        """
        if self._Dsym is not None:
            return self._Dsym

        Dab = self.Dab
        noa = self.noa
        dsym = full.matrix((3, noa * (noa + 1) // 2))
        ab = 0
        for a in range(noa):
            for b in range(a):
                dsym[:, ab] = Dab[:, a, b] + Dab[:, b, a]
                ab += 1
            dsym[:, ab] = Dab[:, a, a]
            ab += 1

        self._Dsym = dsym
        return self._Dsym

    @property
    def Dtot(self):
        _Dtot = self._Detot() + self._Dntot()
        return _Dtot

    def _Detot(self):
        Detot = self.Da.sum(axis=1).view(full.matrix)
        return Detot

    def _Dntot(self):
        Dntot = self.Qa @ self.R - self.Qa.sum() * self.Rc
        return Dntot

    @property
    def QUab(self):
        """Quadrupole moment"""
        if self._QUab is not None:
            return self._QUab

        D = self.D
        dRab = self.dRab
        Qab = self.Qab
        Dab = self.Dab

        xy = self.get_quadrupole_matrices()

        noa = self.noa
        QUab = full.matrix((6, noa, noa))
        rrab = full.matrix((6, noa, noa))
        rRab = full.matrix((6, noa, noa))
        RRab = full.matrix((6, noa, noa))
        Rab = self.Rab
        for a in range(noa):
            for b in range(noa):
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        rrab[ij, a, b] = -(xy[ij].subblock[a][b] & D.subblock[a][b])
                        rRab[ij, a, b] = (
                            Dab[i, a, b] * Rab[a, b, j] + Dab[j, a, b] * Rab[a, b, i]
                        )
                        RRab[ij, a, b] = Rab[a, b, i] * Rab[a, b, j] * Qab[a, b]
                        ij += 1
        QUab = rrab - rRab - RRab
        self._QUab = QUab
        #
        # Addition term - gauge correction summing up bonds
        #
        dQUab = full.matrix(self.QUab.shape)
        for a in range(noa):
            for b in range(noa):
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        dQUab[ij, a, b] = (
                            dRab[a, b, i] * Dab[j, a, b] + dRab[a, b, j] * Dab[i, a, b]
                        )
                        ij += 1
        self.dQUab = -dQUab

        return self._QUab

    @property
    def QUa(self):
        """Sum up quadrupole bond terms to atoms"""
        if self._QUa is not None:
            return self._QUa

        QUab = self.QUab + self.dQUab

        self._QUa = QUab.sum(axis=2).view(full.matrix)
        return self._QUa

    @property
    def QUsym(self):
        """Quadrupole moment symmetrized over atom pairs"""
        if self._QUsym is not None:
            return self._QUsym

        QUab = self.QUab
        noa = self.noa
        qusym = full.matrix((6, noa * (noa + 1) // 2))
        ab = 0
        for a in range(noa):
            for b in range(a):
                qusym[:, ab] = QUab[:, a, b] + QUab[:, b, a]
                ab += 1
            qusym[:, ab] = QUab[:, a, a]
            ab += 1

        self._QUsym = qusym
        return self._QUsym

    @property
    def QUN(self):
        """Nuclear contribution to quadrupole"""
        if self._QUN is not None:
            return self._QUN

        qn = full.matrix(6)
        Z = self.Z
        R = self.R
        Rc = self.Rc
        for a in range(len(Z)):
            ij = 0
            for i in range(3):
                for j in range(i, 3):
                    qn[ij] += Z[a] * (R[a, i] - Rc[i]) * (R[a, j] - Rc[j])
                    ij += 1
        self._QUN = qn
        return self._QUN

    @property
    def QUc(self):
        if self._QUc is not None:
            return self._QUc

        rRab = full.matrix((6, self.noa, self.noa))
        RRab = full.matrix((6, self.noa, self.noa))
        Rabc = 1.0 * self.Rab
        for a in range(self.noa):
            for b in range(self.noa):
                Rabc[a, b, :] -= self.Rc
        for a in range(self.noa):
            for b in range(self.noa):
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        rRab[ij, a, b] = (
                            self.Dab[i, a, b] * Rabc[a, b, j]
                            + self.Dab[j, a, b] * Rabc[a, b, i]
                        )
                        RRab[ij, a, b] = (
                            self.Qab[a, b]
                            * (self.R[a, i] - self.Rc[i])
                            * (self.R[b, j] - self.Rc[j])
                        )
                        ij += 1
        QUcab = self.QUab + rRab + RRab
        self._QUc = QUcab.sum(axis=2).sum(axis=1).view(full.matrix)
        return self._QUc

    def ao_to_blocked_loprop(self, *aos):
        cpa = self.cpa
        T = self.T
        return ((T.T @ ao @ T).subblocked(cpa, cpa) for ao in aos)

    def contravariant_ao_to_blocked_loprop(self, aos: dict) -> dict:

        cpa = self.cpa
        T = self.T
        blocked_loprop = {
            k: (T.I @ v @ T.I.T).subblocked(cpa, cpa) for k, v in aos.items()
        }
        return blocked_loprop

    @property
    def Fab(self, **kwargs):
        """Penalty function"""
        if self._Fab is not None:
            return self._Fab

        Fab = full.matrix((self.noa, self.noa))
        for a in range(self.noa):
            Za = self.Z[a]
            Ra = self.R[a]
            for b in range(a):
                Zb = self.Z[b]
                Rb = self.R[b]
                Fab[a, b] = self.pf(Za, Ra, Zb, Rb, **kwargs)
                Fab[b, a] = Fab[a, b]
        for a in range(self.noa):
            Fab[a, a] += -Fab[a, :].sum()
        self._Fab = Fab
        return self._Fab

    @property
    def la(self):
        """Lagrangian for local polarizabilities"""
        #
        # The shift should satisfy
        #   sum(a) sum(b) (F(a,b) + C)l(b) = sum(a) dq(a) = 0
        # =>sum(a, b) F(a, b) + N*C*sum(b) l(b) = 0
        # => C = -sum(a, b)F(a,b) / sum(b)l(b)
        #

        if self._la is not None:
            return self._la
        #
        dQa = self.dQa
        Fab = self.Fab
        Lab = Fab + self.sf(Fab)
        self._la = [Lab.solve(rhs) for rhs in dQa]
        return self._la

    @property
    def l2a(self):
        """Lagrangian for local hyperpolarizabilities"""
        #
        # The shift should satisfy
        #   sum(a) sum(b) (F(a,b) + C)l(b) = sum(a) dq(a) = 0
        # =>sum(a, b) F(a, b) + N*C*sum(b) l(b) = 0
        # => C = -sum(a, b)F(a,b) / sum(b)l(b)
        #

        if self._l2a is not None:
            return self._l2a
        #
        d2Qa = self.d2Qa
        Fab = self.Fab
        Lab = Fab + self.sf(Fab)
        self._l2a = [Lab.solve(rhs) for rhs in d2Qa]
        return self._l2a

    @abc.abstractmethod
    def Dk(self):
        """
        Read linear response perturbed densities
        """

    @abc.abstractmethod
    def D2k(self):
        """
        Read quadratic response perturbed densities
        """

    @abc.abstractmethod
    def x(self):
        """
        Read dipole matrices to blocked loprop basis
        """

    @property
    def dQa(self):
        """
        Charge shift per atom
        """
        if self._dQa is not None:
            return self._dQa

        noa = self.noa

        Dk = self.Dk
        labs = self.dipole_labels

        dQa = full.matrix((self.nfreqs, noa, 3))
        for a in range(noa):
            for il, l in enumerate(labs):
                for iw, w in enumerate(self.freqs):
                    dQa[iw, a, il] = -Dk[(l, w)].subblock[a][a].tr()
        self._dQa = dQa
        return self._dQa

    @property
    def d2Qa(self):
        """Charge shift per atom"""
        if self._d2Qa is not None:
            return self._d2Qa

        noa = self.noa

        D2k = self.D2k

        # static
        wb = wc = 0.0
        d2Qa = full.matrix((1, noa, 6))

        lab = ["XDIPLEN ", "YDIPLEN ", "ZDIPLEN "]
        qrlab = [lab[j] + lab[i] for i in range(3) for j in range(i, 3)]

        for a in range(noa):
            for il, l in enumerate(qrlab):
                il = qrlab.index(l)
                d2Qa[0, a, il] = -D2k[(l, wb, wc)].subblock[a][a].tr()
        self._d2Qa = d2Qa
        return self._d2Qa

    @property
    def dQab(self):
        """
        Charge transfer matrix

        refers to Eq.22 in Gagliardi et al.

        - (λa - λb)/2(pf(rA, rB))

        where λa is the Lagragian and pf the penalty function

        :returns: charge transfer matrix
        :rvalue: numpy.ndarray
        """
        if self._dQab is not None:
            return self._dQab

        la = self.la
        noa = self.noa

        dQab = full.matrix((self.nfreqs, noa, noa, 3))
        for field in range(3):
            for a in range(noa):
                Za = self.Z[a]
                Ra = self.R[a]
                for b in range(a):
                    Zb = self.Z[b]
                    Rb = self.R[b]
                    for w in self.rfreqs:
                        dQab[w, a, b, field] = -(
                            la[w][a, field] - la[w][b, field]
                        ) * self.pf(Za, Ra, Zb, Rb)
                        dQab[w, b, a, field] = -dQab[w, a, b, field]

        self._dQab = dQab
        return self._dQab

    @property
    def d2Qab(self):
        """Charge transfer matrix for double perturbation"""
        if self._d2Qab is not None:
            return self._d2Qab

        l2a = self.l2a
        noa = self.noa

        d2Qab = full.matrix((self.nfreqs, noa, noa, 6))
        for field in range(6):
            for a in range(noa):
                Za = self.Z[a]
                Ra = self.R[a]
                for b in range(a):
                    Zb = self.Z[b]
                    Rb = self.R[b]
                    for w in self.rfreqs:
                        d2Qab[w, a, b, field] = -(
                            l2a[w][a, field] - l2a[w][b, field]
                        ) * self.pf(Za, Ra, Zb, Rb)
                        d2Qab[w, b, a, field] = -d2Qab[w, a, b, field]

        self._d2Qab = d2Qab
        return self._d2Qab

    @property
    def Aab(self):
        """
        Localized polariziabilities:

        Contribution from change in localized dipole moment

            -Δ(r - R(AB)):D(AB) = - r:ΔD(AB) + ΔQ(A)R(A) δ(A, B)

        """

        if self._Aab is not None:
            return self._Aab

        Dk = self.Dk

        cpa = self.cpa
        Rab = self.Rab
        dQa = self.dQa
        x = self.x

        noa = len(cpa)
        labs = self.dipole_labels
        Aab = full.matrix((self.nfreqs, 3, 3, noa, noa))

        # correction term for shifting origin from O to Rab
        for i, li in enumerate(labs):
            for j, lj in enumerate(labs):
                for a in range(noa):
                    for b in range(noa):
                        for jw, w in enumerate(self.freqs):
                            Aab[jw, i, j, a, b] = (
                                -x[i].subblock[a][b] & Dk[(lj, w)].subblock[a][b]
                            )
                    for jw in self.rfreqs:
                        Aab[jw, i, j, a, a] -= dQa[jw, a, j] * Rab[a, a, i]

        self._Aab = Aab
        return self._Aab

    @property
    def dAab(self):
        """
        Charge transfer contribution to bond polarizability
        """

        if self._dAab is not None:
            return self._dAab

        dQab = self.dQab
        dRab = self.dRab
        noa = self.noa
        dAab = full.matrix((self.nfreqs, 3, 3, noa, noa))
        for a in range(noa):
            for b in range(noa):
                for i in range(3):
                    for j in range(3):
                        dAab[:, i, j, a, b] = (
                            dRab[a, b, i] * dQab[:, a, b, j]
                            + dRab[a, b, j] * dQab[:, a, b, i]
                        )
        self._dAab = dAab
        return self._dAab

    @property
    def Am(self):
        """Molecular polarizability:

        To reconstruct the molecular polarizability  from localized
        polarizabilties one has to reexpand in terms of an arbitrary but common
        origin leading to the correction term below

        d<-r> = - sum(A,B) (r-R(A,B))dD(A,B) + R(A) dQ(A) Δ(A,B)
        """

        if self._Am is not None:
            return self._Am

        Aab = self.Aab
        dAab = self.dAab

        self._Am = (Aab + 0.5 * dAab).sum(axis=4).sum(axis=3).view(full.matrix)
        return self._Am

    @property
    def Bab(self):
        """Localized hyperpolariziabilities"""
        if self._Bab is not None:
            return self._Bab

        D2k = self.D2k
        Rab = self.Rab
        d2Qa = self.d2Qa
        x = self.x

        labs = ("XDIPLEN ", "YDIPLEN ", "ZDIPLEN ")
        qlabs = [labs[i] + labs[j] for i in range(3) for j in range(i, 3)]
        Bab = full.matrix((self.nfreqs, 3, 6, self.noa, self.noa))

        # correction term for shifting origin from O to Rab
        for i, li in enumerate(labs):
            for jk, ljk in enumerate(qlabs):
                for a in range(self.noa):
                    for b in range(self.noa):
                        for iw, w in enumerate(self.freqs):
                            Bab[iw, i, jk, a, b] = (
                                -x[i].subblock[a][b] & D2k[(ljk, w, w)].subblock[a][b]
                            )
                    for iw in self.rfreqs:
                        Bab[iw, i, jk, a, a] -= d2Qa[iw, a, jk] * Rab[a, a, i]

        self._Bab = Bab
        return self._Bab

    @property
    def dBab(self):
        """Charge transfer contribution to bond hyperpolarizability"""
        if self._dBab is not None:
            return self._dBab

        d2Qab = self.d2Qab
        dRab = self.dRab
        dBab = full.matrix((self.nfreqs, 3, 6, self.noa, self.noa))
        for a in range(self.noa):
            for b in range(self.noa):
                for i in range(3):
                    for j in range(6):
                        dBab[:, i, j, a, b] = 2 * dRab[a, b, i] * d2Qab[:, a, b, j]
        self._dBab = dBab
        return self._dBab

    @property
    def Bm(self):
        "Molecular hyperpolarizability"

        if self._Bm is not None:
            return self._Bm

        Bab = self.Bab
        dBab = self.dBab
        self._Bm = (Bab + 0.5 * dBab).sum(axis=4).sum(axis=3).view(full.matrix)

        return self._Bm

    def output_by_atom(
        self,
        fmt="%9.5f",
        max_l=0,
        pol=0,
        hyperpol=0,
        bond_centers=False,
        angstrom=False,
        output=None,
    ):
        """
        Printout of localized properties 

        Arguments:

        :param fmt: output format for floats
        :type fmt: str

        :param max_l: max angular momentum
        :type max_l: int

        :param pol: polarizability index (0=None, 1=isotropic, 2=anisotropic)
        :type pol: int

        :param hyperpol: hyperpolarizability index
        :type hyperpol: int

        :param bond_centers: if True include bond centers in output
        :type bond_centers: bool

        :param angstrom: if True, output printed in Ångstrőm, default atomic units
        :type angstrom: bool

        :param output: output stream
        :type output: file-like

        :returns: None
        :rtype: NoneType
        """

        print = builtins.print
        if output is not None:
            print = partial(builtins.print, file=output)

        if max_l >= 0:
            Qab = self.Qab
            Qa = Qab.diagonal()
        if max_l >= 1:
            Dab = self.Dab
            Da = self.Da
        if max_l >= 2:
            QUab = self.QUab
            QUN = self.QUN
            QUa = self.QUa
        if pol:
            Aab = self.Aab + self.dAab
        if hyperpol:
            Bab = self.Bab + self.dBab

        if angstrom:
            xconv = AU2ANG
            xconv3 = AU2ANG ** 3
        else:
            xconv = 1
            xconv3 = 1

        Z = self.Z
        R = self.R
        Rc = self.Rc
        noa = self.noa
        #
        # Form net atomic properties P(a) = sum(b) P(a,b)
        #
        if self._Aab is not None:
            Aab = self.Aab + 0.5 * self.dAab
            Aa = Aab.sum(axis=4)

        if self._Bab is not None:
            Bab = self.Bab + 0.5 * self.dBab
            Ba = Bab.sum(axis=4)

        if bond_centers:
            for a in range(noa):
                for b in range(a):
                    header("Bond    %d %d" % (a + 1, b + 1), output=output)
                    print(
                        "Bond center:       "
                        + (3 * fmt) % tuple(0.5 * (R[a, :] + R[b, :]) * xconv)
                    )
                    if max_l >= 0:
                        print("Electronic charge:   " + fmt % Qab[a, b])
                        print("Total charge:        " + fmt % Qab[a, b])
                    if self._Dab is not None:
                        print(
                            "Electronic dipole    "
                            + (3 * fmt) % tuple(Dab[:, a, b] + Dab[:, b, a])
                        )
                        print(
                            "Electronic dipole norm"
                            + fmt % (Dab[:, a, b] + Dab[:, b, a]).norm2()
                        )
                    if self._QUab is not None:
                        raise NotImplementedError
                    if self._Aab is not None:
                        for iw, w in enumerate(self.freqs):
                            Asym = Aab[iw, :, :, a, b] + Aab[iw, :, :, b, a]
                            if pol > 0:
                                print(
                                    "Isotropic polarizability (%g)" % w
                                    + fmt % (Asym.trace() / 3 * xconv3)
                                )
                            if pol > 1:
                                print("Polarizability (%g)      " % w)
                                print(
                                    (6 * fmt) % tuple(Asym.pack().view(Matrix) * xconv3)
                                )

                    if self._Bab is not None:
                        for iw, w in enumerate(self.freqs):
                            Bsym = Bab[iw, :, :, a, b] + Bab[iw, :, :, b, a]
                            output_beta(Bsym, self.Da[:, a])

                header("Atom    %d" % (a + 1), output=output)
                print("Atom center:       " + (3 * fmt) % tuple(R[a, :] * xconv))
                print("Nuclear charge:    " + fmt % Z[a])
                if max_l >= 0:
                    print("Electronic charge:   " + fmt % Qab[a, a])
                    print("Total charge:        " + fmt % (Z[a] + Qab[a, a]))
                if self._Dab is not None:
                    print("Electronic dipole    " + (3 * fmt) % tuple(Dab[:, a, a]))
                    print("Electronic dipole norm" + fmt % Dab[:, a, a].norm2())
                if self._QUab is not None:
                    print("Electronic quadrupole" + (6 * fmt) % tuple(QUab[:, a, a]))
                if self._Aab is not None:
                    for iw, w in enumerate(self.freqs):
                        Asym = Aab[iw, :, :, a, a]
                        if pol > 0:
                            print(
                                "Isotropic polarizability (%g)" % w
                                + fmt % (Asym.trace() / 3 * xconv3)
                            )
                        if pol > 1:
                            print(
                                "Polarizability (%g)       " % w
                                + (6 * fmt)
                                % tuple(Asym.pack().view(full.matrix) * xconv3)
                            )
                if self._Bab is not None:
                    for iw, w in enumerate(self.freqs):
                        Bsym = Bab[iw, :, :, a, a]
                        output_beta(Bsym, Da[:, a])

        else:
            for a in range(noa):
                header("Atomic domain %d" % (a + 1), output=output)
                print("Domain center:       " + (3 * fmt) % tuple(R[a, :] * xconv))
                line = " 0"
                line += (3 * "%17.10f") % tuple(AU2ANG * R[a, :])
                print("Nuclear charge:      " + fmt % Z[a])
                if self.max_l >= 0:
                    print("Electronic charge:   " + fmt % Qa[a])
                    print("Total charge:        " + fmt % (Z[a] + Qa[a]))
                if self._Dab is not None:
                    print("Electronic dipole    " + (3 * fmt) % tuple(self.Da[:, a]))
                    print(
                        "Electronic dipole norm"
                        + (fmt) % self.Da[:, a].view(full.matrix).norm2()
                    )
                if self._QUab is not None:
                    print("Electronic quadrupole" + (6 * fmt) % tuple(QUa[:, a]))
                if self._Aab is not None:
                    for iw, w in enumerate(self.freqs):
                        Asym = Aa[iw, :, :, a].view(full.matrix)
                        print(
                            "Isotropic polarizablity (w=%g)" % w
                            + fmt % (Aa[iw, :, :, a].trace() / 3 * xconv3)
                        )
                        print(
                            "Electronic polarizability (w=%g)" % w
                            + (6 * fmt) % tuple(Asym.pack().view(Matrix) * xconv3)
                        )
                if self._Bab is not None:
                    for iw, w in enumerate(self.freqs):
                        Bsym = Ba[iw, :, :, a].view(full.matrix)
                        output_beta(Bsym, self.Da[:, a])
        #
        # Total molecular properties
        #
        Ztot = Z.sum()
        if self.max_l >= 0:
            Qtot = Qa.sum()
        if self.max_l >= 1:
            Dm = self.Da.sum(axis=1).view(full.matrix)
            Dc = Qa @ (R - Rc)
            DT = Dm + Dc
        if self._QUab is not None:
            QUm = self.QUc
            QUT = QUm + QUN
        if self._Bab is not None:
            Dm = self.Da.sum(axis=1).view(full.matrix)

        header("Molecular", output=output)
        print("Domain center:       " + (3 * fmt) % tuple(Rc * xconv))
        print("Nuclear charge:      " + fmt % Ztot)

        if self.max_l >= 0:
            print("Electronic charge:   " + fmt % Qtot)
            print("Total charge:        " + fmt % (Ztot + Qtot))

        if self.max_l >= 1:
            print("Electronic dipole    " + (3 * fmt) % tuple(Dm))
            print("Gauge   dipole       " + (3 * fmt) % tuple(Dc))
            print("Total   dipole       " + (3 * fmt) % tuple(DT))

        if self._QUab is not None:
            print("Electronic quadrupole" + (6 * fmt) % tuple(QUm))
            print("Nuclear    quadrupole" + (6 * fmt) % tuple(QUN))
            print("Total      quadrupole" + (6 * fmt) % tuple(QUT))

        if self._Aab is not None:
            for iw, w in enumerate(self.freqs):
                Am = self.Am[iw]
                print("Polarizability av (%g)   " % w + fmt % (Am.trace() / 3 * xconv3))
                print(
                    "Polarizability (%g)      " % w
                    + (6 * fmt) % tuple(Am.pack().view(full.matrix) * xconv3)
                )

        if self._Bab is not None:
            for iw, w in enumerate(self.freqs):
                Bm = self.Bm[iw]
                output_beta(Bm, dip=Dm, fmt=fmt)

    def output_template(
        self,
        maxl=0,
        pol=0,
        hyper=0,
        template_full=False,
        decimal=4,
        full_loc=0,
        freqs=None,
    ):
        l_dict = {0: "charge", 1: "dipole", 2: "quadrupole"}
        # Upper triangular alpha
        a_dict = {0: "", 2: "alpha"}
        # Upper triangular beta
        b_dict = {0: "", 2: "beta"}

        fmt = "%." + "%df" % decimal
        line = ""

        if pol > 0:
            Aab = self.Aab + 0.5 * self.dAab
        if hyper > 0:
            Bab = self.Bab + 0.5 * self.dBab

        if maxl not in l_dict:
            raise RuntimeError(
                "ERROR: called output_template with wrong argument range"
            )
        if pol not in a_dict:
            raise RuntimeError(
                "ERROR: called output_template with wrong argument range"
            )
        if hyper not in b_dict:
            raise RuntimeError(
                "ERROR: called output_template with wrong argument range"
            )

        elem_dict = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
        if maxl >= 0:
            if template_full:
                # Put point dipole on center of charge
                line += "( '%s%d', " % (
                    elem_dict[self.Z[full_loc]],
                    full_loc + 1,
                ) + '"charge") : [ %s ],\n' % fmt % (self.Z.sum() + self.Qab.sum())
            else:
                for a in range(self.noa):
                    line += "( '%s%d', " % (
                        elem_dict[self.Z[a]],
                        a + 1,
                    ) + '"charge") : [ %s ],\n' % fmt % (self.Z[a] + self.Qab[a, a])
        if maxl >= 1:
            if template_full:
                Dm = self.Da.sum(axis=1).view(full.matrix)
                Dc = self.Qab.diagonal() @ (self.R - self.Rc)
                DT = Dm + Dc
                line += "( '%s%d', " % (
                    elem_dict[self.Z[full_loc]],
                    full_loc + 1,
                ) + '"dipole") : [ %s, %s, %s ],\n' % tuple(
                    [fmt for i in range(3)]
                ) % tuple(
                    DT
                )
            else:
                for a in range(self.noa):
                    line += "( '%s%d', " % (
                        elem_dict[self.Z[a]],
                        a + 1,
                    ) + '"dipole") : [ %s, %s, %s ],\n' % tuple(
                        [fmt for i in range(3)]
                    ) % (
                        tuple(self.Dab.sum(axis=2)[:, a])
                    )
        if maxl >= 2:
            if template_full:
                line += "( '%s%d', " % (
                    elem_dict[self.Z[full_loc]],
                    full_loc + 1,
                ) + '"quadrupole") : [ %s, %s, %s, %s, %s, %s ],\n' % tuple(
                    [fmt for i in range(6)]
                ) % (
                    tuple((self.QUab + self.dQUab).sum(axis=(1, 2))[:])
                )
            else:
                for a in range(self.noa):
                    line += (
                        "( '%s%d', " % (elem_dict[self.Z[a]], a + 1)
                        + '"quadrupole") : '
                        + "[ %s, %s, %s, %s, %s, %s ],\n"
                        % tuple([fmt for i in range(6)])
                        % (tuple((self.QUab + self.dQUab).sum(axis=2)[:, a]))
                    )
        if pol >= 2:
            if template_full:
                Asym = Aab.sum(axis=(3, 4))[0, :, :].view(full.matrix)
                A = Asym.pack().view(full.matrix).copy()
                A[2], A[3] = A[3], A[2]
                line += "( '%s%d', " % (
                    elem_dict[self.Z[full_loc]],
                    full_loc + 1,
                ) + '"alpha") : [ %s, %s, %s, %s, %s, %s ],\n' % tuple(
                    [fmt for i in range(6)]
                ) % (
                    tuple(A)
                )
            else:
                for a in range(self.noa):
                    # Only for one frequency for now, todo, fix later if needed
                    Asym = Aab.sum(axis=4)[0, :, :, a].view(full.matrix)
                    A = Asym.pack().view(full.matrix).copy()
                    A[2], A[3] = A[3], A[2]
                    line += "( '%s%d', " % (
                        elem_dict[self.Z[a]],
                        a + 1,
                    ) + '"alpha") : [ %s, %s, %s, %s, %s, %s ],\n' % tuple(
                        [fmt for i in range(6)]
                    ) % (
                        tuple(A)
                    )
        if hyper >= 2:
            if template_full:
                Bsym = symmetrize_first_beta(
                    Bab.sum(axis=(3, 4))[0, :, :].view(full.matrix)
                )
                line += (
                    "( '%s%d', " % (elem_dict[self.Z[full_loc]], full_loc + 1)
                    + '"beta") : '
                    + "[ %s, %s, %s, %s, %s, %s, %s, %s, %s, %s ],\n"
                    % tuple([fmt for i in range(len(Bsym))])
                    % tuple(Bsym)
                )
            else:
                for a in range(self.noa):
                    # Only for one frequency for now, todo, fix later if needed
                    Bsym = symmetrize_first_beta(
                        Bab.sum(axis=4)[0, :, :, a].view(full.matrix)
                    )
                    line += (
                        "( '%s%d', " % (elem_dict[self.Z[a]], a + 1)
                        + '"beta") : '
                        + "[ %s, %s, %s, %s, %s, %s, %s, %s, %s, %s ],\n"
                        % (tuple([fmt for i in range(len(Bsym))]))
                        % (tuple(Bsym))
                    )
        print(line)
        return line

    def output_potential_file(
        self, maxl, pol, hyper, bond_centers=False, angstrom=False, decimal=3
    ):
        """Output potential file"""
        fmt = "%" + "%d." % (7 + decimal) + "%df" % decimal
        lines = []
        if angstrom:
            unit = "AA"
            xconv = AU2ANG
            xconv3 = AU2ANG ** 3
        else:
            unit = "AU"
            xconv = 1
            xconv3 = 1
        lines.append(unit)

        noa = self.noa

        if bond_centers:
            # To get number of centers and bonding is on
            bond_mat = numpy.zeros((noa, noa), dtype=int)
            for a in range(noa):
                for b in range(a):
                    r = numpy.sqrt(((self.R[a] - self.R[b]) ** 2).sum())
                    if r < bond_co[(int(self.Z[a]), int(self.Z[b]))] / AU2ANG:
                        bond_mat[a, b] = 1
                        bond_mat[b, a] = 1

        if bond_centers:
            # Where the number of bonds is the diagonal
            # plus each entry with '1'
            # in the upper triangular of bond_mat
            noc = bond_mat.shape[0] + reduce(
                lambda a, x: a + len(numpy.where(x == 1)[0]),
                [row[i + 1:] for i, row in enumerate(bond_mat)],
                0,
            )
            # noc = noa*(noa + 1)/2
        else:
            noc = self.noa

        lines.append("%d %d %d %d" % (noc, maxl, pol, 1))

        if maxl >= 0:
            Qab = self.Qab
        if maxl >= 1:
            Dab = self.Dab
        if maxl >= 2:
            QUab = self.QUab
            dQUab = self.dQUab
        if pol > 0:
            Aab = self.Aab + 0.5 * self.dAab

        if hyper > 0:
            Bab = self.Bab + 0.5 * self.dBab

        if bond_centers:
            ab = 0
            for a in range(noa):
                for b in range(a):
                    if bond_mat[a, b]:
                        line = ("1" + 3 * fmt) % tuple(self.Rab[a, b, :])
                        if maxl >= 0:
                            line += fmt % Qab[a, b]
                        if maxl >= 1:
                            line += (3 * fmt) % tuple(Dab[:, b, a] + Dab[:, a, b])
                        if maxl >= 2:
                            line += (6 * fmt) % tuple(QUab[:, a, b] + QUab[:, b, a])
                        if pol > 0:
                            Aab = self.Aab + 0.5 * self.dAab
                            for iw, w in enumerate(self.freqs):
                                Asym = Aab[iw, :, :, a, b] + Aab[iw, :, :, b, a]
                                if pol == 1:
                                    line += fmt % (Asym.trace() * xconv3 / 3)
                                elif (pol % 10) == 2:
                                    out = Asym.pack().view(Matrix) * xconv3
                                    out[2:4] = out[3:1:-1]
                                    line += (6 * fmt) % tuple(out)
                        if hyper > 0:
                            for iw, w in enumerate(self.freqs):
                                Bsym = Bab[iw, :, :, a, b] + Bab[iw, :, :, b, a]
                                Btotsym = symmetrize_first_beta(Bsym)
                                line += 10 * fmt % tuple(Btotsym)
                        lines.append(line)

                # For atom a, non_bond_pos holds atoms that are not bonded to a
                # Include only non bonded to atomic prop here

                nbond_pos = numpy.where(bond_mat[a] == 0)[0]

                line = ("1" + 3 * fmt) % tuple(self.Rab[a, a, :])
                if maxl >= 0:
                    line += fmt % (self.Z[a] + Qab[a, a])
                if maxl >= 1:
                    line += (3 * fmt) % tuple(
                        reduce(lambda x, y: x + Dab[:, a, y], nbond_pos, 0.0)
                    )
                if maxl >= 2:
                    raise NotImplementedError
                if pol > 0:
                    for iw, w in enumerate(self.freqs):
                        if pol % 10 == 2:
                            out = (
                                reduce(
                                    lambda x, y: x + Aab[iw, :, :, a, y], nbond_pos, 0.0
                                )
                                .pack()
                                .view(full.matrix)
                                * xconv3
                            )
                            out[2:4] = out[3:1:-1]
                            line += (6 * fmt) % tuple(out)
                        elif pol == 1:
                            out = (
                                reduce(
                                    lambda x, y: x + Aab[iw, :, :, a, y], nbond_pos, 0.0
                                )
                                .view(full.matrix)
                                .trace()
                                / 3.0
                                * xconv3
                            )
                            line += fmt % out
                if hyper > 0:
                    for iw, w in enumerate(self.freqs):
                        Bsym = reduce(
                            lambda x, y: x + Bab[iw, :, :, a, y], nbond_pos, 0.0
                        ).view(full.matrix)
                        Btotsym = symmetrize_first_beta(Bsym)
                        line += 10 * fmt % tuple(Btotsym)
                ab += 1
                lines.append(line)
        else:
            for a in range(noa):
                line = ("1" + 3 * fmt) % tuple(self.Rab[a, a, :] * xconv)
                if maxl >= 0:
                    line += fmt % (self.Z[a] + Qab[a, a])
                if maxl >= 1:
                    line += (3 * fmt) % tuple(Dab.sum(axis=2)[:, a])
                if maxl >= 2:
                    line += (6 * fmt) % tuple((QUab + dQUab).sum(axis=2)[:, a])
                if pol > 0:
                    for iw in range(self.nfreqs):
                        Asym = Aab.sum(axis=4)[iw, :, :, a].view(full.matrix)
                        if pol == 1:
                            line += fmt % (Asym.trace() / 3 * xconv3)
                        elif pol % 10 == 2:
                            out = Asym.pack().view(full.matrix)
                            out[2:4] = out[3:1:-1]
                            line += (6 * fmt) % tuple(out * xconv3)

                if hyper > 0:
                    for iw in range(self.nfreqs):
                        Bsym = Bab.sum(axis=4)[iw, :, :, a].view(full.matrix)
                        if hyper == 1:
                            dip = self.Da[:, a]
                            betakk = Bsym[:, 0] + Bsym[:, 3] + Bsym[:, 5]
                            line += fmt % (0.2 * (betakk & dip) / dip.norm2())
                        if hyper == 2:
                            Btotsym = symmetrize_first_beta(Bsym)
                            line += 10 * fmt % tuple(Btotsym)
                lines.append(line)

        return "\n".join(lines) + "\n"

    def print_atom_domain(self, n, angstrom=False):
        fmt = "%9.5f"
        if angstrom:
            xconv = AU2ANG
        else:
            xconv = 1
        retstr = """\
---------------
Atomic domain %d
---------------
Domain center:       """ % (
            n + 1,
        ) + (
            3 * fmt + "\n"
        ) % tuple(
            self.Rab[n, n, :] * xconv
        )

        print("self.max_l", self.max_l)
        if self.max_l >= 0:
            retstr += ("Nuclear charge:      " + fmt + "\n") % self.Z[n]
            retstr += ("Electronic charge:   " + fmt + "\n") % self.Qab[n, n]
            retstr += ("Total charge:        " + fmt + "\n") % (
                self.Z[n] + self.Qab[n, n]
            )
        if self.max_l >= 1:
            retstr += ("Electronic dipole    " + 3 * fmt + "\n") % tuple(
                self.Dab.sum(axis=2)[:, n]
            )

        if self.max_l >= 2:
            retstr += ("Electronic quadrupole" + 6 * fmt + "\n") % tuple(
                (self.QUab + self.dQUab).sum(axis=2)[:, n]
            )

        if self.pol == 1:
            Aab = self.Aab + 0.5 * self.dAab
            for iw, w in enumerate(self.freqs):
                retstr += ("Isotropic polarizablity (w=%g)" % w + fmt + "\n") % (
                    Aab.sum(axis=4)[iw, :, :, n].trace() / 3
                )

        if self.pol == 2:
            for iw, w in enumerate(self.freqs):
                a_lower = (
                    (self.Aab + 0.5 * self.dAab)
                    .sum(axis=4)[iw, :, :, n]
                    .view(full.matrix)
                    .pack()
                )
                retstr += (
                    "Electronic polarizability (w=%g)" % w + 6 * fmt + "\n"
                ) % tuple(a_lower)

        return retstr
