import os

from daltools import one, mol, dens, prop, lr, qr, sirifc
import numpy as np

from .core import MolFrag


class MolFragDalton(MolFrag):

    dipole_labels = ('XDIPLEN', 'YDIPLEN', 'ZDIPLEN')

    def __init__(self, tmpdir, **kwargs):
        super().__init__(tmpdir, **kwargs)
        #
        # Dalton files
        #
        self.aooneint = os.path.join(tmpdir, "AOONEINT")
        self.dalton_bas = os.path.join(tmpdir, "DALTON.BAS")
        self.sirifc = os.path.join(tmpdir, "SIRIFC")
        assert sirifc.sirifc(name=self.sirifc).nsym == 1
        self.get_basis_info()
        self.get_molecule_info()

    def get_basis_info(self):
        """
        Obtain basis set info from DALTON.BAS
        """
        molecule = mol.readin(self.dalton_bas)
        self.cpa = mol.contracted_per_atom(molecule)
        self.cpa_l = mol.contracted_per_atom_l(molecule)
        self.opa = mol.occupied_per_atom(molecule)
        self.noa = len(self.opa)
        #
        # Total number of basis functions and occupied orbitals
        #
        self.nbf = sum(self.cpa)
        self.noc = 0
        for o in self.opa:
            self.noc += len(o)

    def get_molecule_info(self):
        """
     `   Get overlap, nuclear charges and coordinates from AOONEINT
        """
        #
        # Data from the ISORDK section in AOONEINT
        #
        isordk = one.readisordk(filename=self.aooneint)
        #
        # Number of nuclei
        #
        N = isordk["nucdep"]
        #
        # MXCENT , Fix dimension defined in nuclei.h
        #
        mxcent = len(isordk["chrn"])
        #
        # Nuclear charges
        #
        self.Z = np.zeros((N,))
        self.Z[:] = isordk["chrn"][:N]
        #
        # Nuclear coordinates
        #
        R = np.zeros((mxcent * 3,))
        R[:] = isordk["cooo"][:]
        self.R = R.reshape((mxcent, 3), order="F")[:N, :]
        #
        # Bond center matrix and half bond vector
        #
        noa = self.noa
        self.Rab = np.zeros((noa, noa, 3))
        self.dRab = np.zeros((noa, noa, 3))
        for a in range(noa):
            for b in range(noa):
                self.Rab[a, b, :] = (self.R[a, :] + self.R[b, :]) * .5
                self.dRab[a, b, :] = (self.R[a, :] - self.R[b, :]) * .5

    def S(self):
        """
        Get overlap, nuclear charges and coordinates from AOONEINT
        """
        S = one.read("OVERLAP", self.aooneint)
        return S.unpack().unblock()

    def get_density_matrix(self):
        D = sum(dens.Dab(filename=self.sirifc))
        return D

    @property
    def x(self):
        """Read dipole matrices to blocked loprop basis"""

        if self._x is not None:
            return self._x

        lab = ["XDIPLEN", "YDIPLEN", "ZDIPLEN"]

        self._x = self.getprop(*lab)
        return self._x

    def getprop(self, *args):
        """
        Read general property matrices to blocked loprop basis
        """

        T = self.T
        # cpa = self.cpa
        prp = os.path.join(self.tmpdir, "AOPROPER")

        return [
            (T.T @ p @ T)  # .subblocked(cpa, cpa)
            for p in prop.read(*args, filename=prp, unpack=True)
        ]

    def get_dipole_matrices(self):
        lab = (
            "XDIPLEN",
            "YDIPLEN",
            "ZDIPLEN",
        )
        return self.getprop(*lab)

    def get_quadrupole_matrices(self):
        lab = (
            "XXSECMOM",
            "XYSECMOM",
            "XZSECMOM",
            "YYSECMOM",
            "YZSECMOM",
            "ZZSECMOM"
        )
        xy = self.getprop(*lab)
        return xy

    @property
    def Dk(self):
        """Read perturbed densities"""

        if self._Dk is not None:
            return self._Dk

        lab = ["XDIPLEN", "YDIPLEN", "ZDIPLEN"]

        Dkao = lr.Dk(
            *lab,
            freqs=self.freqs,
            tmpdir=self.tmpdir,
            absorption=self.damping,
            lr_vecs=True
        )
        if self.damping:
            Re_Dkao, Im_Dkao = Dkao
            if self.real_pol:
                _Dk = self.contravariant_ao_to_loprop(Re_Dkao)
            elif self.imag_pol:
                _Dk = self.contravariant_ao_to_loprop(Im_Dkao)
            else:
                raise ValueError
        else:
            _Dk = self.contravariant_ao_to_loprop(Dkao)

        self._Dk = _Dk
        return self._Dk

    @property
    def D2k(self):
        """Read perturbed densities"""

        if self._D2k is not None:
            return self._D2k

        lab = ["XDIPLEN ", "YDIPLEN ", "ZDIPLEN "]
        qrlab = [lab[j] + lab[i] for i in range(3) for j in range(i, 3)]
        Ti = self.Ti

        Dkao = qr.D2k(*qrlab, freqs=self.freqs, tmpdir=self.tmpdir)
        # print("Dkao.keys", Dkao.keys())
        _D2k = {
            lw: (Ti @ Dkao[lw] @ Ti.T) for lw in Dkao
        }

        self._D2k = _D2k
        return self._D2k

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
        Bab = np.zeros((self.nfreqs, 3, 6, self.noa, self.noa))

        # correction term for shifting origin from O to Rab
        for i, li in enumerate(labs):
            for jk, ljk in enumerate(qlabs):
                for a in range(self.noa):
                    for b in range(self.noa):
                        ab = self.extract_indices(a, b)
                        for iw, w in enumerate(self.freqs):
                            Bab[iw, i, jk, a, b] = (
                                -np.einsum('ij,ij', x[i][ab], D2k[(ljk, w, w)][ab])
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
        dBab = np.zeros((self.nfreqs, 3, 6, self.noa, self.noa))
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
        self._Bm = (Bab + 0.5 * dBab).sum(axis=4).sum(axis=3)

        return self._Bm
