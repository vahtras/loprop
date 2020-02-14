import os

import h5py
from util.full import Matrix

from .core import MolFrag


class MolFragVeloxChem(MolFrag):

    def __init__(self, tmpdir, **kwargs):
        super().__init__(tmpdir, **kwargs)
        #
        # Veloxchem files
        #
        self.interface = os.path.join(tmpdir, 'h2o.loprop.h5')
        self.scf = os.path.join(tmpdir, 'h2o.scf.h5')
        self._Z = None
        self._R = None
        self._Rab = None
        self.cpa = []
        self.opa = []
        self.noa = 0

        self.get_basis_info()
        self.get_isordk()

    def get_basis_info(self):
        with h5py.File(self.interface, 'r') as f:
            self.cpa = [int(i) for i in f['contracted_per_atom'][...]]
            self.opa = [
                [int(i) for i in v]
                for v in f['occupied_per_atom'].values()
            ]
        self.noa = len(self.cpa)

    def get_isordk(self):
        noa = self.noa
        self.Rab = Matrix((noa, noa, 3))
        self.dRab = Matrix((noa, noa, 3))
        for a in range(noa):
            for b in range(noa):
                self.Rab[a, b, :] = (self.R[a, :] + self.R[b, :]) / 2
                self.dRab[a, b, :] = (self.R[a, :] - self.R[b, :]) / 2

    def get_density_matrix(self):
        with h5py.File(self.interface, 'r') as f:
            D = f['ao_density_matrix'][...]
        return D

    @property
    def D2k(self):
        pass

    def S(self):
        #
        # read overlap from hd5 file
        #
        with h5py.File(self.interface, 'r') as f:
            S = f['ao_overlap_matrix'][...].view(Matrix)
        return S

    @property
    def Z(self):
        if self._Z is None:
            with h5py.File(self.scf, 'r') as f:
                self._Z = f['nuclear_charges'][...]

        return self._Z

    @property
    def R(self):
        if self._R is None:
            with h5py.File(self.interface, 'r') as f:
                self._R = f['nuclear_coordinates'][...]

        return self._R
