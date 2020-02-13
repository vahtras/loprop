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
        self.aooneint = os.path.join(tmpdir, 'h2o.integrals.h5')

    def get_basis_info(self):
        pass

    def get_isordk(self):
        pass

    @property
    def D(self):
        pass

    @property
    def D2k(self):
        pass

    def S(self):
        #
        # read overlap from hd5 file
        #
        with h5py.File(self.aooneint, 'r') as f:
            S = f['overlap'][...].view(Matrix)
        return S

    @property
    def Z(self):
        with h5py.File(self.hdf5, 'r') as f:
            _Z = f['nuc']
        return _Z
