import os

import h5py
from util.full import Matrix

from .core import MolFrag


class MolFragVeloxChem(MolFrag):

    dipole_labels = ('x', 'y', 'z')

    def __init__(self, tmpdir, **kwargs):
        super().__init__(tmpdir, **kwargs)
        #
        # Veloxchem files
        #
        self.interface = kwargs['checkpoint_file']
        self.scf = kwargs['scf_checkpoint_file']
        self._Z = None
        self._R = None
        self._Rab = None
        self.cpa = []
        self.opa = []
        self.noa = 0

        self.get_basis_info()
        self.get_molecule_info()

    def get_basis_info(self):
        """
        Obtain basis set info from checkpoint file
        """
        with h5py.File(self.interface, 'r') as f:
            self.cpa = [int(i) for i in f['contracted_per_atom'][...]]
            self.opa = [
                occ[...]
                for occ in f[f'occupied_per_atom'].values()
            ]
        self.noa = len(self.cpa)

    def get_molecule_info(self):
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
    def x(self):
        """
        Read dipole matrices to blocked loprop basis
        """

        if self._x is not None:
            return self._x

        self._x = self.get_dipole_matrices()
        return self._x

    def get_dipole_matrices(self):
        with h5py.File(self.interface, 'r') as f:
            Dx = f['ao_dipole_matrices/x'][...].view(Matrix)
            Dy = f['ao_dipole_matrices/y'][...].view(Matrix)
            Dz = f['ao_dipole_matrices/z'][...].view(Matrix)

        return tuple(self.ao_to_blocked_loprop(Dx, Dy, Dz))

    def get_quadrupole_matrices(self):
        with h5py.File(self.interface, 'r') as f:
            Qxx = f['ao_quadrupole_matrices/xx'][...].view(Matrix)
            Qxy = f['ao_quadrupole_matrices/xy'][...].view(Matrix)
            Qxz = f['ao_quadrupole_matrices/xz'][...].view(Matrix)
            Qyy = f['ao_quadrupole_matrices/yy'][...].view(Matrix)
            Qyz = f['ao_quadrupole_matrices/yz'][...].view(Matrix)
            Qzz = f['ao_quadrupole_matrices/zz'][...].view(Matrix)

        return tuple(
            self.ao_to_blocked_loprop(
                Qxx, Qxy, Qxz, Qyy, Qyz, Qzz
            )
        )

    @property
    def Dk(self):
        """
        Read perturbed ao density matrices from file
        """

        if self._Dk is not None:
            return self._Dk

        Dk = {}
        freq = 0.0
        with h5py.File(self.interface, 'r') as f:
            for c in 'x', 'y', 'z':
                Dk[(c, freq)] = f[f'ao_lr_density_matrix/{c}/{freq}'][...]
        self._Dk = self.contravariant_ao_to_blocked_loprop(Dk)

        return self._Dk

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
