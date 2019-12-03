from .core import MolFrag


class MolFragVeloxChem(MolFrag):

    def __init__(self, tmpdir, **kwargs):
        super().__init__(tmpdir, **kwargs)
        #
        # Veloxchem files
        #

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

    @property
    def S(self):
        #
        # read overlap from hd5 file
        #
        return self.vlx.get_overlap() typ
