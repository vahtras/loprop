from .common import loprop, LoPropTestCase
import unittest, os
from loprop.core import MolFrag, penalty_function, shift_function

case = "pff"
DIR = os.path.join(case, 'tmp')
exec('from . import %s_data as ref'%case)

class TestSulphur(LoPropTestCase):

    def setup(self):
        self.tmp = os.path.join( os.path.dirname( __file__ ), DIR)
        self.mf = MolFrag( tmpdir = self.tmp, max_l =0, pol = 0,
                freqs = None,
                pf = penalty_function(2.0),
                sf = shift_function,
                gc = None)
        self.maxDiff = None

    def test_dir(self):
        assert  os.path.isdir(self.tmp)

    def test_nuclear_charge(self):
        Z = self.mf.Z
        self.assert_allclose(Z, ref.Z)

if __name__ == '__main__':#pragma: no cover
    unittest.main()
