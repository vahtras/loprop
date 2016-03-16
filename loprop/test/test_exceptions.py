import unittest
import os
from ..daltools import sirifc
from ..core import MolFrag

class TestException(unittest.TestCase):

    def setUp(self):
        self.thisdir = os.path.dirname(__file__)
        self.tmpdir = os.path.join(self.thisdir, 'h2o_sym', 'tmp')

    def test_raise_exception_if_symmetry(self):
        nsym = sirifc.sirifc(name=os.path.join(self.tmpdir, "SIRIFC")).nsym
        self.assertRaises(AssertionError, MolFrag, self.tmpdir)
        
    


