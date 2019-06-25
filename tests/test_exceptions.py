import pytest
import os
from loprop.core import MolFrag


class TestException:

    def setup(self):
        self.thisdir = os.path.dirname(__file__)
        self.tmpdir = os.path.join(self.thisdir, 'h2o_sym', 'tmp')

    def test_raise_exception_if_symmetry(self):
        with pytest.raises(AssertionError):
            MolFrag(self.tmpdir)
