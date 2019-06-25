import pytest
import os
from pathlib import Path
from loprop.core import MolFrag


class TestException:

    def test_raise_exception_if_symmetry(self):
        self.thisdir = Path(__file__).parent
        self.tmpdir = self.thisdir/'h2o_sym'/'tmp'
        with pytest.raises(AssertionError):
            MolFrag(self.tmpdir)
