import os

import pytest

from loprop.core import penalty_function, shift_function
from loprop.dalton import MolFragDalton

from . import pff_data as ref
from .common import LoPropTestCase

case = "pff"
DIR = os.path.join(case, "tmp")


@pytest.fixture
def molfrag(request):
    tmp = os.path.join(os.path.dirname(__file__), DIR)
    cls = request.param
    return cls(
        tmpdir=tmp,
        max_l=0,
        pol=0,
        freqs=None,
        pf=penalty_function(2.0),
        sf=shift_function,
        gc=None,
    )


@pytest.mark.parametrize("molfrag", [MolFragDalton], ids=["dalton"], indirect=True)
class TestSulphur(LoPropTestCase):
    def test_dir(self, molfrag):
        assert os.path.isdir(molfrag.tmpdir)

    def test_nuclear_charge(self, molfrag):
        Z = molfrag.Z
        self.assert_allclose(Z, ref.Z)
