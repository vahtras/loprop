import pathlib

import pytest
import numpy as np

from loprop.core import penalty_function, AU2ANG
from loprop.dalton import MolFragDalton
# from loprop.veloxchem import MolFragVeloxChem
from util import full

from . import ch4_absorption_data as ref

thisdir = pathlib.Path(__file__).parent
case = "ch4_absorption"
tmpdir = thisdir/case/"tmp"


@pytest.fixture
def mf(request):
    cls = request.param
    return cls(
        tmpdir,
        freqs=(0.4425,),
        damping=0.004556,
        pf=penalty_function(2.0 / AU2ANG ** 2),
    )


@pytest.mark.parametrize(
    "mf",
    [MolFragDalton],  # , MolFragVeloxchem],
    ids=["dalton"],  # , 'veloxchem'],
    indirect=True,
)
class TestCH4Absorption:
    def test_nuclear_charge(self, mf):
        Z = mf.Z
        np.testing.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self, mf):
        R = mf.R
        np.testing.assert_allclose(R, ref.R)

    def test_default_gauge(self, mf):
        np.testing.assert_allclose(mf.Rc, ref.Rc)

    def test_total_charge(self, mf):
        Qtot = mf.Qab.sum()
        np.testing.assert_allclose(Qtot, ref.Qtot)

    def test_total_dipole(self, mf):
        np.testing.assert_allclose(mf.Dtot, ref.Dtot, atol=1e-6)

    def test_total_charge_shift_real(self, mf):
        mf.set_real_pol()
        RedQ = mf.dQa[0].sum(axis=0).view(full.matrix)
        np.testing.assert_allclose(RedQ, [0, 0, 0], atol=1e-5)

    def test_total_charge_shift_imag(self, mf):
        mf.set_imag_pol()
        ImdQ = mf.dQa[0].sum(axis=0).view(full.matrix)
        np.testing.assert_allclose(ImdQ, [0, 0, 0], atol=1e-5)

    def test_real_polarizability_total(self, mf):

        mf.set_real_pol()
        ref_Am = full.init(
            [
                [30.854533, -0.000004, 0.000000],
                [-0.000004, 30.854527, 0.000000],
                [0.000000, 0.000000, 30.854522],
            ]
        )
        Am = mf.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-6, atol=1e-6)

    def test_imag_polarizability_total(self, mf):

        mf.set_imag_pol()
        ref_Am = full.init(
            [
                [1.228334, 0.000000, 0.000000],
                [0.000000, 1.228335, 0.000000],
                [0.000000, 0.000000, 1.228333],
            ]
        )
        Am = mf.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-6, atol=1e-6)

    def test_capture_error_setup(self, mf):
        mf._real_pol = False
        mf._imag_pol = False
        with pytest.raises(ValueError):
            _ = mf.Dk
