import pathlib

import numpy as np
import pytest

from loprop.core import penalty_function, AU2ANG
from loprop.dalton import MolFragDalton

from . import hcl_absorption_data as ref

thisdir = pathlib.Path(__file__).parent
case = "hcl_absorption"
tmpdir = thisdir / case / "tmp"


@pytest.fixture
def molfrag(request):
    cls = request.param
    return cls(
        tmpdir,
        freqs=(0.4425,),
        damping=0.004556,
        pf=penalty_function(2.0 / AU2ANG ** 2),
    )


@pytest.mark.parametrize("molfrag", [MolFragDalton], ids=["dalton"], indirect=True)
class Test:

    # def setup(self):
    # modify Gagliardi penalty function to include unit conversion bug
    #    molfrag = MolFrag(tmpdir, freqs=(0.4425,), damping=0.004556, pf=penalty_function(2.0/AU2ANG**2))

    # def tearDown(self):
    #    pass

    def test_nuclear_charge(self, molfrag):
        Z = molfrag.Z
        np.testing.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self, molfrag):
        R = molfrag.R
        np.testing.assert_allclose(R, ref.R)

    def test_default_gauge(self, molfrag):
        np.testing.assert_allclose(ref.Rc, molfrag.Rc)

    def test_total_charge(self, molfrag):
        Qtot = molfrag.Qab.sum()
        np.testing.assert_allclose(Qtot, ref.Qtot)

    #   def test_total_dipole(self, molfrag):
    #       np.testing.assert_allclose(molfrag.Dtot, ref.Dtot, atol=1e-6)

    def test_total_charge_shift_real(self, molfrag):
        molfrag.set_real_pol()
        RedQ = molfrag.dQa[0].sum(axis=0)
        np.testing.assert_allclose(RedQ, [0, 0, 0], atol=1e-5)

    def test_total_charge_shift_imag(self, molfrag):
        molfrag.set_imag_pol()
        ImdQ = molfrag.dQa[0].sum(axis=0)
        np.testing.assert_allclose(ImdQ, [0, 0, 0], atol=1e-5)

    def test_real_polarizability_total(self, molfrag):

        molfrag.set_real_pol()
        ref_Am = np.array(
            [
                [7.626564, 0.000000, 0.000000],
                [0.000000, 7.626564, 0.000000],
                [0.000000, 0.000000, 42.786381],
            ]
        ).T
        Am = molfrag.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-5, atol=1e-5)

    def test_imag_polarizability_total(self, molfrag):

        molfrag.set_imag_pol()
        ref_Am = np.array(
            [
                [0.063679, -0.000000, 0.000000],
                [-0.000004, 0.063679, 0.000000],
                [0.000000, 0.000000, 2.767574],
            ]
        ).T
        Am = molfrag.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-5, atol=1e-5)

    def test_capture_error_setup(self, molfrag):
        molfrag._real_pol = False
        molfrag._imag_pol = False
        with pytest.raises(ValueError):
            _ = molfrag.Dk
