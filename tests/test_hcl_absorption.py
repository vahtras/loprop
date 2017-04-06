import unittest
import os 
import numpy as np
from loprop.core import penalty_function, AU2ANG, pairs, MolFrag
from util import full

import re
thisdir  = os.path.dirname(__file__)
case = "hcl_absorption"
tmpdir=os.path.join(thisdir, case, 'tmp')
exec('from . import %s_data as ref'%case)


import unittest


class NewTest(unittest.TestCase):

    def setUp(self):
    # modify Gagliardi penalty function to include unit conversion bug
        self.m = MolFrag(tmpdir, freqs=(0.4425,), damping=0.004556, pf=penalty_function(2.0/AU2ANG**2))

    def tearDown(self):
        pass

    def test_nuclear_charge(self):
        Z = self.m.Z
        np.testing.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self):
        R = self.m.R
        np.testing.assert_allclose(R, ref.R)

    def test_default_gauge(self):
        np.testing.assert_allclose(ref.Rc, self.m.Rc)

    def test_total_charge(self):
        Qtot = self.m.Qab.sum()
        np.testing.assert_allclose(Qtot, ref.Qtot)

#   def test_total_dipole(self):
#       np.testing.assert_allclose(self.m.Dtot, ref.Dtot, atol=1e-6)

    def test_total_charge_shift_real(self):
        self.m.set_real_pol()
        RedQ = self.m.dQa[0].sum(axis=0).view(full.matrix)
        np.testing.assert_allclose(RedQ, [0,0,0], atol=1e-5)

    def test_total_charge_shift_imag(self):
        self.m.set_imag_pol()
        ImdQ = self.m.dQa[0].sum(axis=0).view(full.matrix)
        np.testing.assert_allclose(ImdQ, [0,0,0], atol=1e-5)

    def test_real_polarizability_total(self):

        self.m.set_real_pol()
        ref_Am = full.init([
            [7.626564, 0.000000, 0.000000],
            [0.000000, 7.626564, 0.000000],    
            [0.000000, 0.000000,42.786381]
            ])
        Am = self.m.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-5, atol=1e-5)

    def test_imag_polarizability_total(self):

        self.m.set_imag_pol()
        ref_Am = full.init([
            [0.063679, -0.000000,  0.000000],
            [-0.000004, 0.063679,  0.000000],    
            [ 0.000000,  0.000000, 2.767574]
            ])
        Am = self.m.Am[0]
        np.testing.assert_allclose(Am, ref_Am, rtol=1e-5, atol=1e-5)

    def test_capture_error_setup(self):
        self.m._real_pol = False
        self.m._imag_pol = False
        with self.assertRaises(ValueError):
            Dk = self.m.Dk

if __name__ == "__main__": #pragma: no cover
    unittest.main()
