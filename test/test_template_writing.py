import unittest, os
from ..loprop import MolFrag, penalty_function, shift_function

DIR = "/home/ignat/repos/loprop_fix/test/h2o_beta/tmp"

class TemplateTest( unittest.TestCase ):

    def setUp(self):
        self.tmp = DIR

    def test_h2o_beta_dir(self):
        assert os.path.isdir( self.tmp )

    def test_h2o_template(self):
        assert MolFrag( tmpdir = self.tmp, max_l =2, pol = 2,
                freqs = None,
                pf = penalty_function(2.0),
                sf = shift_function,
                gc = None).output_template(2, 2, 2) == \
"""["charge"] : [ -0.703 ],
["charge"] : [ 0.352 ],
["charge"] : [ 0.352 ],
["dipole"] : [ -0.000, -0.000, -0.284 ],
["dipole"] : [ 0.153, -0.000, 0.127 ],
["dipole"] : [ -0.153, -0.000, 0.127 ],
["quadrupole"] : [ -3.293, -0.000, -0.000, -4.543, 0.000, -4.005 ],
["quadrupole"] : [ -0.132, -0.000, 0.250, -0.445, -0.000, -0.261 ],
["quadrupole"] : [ -0.132, 0.000, -0.250, -0.445, -0.000, -0.261 ],
["alpha"] : [ 3.875, 0.000, -0.000, 3.000, -0.000, 3.524 ],
["alpha"] : [ 2.156, -0.000, 1.106, 1.051, -0.000, 1.520 ],
["alpha"] : [ 2.156, 0.000, -1.106, 1.051, -0.000, 1.520 ],
["beta"] : [ 0.000, -0.000, 5.719, 0.000, 0.000, -0.000, -0.000, 0.103, -0.000, 0.038 ],
["beta"] : [ 8.671, -0.000, 4.511, 2.783, -0.000, 4.409, -0.000, 1.590, -0.000, 4.585 ],
["beta"] : [ -8.671, -0.000, 4.511, -2.783, 0.000, -4.409, -0.000, 1.590, -0.000, 4.585 ],
"""

if __name__ == '__main__':
    unittest.main()
