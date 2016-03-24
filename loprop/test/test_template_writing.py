import unittest, os
from ..core import MolFrag, penalty_function, shift_function

DIR = "h2o_beta/tmp"

class TemplateTest( unittest.TestCase ):

    def setUp(self):
        self.tmp = os.path.join( os.path.dirname( __file__ ), DIR)
        self.mf = MolFrag( tmpdir = self.tmp, max_l =2, pol = 2,
                freqs = None,
                pf = penalty_function(2.0),
                sf = shift_function,
                gc = None)
        self.maxDiff = None

    def test_h2o_beta_dir(self):
        self.assertTrue(os.path.isdir(self.tmp))

    def test_h2o_template(self):

        reference = """( 'O1', "charge") : [ -0.703 ],
( 'H2', "charge") : [ 0.352 ],
( 'H3', "charge") : [ 0.352 ],
( 'O1', "dipole") : [ -0.000, -0.000, -0.284 ],
( 'H2', "dipole") : [ 0.153, -0.000, 0.127 ],
( 'H3', "dipole") : [ -0.153, -0.000, 0.127 ],
( 'O1', "quadrupole") : [ -3.293, -0.000, -0.000, -4.543, 0.000, -4.005 ],
( 'H2', "quadrupole") : [ -0.132, -0.000, 0.250, -0.445, -0.000, -0.261 ],
( 'H3', "quadrupole") : [ -0.132, 0.000, -0.250, -0.445, -0.000, -0.261 ],
( 'O1', "alpha") : [ 3.875, 0.000, -0.000, 3.000, -0.000, 3.524 ],
( 'H2', "alpha") : [ 2.156, -0.000, 1.106, 1.051, -0.000, 1.520 ],
( 'H3', "alpha") : [ 2.156, 0.000, -1.106, 1.051, -0.000, 1.520 ],
( 'O1', "beta") : [ 0.000, -0.000, 5.719, 0.000, 0.000, -0.000, -0.000, 0.103, -0.000, 0.038 ],
( 'H2', "beta") : [ 8.671, -0.000, 4.511, 2.783, -0.000, 4.409, -0.000, 1.590, -0.000, 4.585 ],
( 'H3', "beta") : [ -8.671, -0.000, 4.511, -2.783, 0.000, -4.409, -0.000, 1.590, -0.000, 4.585 ],
"""

        string =  self.mf.output_template(2, 2, 2, decimal =3) 
        # Normalize blanks,zeros
        string = string.replace("-0.000", " 0.000")
        string = " ".join(string.split())
        reference = reference.replace("-0.000", " 0.000")
        reference = " ".join(reference.split())
        
        self.assertEqual(string, reference)

    def test_h2o_template_wrong_l(self):
        with self.assertRaises(RuntimeError):
            str_ = self.mf.output_template(maxl=3)

    def test_h2o_template_wrong_a(self):
        with self.assertRaises(RuntimeError):
            str_ = self.mf.output_template(pol=3)

    def test_h2o_template_wrong_b(self):
        with self.assertRaises(RuntimeError):
            str_ = self.mf.output_template(hyper=3)

    def test_h2o_template_full(self):

        reference = """\
( 'O1', "charge") : [ -0.000 ],
( 'O1', "dipole") : [ -0.000, -0.000, -0.765 ],
( 'O1', "quadrupole") : [ -3.557, 0.000, -0.000, -5.432, 0.000, -4.526 ],
( 'O1', "alpha") : [ 8.187, 0.000, -0.000, 5.103, -0.000, 6.565 ],
( 'O1', "beta") : [ 0.000, -0.000, 14.741, 0.000, 0.000, -0.000, -0.000, 3.284, -0.000, 9.208 ],
"""

        string =  self.mf.output_template(2, 2, 2, decimal=3, template_full=True)
        # Normalize blanks,zeros
        string = string.replace("-0.000", " 0.000")
        string = " ".join(string.split())
        reference = reference.replace("-0.000", " 0.000")
        reference = " ".join(reference.split())
        
        print(string)
        self.assertEqual(string, reference)

if __name__ == '__main__':#pragma: no cover
    unittest.main()
