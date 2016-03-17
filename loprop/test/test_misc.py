import unittest
import sys
import numpy
from ..core import output_beta, header
from ..daltools.util import full


class NewTest(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        self.beta = full.matrix((3, 6)).random()
        self.dipole = full.matrix(3).random()

    def tearDown(self):
        pass

    def test_header(self):
        if not hasattr(sys.stdout, "getvalue"):
            self.fail("Run in buffered mode")
        header('yo')
        print_output = sys.stdout.getvalue().strip()
        ref_output = """\
--
yo
--"""
        self.assertEqual(print_output, ref_output)

    def test_output_beta(self):
        if not hasattr(sys.stdout, "getvalue"):
            self.fail("Run in buffered mode")
        output_beta(self.beta)
        print_output = sys.stdout.getvalue().strip()
        ref_output="""\
Hyperpolarizability
beta(:, xx xy xz yy yz zz)
--------------------------
beta(x, *)     0.548814    0.715189    0.602763    0.544883    0.423655    0.645894
beta(y, *)     0.437587    0.891773    0.963663    0.383442    0.791725    0.528895
beta(z, *)     0.568045    0.925597    0.071036    0.087129    0.020218    0.832620
beta(:, kk)    1.739591    1.349924    1.487794"""
        self.assertEqual(print_output, ref_output)

    def test_output_beta_d(self):
        if not hasattr(sys.stdout, "getvalue"):
            self.fail("Run in buffered mode")
        output_beta(self.beta, self.dipole)
        print_output = sys.stdout.getvalue().strip()
        ref_output="""\
Hyperpolarizability
beta(:, xx xy xz yy yz zz)
--------------------------
beta(x, *)     0.548814    0.715189    0.602763    0.544883    0.423655    0.645894
beta(y, *)     0.437587    0.891773    0.963663    0.383442    0.791725    0.528895
beta(z, *)     0.568045    0.925597    0.071036    0.087129    0.020218    0.832620
beta(:, kk)    1.739591    1.349924    1.487794
beta//dip      0.523123"""
        self.assertEqual(print_output, ref_output)

if __name__ == "__main__":
    unittest.main()

