import io
import textwrap
import unittest

import pytest
import numpy

from loprop.core import output_beta, header
from .common import LoPropTestCase, codes


@pytest.mark.parametrize("code", codes)
class TestNew(LoPropTestCase):
    def setup_class(cls):
        numpy.random.seed(0)
        # cls.beta = full.matrix((3, 6)).random()
        cls.beta = numpy.random.random((3, 6))
        # cls.dipole = full.matrix(3).random()
        cls.dipole = numpy.random.random(3)

    def test_header_default(self, code):
        header("yo")
        print_output = self.capfd.readouterr().out
        ref_output = textwrap.dedent(
            """
            --
            yo
            --
            """
        )
        assert print_output == ref_output

    def test_header_output(self, code):
        print_output = io.StringIO()
        header("yo", output=print_output)
        print_output = print_output.getvalue()
        ref_output = textwrap.dedent(
            """
            --
            yo
            --
            """
        )
        assert print_output == ref_output

    def test_output_beta(self, code):
        output_beta(self.beta)
        print_output = self.capfd.readouterr().out.strip()
        ref_output = """\
Hyperpolarizability
beta(:, xx xy xz yy yz zz)
--------------------------
beta(x, *)     0.548814    0.715189    0.602763    0.544883    0.423655    0.645894
beta(y, *)     0.437587    0.891773    0.963663    0.383442    0.791725    0.528895
beta(z, *)     0.568045    0.925597    0.071036    0.087129    0.020218    0.832620
beta(:, kk)    1.739591    1.349924    1.487794"""
        # self.assertEqual(print_output, ref_output)
        assert print_output == ref_output

    def test_output_beta_d(self, code):
        output_beta(self.beta, self.dipole)
        print_output = self.capfd.readouterr().out.strip()
        ref_output = """\
Hyperpolarizability
beta(:, xx xy xz yy yz zz)
--------------------------
beta(x, *)     0.548814    0.715189    0.602763    0.544883    0.423655    0.645894
beta(y, *)     0.437587    0.891773    0.963663    0.383442    0.791725    0.528895
beta(z, *)     0.568045    0.925597    0.071036    0.087129    0.020218    0.832620
beta(:, kk)    1.739591    1.349924    1.487794
beta//dip      0.523123"""
        # self.assertEqual(print_output, ref_output)
        assert print_output == ref_output


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
