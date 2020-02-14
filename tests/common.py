import pytest
import numpy
import numpy.testing as npt
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

codes = ["dal", "vlx"]


class LoPropTestCase:  # (unittest.TestCase):
    @pytest.fixture(autouse=True)
    def capfd(self, capfd):
        self.capfd = capfd

    def assert_str(self, this, ref):
        def stripm0(numstr):
            # allow string inequality from round-off errors
            return numstr.replace("-0.000", " 0.000")

        thism0 = stripm0(this)
        refm0 = stripm0(ref)
        # self.assertEqual(thism0, refm0)
        assert thism0 == refm0

    def assert_allclose(self, *args, **kwargs):
        kwargs["atol"] = kwargs.get("atol", 1e-5)
        if "text" in kwargs:
            kwargs["err_msg"] = kwargs.get("text", "")
            del kwargs["text"]
        npt.assert_allclose(*(numpy.array(a) for a in args), **kwargs)

    @staticmethod
    def skip_if_not_implemented(method, code):
        if method not in dir(code):
            pytest.skip(
                f'Method {method} not implmented in {code.__class__.__name__}'
            )
