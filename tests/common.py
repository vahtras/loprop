import unittest
import numpy
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import loprop

class LoPropTestCase(unittest.TestCase):

    def assert_str(self, this, ref):
        def stripm0(numstr):
            # allow string inequality from round-off errors
            return numstr.replace("-0.000", " 0.000")
        thism0 = stripm0(this)
        refm0 = stripm0(ref)
        self.assertEqual(thism0, refm0)


    def assert_allclose(self, *args, **kwargs):
        kwargs['atol'] = kwargs.get('atol', 1e-5)
        if 'text' in kwargs:
            kwargs['err_msg'] = kwargs.get('text', '')
            del(kwargs['text'])
        numpy.testing.assert_allclose(*args, **kwargs)
