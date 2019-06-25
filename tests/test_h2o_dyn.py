import pytest
from .common import loprop, LoPropTestCase
import os 
import sys
import numpy as np
from loprop.core import MolFrag
from util import full

import re
thisdir  = os.path.dirname(__file__)
case = "h2o_dyn"
tmpdir=os.path.join(thisdir, case, 'tmp')
exec('from . import %s_data as ref'%case)
from . import h2o_data

from loprop.core import penalty_function, AU2ANG, pairs


@pytest.fixture
def molfrag(request):
     cls = request.param
     return cls(tmpdir, freqs=(0., 0.15), pf=penalty_function(2.0/AU2ANG**2))

@pytest.mark.parametrize('molfrag', [MolFrag], ids=['dalton'], indirect=True)
class TestNew(LoPropTestCase):

    #def setup(self, molfrag):
    #    molfrag = MolFrag(tmpdir, freqs=(0., 0.15), pf=penalty_function(2.0/AU2ANG**2))
    #    molfragaxDiff = None

    #def teardown(self, molfrag):
    #    pass


    def test_nuclear_charge(self, molfrag):
        Z = molfrag.Z
        self.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self, molfrag):
        R = molfrag.R
        self.assert_allclose(R, ref.R)

    def test_default_gauge(self, molfrag):
        self.assert_allclose(molfrag.Rc, ref.Rc)

    def test_defined_gauge(self, molfrag):
        m = MolFrag(tmpdir, gc=[1,2,3])
        self.assert_allclose(m.Rc, [1,2,3])

    def test_total_charge(self, molfrag):
        Qtot = molfrag.Qab.sum()
        self.assert_allclose(Qtot, ref.Qtot)

    def test_charge(self, molfrag):
        Qaa = molfrag.Qa
        self.assert_allclose(ref.Q, Qaa)

    def test_total_dipole(self, molfrag):
        self.assert_allclose(molfrag.Dtot, ref.Dtot)

    def test_dipole_allbonds(self, molfrag):
        D = full.matrix(ref.D.shape)
        Dab = molfrag.Dab
        for ab, a, b in pairs(molfrag.noa):
            D[:, ab] += Dab[:, a, b ] 
            if a != b: D[:, ab] += Dab[:, b, a] 
        self.assert_allclose(D, ref.D)

    def test_dipole_allbonds_sym(self, molfrag):
        Dsym = molfrag.Dsym
        self.assert_allclose(Dsym, ref.D)

    def test_dipole_nobonds(self, molfrag):
        Daa = molfrag.Dab.sum(axis=2).view(full.matrix)
        self.assert_allclose(Daa, ref.Daa)

    def test_quadrupole_total(self, molfrag):
        QUc = molfrag.QUc
        self.assert_allclose(QUc, ref.QUc)
    
    def test_nuclear_quadrupole(self, molfrag):
        QUN = molfrag.QUN
        self.assert_allclose(QUN, ref.QUN)

    def test_quadrupole_allbonds(self, molfrag):
        QU = full.matrix(ref.QU.shape)
        QUab = molfrag.QUab
        for ab, a, b in pairs(molfrag.noa):
            QU[:, ab] += QUab[:, a, b ] 
            if a != b: QU[:, ab] += QUab[:, b, a] 
        self.assert_allclose(QU, ref.QU)

    def test_quadrupole_allbonds_sym(self, molfrag):
        QUsym = molfrag.QUsym
        self.assert_allclose(QUsym, ref.QU)

    def test_quadrupole_nobonds(self, molfrag):
        self.assert_allclose(molfrag.QUa, ref.QUaa)

    def test_Fab(self, molfrag):
        Fab = molfrag.Fab
        self.assert_allclose(Fab, ref.Fab)

    def test_molcas_shift(self, molfrag):
        Fab = molfrag.Fab
        Lab = Fab + molfrag.sf(Fab)
        self.assert_allclose(Lab, ref.Lab)

    def test_total_charge_shift(self, molfrag):
        dQ = molfrag.dQa[0].sum(axis=0).view(full.matrix)
        dQref = [0., 0., 0.]
        self.assert_allclose(dQref, dQ)

    def test_polarizability_total(self, molfrag):
        Am = molfrag.Am[0]
        self.assert_allclose(Am, ref.Am, 0.015)

    def test_dyn_polarizability_total(self, molfrag):
        Amw = molfrag.Am[1]
        self.assert_allclose(Amw, ref.Amw, 0.015)
            
    def test_potfile_PAn00_angstrom(self, molfrag):
        PAn0 = molfrag.output_potential_file(maxl=-1, pol=0, hyper=0, angstrom=True)
        self.assert_str(PAn0, ref.POTFILE_BY_ATOM_n0_ANGSTROM)

    def test_potfile_PA00(self, molfrag):
        PA00 = molfrag.output_potential_file(maxl=0, pol=0, hyper=0)
        self.assert_str(PA00, ref.PA00)

    def test_potfile_PA10(self, molfrag):
        PA10 = molfrag.output_potential_file(maxl=1, pol=0, hyper=0)
        self.assert_str(PA10, ref.PA10)

    def test_potfile_PA20(self, molfrag):
        PA20 = molfrag.output_potential_file(maxl=2, pol=0, hyper=0)
        self.assert_str(PA20, ref.PA20)

    def test_potfile_PA21(self, molfrag):
        PA21 = molfrag.output_potential_file(maxl=2, pol=1, hyper=0)
        self.assert_str(PA21, ref.PA21)

    def test_potfile_PA22(self, molfrag):
        PA22 = molfrag.output_potential_file(maxl=2, pol=2, hyper=0)
        self.assert_str(PA22, ref.PA22)

    def test_potfile_PAn0b(self, molfrag):
        PAn0b = molfrag.output_potential_file(maxl=-1, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PAn0b, h2o_data.PAn0b)

    def test_potfile_PA00b(self, molfrag):
        PA00b = molfrag.output_potential_file(maxl=0, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PA00b, h2o_data.PA00b)

    def test_potfile_PA10b(self, molfrag):
        PA10b = molfrag.output_potential_file(maxl=1, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PA10b, h2o_data.PA10b)

    def test_potfile_PA20b(self, molfrag):
        with pytest.raises(NotImplementedError):
            PA20b = molfrag.output_potential_file(maxl=2, pol=0, hyper=0, bond_centers=True)

    def test_potfile_PA01b(self, molfrag):
        PA01b = molfrag.output_potential_file(maxl=0, pol=1, hyper=0, bond_centers=True)
        self.assert_str(PA01b, ref.PA01b)

    def test_potfile_PA02(self, molfrag):
        this = molfrag.output_potential_file(maxl=0, pol=2, hyper=0)
        self.assert_str(this, ref.PA02)

    def test_potfile_PA02b(self, molfrag):
        this = molfrag.output_potential_file(maxl=0, pol=2, hyper=0, bond_centers=True)
        self.assert_str(this, ref.PA02b)

    def test_outfile_PAn0_atom_domain(self, molfrag):
        molfrag.max_l = -1
        self.assert_str(molfrag.print_atom_domain(0), ref.OUTPUT_n0_1)

    def test_outfile_PAn0_atom_domain_angstrom(self, molfrag):
        molfrag.max_l = -1
        self.assert_str(molfrag.print_atom_domain(0, angstrom=True), h2o_data.OUTPUT_n0_1_ANGSTROM)

    def test_outfile_PA00_atom_domain(self, molfrag):
        molfrag.max_l = 0
        self.assert_str(molfrag.print_atom_domain(0), ref.OUTPUT_00_1)

    def test_outfile_PA10_atom_domain(self, molfrag):
        molfrag.max_l = 1
        self.assert_str(molfrag.print_atom_domain(0), ref.OUTPUT_10_1)

    def test_outfile_PA01_atom_domain(self, molfrag):
        molfrag.max_l = 0
        molfrag.pol = 1
        self.assert_str(molfrag.print_atom_domain(0), ref.OUTPUT_01_1)

    def test_outfile_PA02_atom_domain(self, molfrag):
        molfrag.max_l = 0
        molfrag.pol = 2
        self.assert_str(molfrag.print_atom_domain(0), ref.OUTPUT_02_1)

    def test_outfile_PA01_by_atom(self, molfrag):
        molfrag.max_l = 0
        molfrag.output_by_atom(fmt="%12.5f", max_l=0, pol=1)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_ATOM_01)

    def test_outfile_PAn1_by_bond(self, molfrag):
        molfrag.max_l = -1
        molfrag.output_by_atom(fmt="%12.5f", max_l=-1, pol=1, bond_centers=True)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_BOND_n1)

    def test_outfile_PAn2_by_bond(self, molfrag):
        molfrag.max_l = -1
        molfrag.output_by_atom(fmt="%12.5f", max_l=-1, pol=2, bond_centers=True)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_BOND_n2)
