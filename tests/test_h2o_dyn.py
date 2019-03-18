import unittest
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


class NewTest(LoPropTestCase):

    def setUp(self):
        self.m = MolFrag(tmpdir, freqs=(0., 0.15), pf=penalty_function(2.0/AU2ANG**2))
        self.maxDiff = None

    def tearDown(self):
        pass


    def test_nuclear_charge(self):
        Z = self.m.Z
        self.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self):
        R = self.m.R
        self.assert_allclose(R, ref.R)

    def test_default_gauge(self):
        self.assert_allclose(self.m.Rc, ref.Rc)

    def test_defined_gauge(self):
        m = MolFrag(tmpdir, gc=[1,2,3])
        self.assert_allclose(m.Rc, [1,2,3])

    def test_total_charge(self):
        Qtot = self.m.Qab.sum()
        self.assertAlmostEqual(Qtot, ref.Qtot)

    def test_charge(self):
        Qaa = self.m.Qa
        self.assert_allclose(ref.Q, Qaa)

    def test_total_dipole(self):
        self.assert_allclose(self.m.Dtot, ref.Dtot)

    def test_dipole_allbonds(self):
        D = full.matrix(ref.D.shape)
        Dab = self.m.Dab
        for ab, a, b in pairs(self.m.noa):
            D[:, ab] += Dab[:, a, b ] 
            if a != b: D[:, ab] += Dab[:, b, a] 
        self.assert_allclose(D, ref.D)

    def test_dipole_allbonds_sym(self):
        Dsym = self.m.Dsym
        self.assert_allclose(Dsym, ref.D)

    def test_dipole_nobonds(self):
        Daa = self.m.Dab.sum(axis=2).view(full.matrix)
        self.assert_allclose(Daa, ref.Daa)

    def test_quadrupole_total(self):
        QUc = self.m.QUc
        self.assert_allclose(QUc, ref.QUc)
    
    def test_nuclear_quadrupole(self):
        QUN = self.m.QUN
        self.assert_allclose(QUN, ref.QUN)

    def test_quadrupole_allbonds(self):
        QU = full.matrix(ref.QU.shape)
        QUab = self.m.QUab
        for ab, a, b in pairs(self.m.noa):
            QU[:, ab] += QUab[:, a, b ] 
            if a != b: QU[:, ab] += QUab[:, b, a] 
        self.assert_allclose(QU, ref.QU)

    def test_quadrupole_allbonds_sym(self):
        QUsym = self.m.QUsym
        self.assert_allclose(QUsym, ref.QU)

    def test_quadrupole_nobonds(self):
        self.assert_allclose(self.m.QUa, ref.QUaa)

    def test_Fab(self):
        Fab = self.m.Fab
        self.assert_allclose(Fab, ref.Fab)

    def test_molcas_shift(self):
        Fab = self.m.Fab
        Lab = Fab + self.m.sf(Fab)
        self.assert_allclose(Lab, ref.Lab)

    def test_total_charge_shift(self):
        dQ = self.m.dQa[0].sum(axis=0).view(full.matrix)
        dQref = [0., 0., 0.]
        self.assert_allclose(dQref, dQ)

    def test_polarizability_total(self):
        Am = self.m.Am[0]
        self.assert_allclose(Am, ref.Am, 0.015)

    def test_dyn_polarizability_total(self):
        Amw = self.m.Am[1]
        self.assert_allclose(Amw, ref.Amw, 0.015)
            
    def test_potfile_PAn00_angstrom(self):
        PAn0 = self.m.output_potential_file(maxl=-1, pol=0, hyper=0, angstrom=True)
        self.assert_str(PAn0, ref.POTFILE_BY_ATOM_n0_ANGSTROM)

    def test_potfile_PA00(self):
        PA00 = self.m.output_potential_file(maxl=0, pol=0, hyper=0)
        self.assert_str(PA00, ref.PA00)

    def test_potfile_PA10(self):
        PA10 = self.m.output_potential_file(maxl=1, pol=0, hyper=0)
        self.assert_str(PA10, ref.PA10)

    def test_potfile_PA20(self):
        PA20 = self.m.output_potential_file(maxl=2, pol=0, hyper=0)
        self.assert_str(PA20, ref.PA20)

    def test_potfile_PA21(self):
        PA21 = self.m.output_potential_file(maxl=2, pol=1, hyper=0)
        self.assert_str(PA21, ref.PA21)

    def test_potfile_PA22(self):
        PA22 = self.m.output_potential_file(maxl=2, pol=2, hyper=0)
        self.assert_str(PA22, ref.PA22)

    def test_potfile_PAn0b(self):
        PAn0b = self.m.output_potential_file(maxl=-1, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PAn0b, h2o_data.PAn0b)

    def test_potfile_PA00b(self):
        PA00b = self.m.output_potential_file(maxl=0, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PA00b, h2o_data.PA00b)

    def test_potfile_PA10b(self):
        PA10b = self.m.output_potential_file(maxl=1, pol=0, hyper=0, bond_centers=True)
        self.assert_str(PA10b, h2o_data.PA10b)

    def test_potfile_PA20b(self):
        with self.assertRaises(NotImplementedError):
            PA20b = self.m.output_potential_file(maxl=2, pol=0, hyper=0, bond_centers=True)

    def test_potfile_PA01b(self):
        PA01b = self.m.output_potential_file(maxl=0, pol=1, hyper=0, bond_centers=True)
        self.assert_str(PA01b, ref.PA01b)

    def test_potfile_PA02(self):
        this = self.m.output_potential_file(maxl=0, pol=2, hyper=0)
        self.assert_str(this, ref.PA02)

    def test_potfile_PA02b(self):
        this = self.m.output_potential_file(maxl=0, pol=2, hyper=0, bond_centers=True)
        self.assert_str(this, ref.PA02b)

    def test_outfile_PAn0_atom_domain(self):
        self.m.max_l = -1
        self.assert_str(self.m.print_atom_domain(0), ref.OUTPUT_n0_1)

    def test_outfile_PAn0_atom_domain_angstrom(self):
        self.m.max_l = -1
        self.assert_str(self.m.print_atom_domain(0, angstrom=True), h2o_data.OUTPUT_n0_1_ANGSTROM)

    def test_outfile_PA00_atom_domain(self):
        self.m.max_l = 0
        self.assert_str(self.m.print_atom_domain(0), ref.OUTPUT_00_1)

    def test_outfile_PA10_atom_domain(self):
        self.m.max_l = 1
        self.assert_str(self.m.print_atom_domain(0), ref.OUTPUT_10_1)

    def test_outfile_PA01_atom_domain(self):
        self.m.max_l = 0
        self.m.pol = 1
        self.assert_str(self.m.print_atom_domain(0), ref.OUTPUT_01_1)

    def test_outfile_PA02_atom_domain(self):
        self.m.max_l = 0
        self.m.pol = 2
        self.assert_str(self.m.print_atom_domain(0), ref.OUTPUT_02_1)

    def test_outfile_PA01_by_atom(self):
        self.m.max_l = 0
        self.m.output_by_atom(fmt="%12.5f", max_l=0, pol=1)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_ATOM_01)

    def test_outfile_PAn1_by_bond(self):
        self.m.max_l = -1
        self.m.output_by_atom(fmt="%12.5f", max_l=-1, pol=1, bond_centers=True)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_BOND_n1)

    def test_outfile_PAn2_by_bond(self):
        self.m.max_l = -1
        self.m.output_by_atom(fmt="%12.5f", max_l=-1, pol=2, bond_centers=True)
        print_output = self.capfd.readouterr().out.strip()
        self.assert_str(print_output, ref.OUTPUT_BY_BOND_n2)

