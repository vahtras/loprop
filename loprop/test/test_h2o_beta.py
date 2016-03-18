from .common import LoPropTestCase
import os 
import sys
import numpy as np
from ..daltools.util import full

import re
thisdir  = os.path.dirname(__file__)
case = "h2o_beta"
tmpdir=os.path.join(thisdir, case, 'tmp')
exec('from . import %s_data as ref'%case)

from ..core import penalty_function, xtang, pairs, MolFrag

class H2OBetaTest(LoPropTestCase):

    def setUp(self):
        self.m = MolFrag(tmpdir, freqs=(0.0,), pf=penalty_function(2.0/xtang**2))
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
        self.assert_allclose(ref.Rc, self.m.Rc)

    def test_total_charge(self):
        Qtot = self.m.Qab.sum()
        self.assert_allclose(Qtot, ref.Qtot)

    def test_charge(self):
        Qaa = self.m.Qab.diagonal()
        self.assert_allclose(ref.Q, Qaa)

    def test_total_dipole(self):
        # molecular dipole moment wrt gauge center gc
        Dtot = self.m.Dab.sum(axis=2).sum(axis=1).view(full.matrix)
        Qa = self.m.Qab.diagonal()
        Q = Qa.sum()
        Dtot += Qa*self.m.R - Q*self.m.Rc
        self.assert_allclose(Dtot, ref.Dtot)

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

    def test_total_charge_shift2(self):
        d2Q = self.m.d2Qa[0].sum(axis=0).view(full.matrix)
        d2Qref = [0., 0., 0., 0., 0., 0.]
        self.assert_allclose(d2Qref, d2Q)

    def test_atomic_charge_shift(self):
        dQa = self.m.dQa[0]
        dQaref = (ref.dQa[:, 1::2] - ref.dQa[:, 2::2])/(2*ref.ff)
        self.assert_allclose(dQa, dQaref, atol=.006)

    def test_lagrangian(self):
    # values per "perturbation" as in atomic_charge_shift below
        la = self.m.la[0]
        laref = (ref.la[:,0:6:2] - ref.la[:,1:6:2])/(2*ref.ff)
    # The sign difference is because mocas sets up rhs with opposite sign
        self.assert_allclose(-laref, la, atol=100)

    def test_bond_charge_shift(self):
        dQab = self.m.dQab[0]
        noa = self.m.noa

        dQabref = (ref.dQab[:, 1:7:2] - ref.dQab[:, 2:7:2])/(2*ref.ff)
        dQabcmp = full.matrix((3, 3))
        ab = 0
        for a in range(noa):
            for b in range(a):
                dQabcmp[ab, :] = dQab[a, b, :]
                ab += 1
    # The sign difference is because mocas sets up rhs with opposite sign
        self.assert_allclose(-dQabref, dQabcmp, atol=0.006)

    def test_bond_charge_shift_sum(self):
        dQa  = self.m.dQab[0].sum(axis=1).view(full.matrix)
        dQaref = self.m.dQa[0]
        self.assert_allclose(dQa, dQaref)

    def test_bond_charge_shift_sum2(self):
        d2Qa  = self.m.d2Qab[0].sum(axis=1).view(full.matrix)
        d2Qaref = self.m.d2Qa[0]
        self.assert_allclose(d2Qa, d2Qaref)


    def test_polarizability_total(self):
        Am = self.m.Am[0]
        self.assert_allclose(Am, ref.Am, atol=0.015)

    def test_beta_zxx(self):
        r = self.m.x
        D2k = self.m.D2k
        z = r[2].unblock()
        xx = D2k[('XDIPLEN XDIPLEN ', 0.0, 0.0)].unblock()
        bzxx = - z&xx
        self.assert_allclose(bzxx, ref.Bm[2, 0], atol=.005)

    def test_beta_xzx(self):
        r = self.m.x
        D2k = self.m.D2k
        x = r[0].unblock()
        zx = D2k[('ZDIPLEN XDIPLEN ', 0.0, 0.0)].unblock()
        bxzx = - x&zx
        self.assert_allclose(bxzx, ref.Bm[0, 2], atol=.005)

    def test_beta_xxz(self):
        r = self.m.x
        D2k = self.m.D2k
        x = r[0].unblock()
        xz = D2k[('XDIPLEN ZDIPLEN ', 0.0, 0.0)].unblock()
        bxxz = - x&xz
        self.assert_allclose(bxxz, ref.Bm[0, 2], atol=.005)

    def test_beta_yyz(self):
        r = self.m.x
        D2k = self.m.D2k
        y = r[1].unblock()
        yz = D2k[('YDIPLEN ZDIPLEN ', 0.0, 0.0)].unblock()
        byyz = - y&yz
        self.assert_allclose(byyz, ref.Bm[1, 4], atol=.005)

    def test_beta_zyy(self):
        r = self.m.x
        D2k = self.m.D2k
        z = r[2].unblock()
        yy = D2k[('YDIPLEN YDIPLEN ', 0.0, 0.0)].unblock()
        bzyy = - z&yy
        self.assert_allclose(bzyy, ref.Bm[2, 3], atol=.005)

    def test_beta_zzz(self):
        r = self.m.x
        D2k = self.m.D2k
        z = r[2].unblock()
        zz = D2k[('ZDIPLEN ZDIPLEN ', 0.0, 0.0)].unblock()
        bzzz = - z&zz
        self.assert_allclose(bzzz, ref.Bm[2, 5], atol=.005)

    def test_hyperpolarizability_total(self):
        Bm = self.m.Bm[0]
        Bab = self.m.Bab.sum(axis=4).sum(axis=3)
        ref.Bm
        self.assert_allclose(Bm, ref.Bm, atol=.005)

    def notest_dynamic_polarizability_total(self):
        Amw = self.m.Am[1]
        assert_(Amw, ref.Amw, 0.015)
        
    def test_polarizability_allbonds_molcas_internal(self):
        O = ref.O
        H1O = ref.H1O
        H1 = ref.H1
        H2O = ref.H2O
        H2H1 = ref.H2H1
        H2 = ref.H2
        rMP = ref.rMP

        RO, RH1, RH2 = self.m.R
        ROx, ROy, ROz = RO
        RH1x, RH1y, RH1z = RH1
        RH2x, RH2y, RH2z = RH2
        

        ihff = 1/(2*ref.ff)

        q, x, y, z = range(4)
        dx1, dx2, dy1, dy2, dz1, dz2 = 1, 2, 3, 4, 5, 6
        o, h1o, h1, h2o, h2h1, h2 = range(6)

        Oxx = ihff*(rMP[x, dx1, o] - rMP[x, dx2, o])
        Oyx = ihff*(rMP[y, dx1, o] - rMP[y, dx2, o]
            +       rMP[x, dy1, o] - rMP[x, dy2, o])/2
        Oyy = ihff*(rMP[y, dy1, o] - rMP[y, dy2, o])
        Ozx = ihff*(rMP[z, dx1, o] - rMP[z, dx2, o]
            +       rMP[x, dz1, o] - rMP[x, dz2, o])/2
        Ozy = ihff*(rMP[z, dy1, o] - rMP[z, dy2, o]
            +       rMP[y, dz1, o] - rMP[y, dz2, o])/2
        Ozz = ihff*(rMP[z, dz1, o] - rMP[z, dz2, o])
        H1Oxx = ihff*(rMP[x, dx1, h1o] - rMP[x, dx2, h1o] \
              - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o])*(RH1x-ROx))
        H1Oyx = ihff*(
             (rMP[y, dx1, h1o] - rMP[y, dx2, h1o] 
            + rMP[x, dy1, h1o] - rMP[x, dy2, h1o])/2
           - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o])*(RH1y-ROy)
    #      - (rMP[0, dy1, h1o] - rMP[0, dy2, h1o])*(RH1x-ROx) THIS IS REALLY... A BUG?
           )
        H1Oyy = ihff*(rMP[y, dy1, h1o] - rMP[y, dy2, h1o] - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o])*(RH1y-ROy))
        H1Ozx = ihff*(
            (rMP[z, dx1, h1o] - rMP[z, dx2, h1o]
           + rMP[x, dz1, h1o] - rMP[x, dz2, h1o])/2
          - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o])*(RH1z-ROz)
    #             - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH1x-ROx) #THIS IS REALLY... A BUG?
                )
        H1Ozy = ihff*(
            (rMP[z, dy1, h1o] - rMP[z, dy2, h1o]
           + rMP[y, dz1, h1o] - rMP[y, dz2, h1o])/2
          - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o])*(RH1z-ROz)
    #     - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH1y-ROy) THIS IS REALLY... A BUG?
            )
        H1Ozz = ihff*(rMP[z, dz1, h1o] - rMP[z, dz2, h1o] - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH1z-ROz))
        H1xx = ihff*(rMP[x, dx1, h1] - rMP[x, dx2, h1])
        H1yx = (ihff*(rMP[y, dx1, h1] - rMP[y, dx2, h1])
             +  ihff*(rMP[x, dy1, h1] - rMP[x, dy2, h1]))/2
        H1yy = ihff*(rMP[y, dy1, h1] - rMP[y, dy2, h1])
        H1zx = (ihff*(rMP[z, dx1, h1] - rMP[z, dx2, h1])
             +  ihff*(rMP[x, dz1, h1] - rMP[x, dz2, h1]))/2
        H1zy = (ihff*(rMP[z, dy1, h1] - rMP[z, dy2, h1])
             +  ihff*(rMP[y, dz1, h1] - rMP[y, dz2, h1]))/2
        H1zz = ihff*(rMP[z, dz1, h1] - rMP[z, dz2, h1])
        H2Oxx = ihff*(rMP[x, dx1, h2o] - rMP[x, dx2, h2o] - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o])*(RH2x-ROx))
        H2Oyx = ihff*(
            (rMP[y, dx1, h2o] - rMP[y, dx2, h2o] 
           + rMP[x, dy1, h2o] - rMP[x, dy2, h2o])/2
           - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o])*(RH2y-ROy)
    #      - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o])*(RH2x-ROx) THIS IS REALLY... A BUG?
           )
        H2Oyy = ihff*(rMP[y, dy1, h2o] - rMP[y, dy2, h2o] - (rMP[q, dy1, h2o] - rMP[q, dy2, h2o])*(RH2y-ROy))
        H2Ozx = ihff*(
            (rMP[z, dx1, h2o] - rMP[z, dx2, h2o]
           + rMP[x, dz1, h2o] - rMP[x, dz2, h2o])/2
          - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o])*(RH2z-ROz)
    #             - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH2x-ROx) #THIS IS REALLY... A BUG?
                )
        H2Ozy = ihff*(
            (rMP[z, dy1, h2o] - rMP[z, dy2, h2o]
           + rMP[y, dz1, h2o] - rMP[y, dz2, h2o])/2
          - (rMP[q, dy1, h2o] - rMP[q, dy2, h2o])*(RH2z-ROz)
    #     - (rMP[q, dz1, h2o] - rMP[q, dz2, h2o])*(RH2y-ROy) THIS IS REALLY... A BUG?
            )
        H2Ozz = ihff*(rMP[z, dz1, h2o] - rMP[z, dz2, h2o] - (rMP[q, dz1, h2o] - rMP[q, dz2, h2o])*(RH2z-ROz))
        H2H1xx = ihff*(rMP[x, dx1, h2h1] - rMP[x, dx2, h2h1] - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1])*(RH2x-RH1x))
        H2H1yx = ihff*(
            (rMP[y, dx1, h2h1] - rMP[y, dx2, h2h1] 
           + rMP[x, dy1, h2h1] - rMP[x, dy2, h2h1])/2
           - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1])*(RH1y-ROy)
    #      - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1])*(RH1x-ROx) THIS IS REALLY... A BUG?
           )
        H2H1yy = ihff*(rMP[y, dy1, h2h1] - rMP[y, dy2, h2h1] - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1])*(RH2y-RH1y))
        H2H1zx = ihff*(
            (rMP[z, dx1, h2h1] - rMP[z, dx2, h2h1]
           + rMP[x, dz1, h2h1] - rMP[x, dz2, h2h1])/2
          - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1])*(RH1z-ROz)
    #     - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1])*(RH1x-ROx) #THIS IS REALLY... A BUG?
                )
        H2H1zy = ihff*(
            (rMP[z, dy1, h2h1] - rMP[z, dy2, h2h1]
           + rMP[y, dz1, h2h1] - rMP[y, dz2, h2h1])/2
          - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1])*(RH1z-ROz)
    #     - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1])*(RH1y-RO[1]) THIS IS REALLY... A BUG?
            )
        H2H1zz = ihff*(rMP[z, dz1, h2h1] - rMP[z, dz2, h2h1] - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1])*(RH2z-RH1z))
        H2xx = ihff*(rMP[x, dx1, h2] - rMP[x, dx2, h2])
        H2yx = (ihff*(rMP[y, dx1, h2] - rMP[y, dx2, h2])
             +  ihff*(rMP[x, dy1, h2] - rMP[x, dy2, h2]))/2
        H2yy = ihff*(rMP[y, dy1, h2] - rMP[y, dy2, h2])
        H2zx = (ihff*(rMP[z, dx1, h2] - rMP[z, dx2, h2])
             +  ihff*(rMP[x, dz1, h2] - rMP[x, dz2, h2]))/2
        H2zy = (ihff*(rMP[z, dy1, h2] - rMP[z, dy2, h2])
             +  ihff*(rMP[y, dz1, h2] - rMP[y, dz2, h2]))/2
        H2zz = ihff*(rMP[z, dz1, h2] - rMP[z, dz2, h2])

        comp = ("XX", "yx", "yy", "zx", "zy", "zz")
        bond = ("O", "H1O", "H1", "H2O", "H2H1", "H2")

        self.assert_allclose(O[0], Oxx, text="Oxx")
        self.assert_allclose(O[1], Oyx, text="Oyx")
        self.assert_allclose(O[2], Oyy, text="Oyy")
        self.assert_allclose(O[3], Ozx, text="Ozx")
        self.assert_allclose(O[4], Ozy, text="Ozy")
        self.assert_allclose(O[5], Ozz, text="Ozz")
        self.assert_allclose(H1O[0], H1Oxx, text="H1Oxx")
        self.assert_allclose(H1O[1], H1Oyx, text="H1Oyx")
        self.assert_allclose(H1O[2], H1Oyy, text="H1Oyy")
        self.assert_allclose(H1O[3], H1Ozx, text="H1Ozx")
        self.assert_allclose(H1O[4], H1Ozy, text="H1Ozy")
        self.assert_allclose(H1O[5], H1Ozz, text="H1Ozz")
        self.assert_allclose(H1[0], H1xx, text="H1xx")
        self.assert_allclose(H1[1], H1yx, text="H1yx")
        self.assert_allclose(H1[2], H1yy, text="H1yy")
        self.assert_allclose(H1[3], H1zx, text="H1zx")
        self.assert_allclose(H1[4], H1zy, text="H1zy")
        self.assert_allclose(H1[5], H1zz, text="H1zz")
        self.assert_allclose(H2O[0], H2Oxx, text="H2Oxx")
        self.assert_allclose(H2O[1], H2Oyx, text="H2Oyx")
        self.assert_allclose(H2O[2], H2Oyy, text="H2Oyy")
        self.assert_allclose(H2O[3], H2Ozx, text="H2Ozx")
        self.assert_allclose(H2O[4], H2Ozy, text="H2Ozy")
        self.assert_allclose(H2O[5], H2Ozz, text="H2Ozz")
        self.assert_allclose(H2H1[0], H2H1xx, text="H2H1xx")
        self.assert_allclose(H2H1[1], H2H1yx, text="H2H1yx")
        self.assert_allclose(H2H1[2], H2H1yy, text="H2H1yy")
        self.assert_allclose(H2H1[3], H2H1zx, text="H2H1zx")
        self.assert_allclose(H2H1[4], H2H1zy, text="H2H1zy")
        self.assert_allclose(H2H1[5], H2H1zz, text="H2H1zz")
        self.assert_allclose(H2[0], H2xx, text="H2xx")
        self.assert_allclose(H2[1], H2yx, text="H2yx")
        self.assert_allclose(H2[2], H2yy, text="H2yy")
        self.assert_allclose(H2[3], H2zx, text="H2zx")
        self.assert_allclose(H2[4], H2zy, text="H2zy")
        self.assert_allclose(H2[5], H2zz, text="H2zz")

    def test_altint(self):
        R = self.m.R
        rMP = ref.rMP
        diff = [(1, 2), (3, 4), (5, 6)]
        atoms = (0, 2, 5) 
        bonds = (1, 3, 4)
        ablab = ("O", "H1O", "H1", "H2O", "H2H1", "H2")
        ijlab = ("xx", "yx", "yy", "zx", "zy", "zz")

        pol = np.zeros((6, self.m.noa*(self.m.noa+1)//2))
        for ab, a, b in pairs(self.m.noa):
            for ij, i, j in pairs(3):
                #from pdb import set_trace; set_trace()
                i1, i2 = diff[i]
                j1, j2 = diff[j]
                pol[ij, ab] += (rMP[i+1, j1, ab] - rMP[i+1, j2, ab]
                            +   rMP[j+1, i1, ab] - rMP[j+1, i2, ab])/(4*ref.ff)
                if ab in bonds:
                    pol[ij, ab] -= (R[a][i]-R[b][i])*(rMP[0, j1, ab] - rMP[0, j2, ab])/(2*ref.ff)
                self.assert_allclose(ref.Aab[ij, ab], pol[ij, ab], text="%s%s"%(ablab[ab], ijlab[ij]))

    def test_polarizability_allbonds_atoms(self):

        Aab = self.m.Aab[0] #+ m.dAab
        noa = self.m.noa

        Acmp=full.matrix(ref.Aab.shape)
        
        ab = 0
        for a in range(noa):
            for b in range(a):
                Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
                ab += 1
            Acmp[:, ab] = Aab[:, :, a, a].pack()
            ab += 1
        # atoms
        self.assert_allclose(ref.Aab[:, 0], Acmp[:, 0], atol=.005)
        self.assert_allclose(ref.Aab[:, 2], Acmp[:, 2], atol=.005)
        self.assert_allclose(ref.Aab[:, 5], Acmp[:, 5], atol=.005)

    def test_polarizability_allbonds_bonds(self):

        Aab = self.m.Aab[0] + self.m.dAab[0]/2
        noa = self.m.noa

        Acmp=full.matrix(ref.Aab.shape)
        
        ab = 0
        for a in range(noa):
            for b in range(a):
                Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
                ab += 1
            Acmp[:, ab] = Aab[:, :, a, a].pack()
            ab += 1
        # atoms
        self.assert_allclose(ref.Aab[:, 1], Acmp[:, 1], atol=.150, err_msg='H1O')
        self.assert_allclose(ref.Aab[:, 3], Acmp[:, 3], atol=.150, err_msg='H2O')
        self.assert_allclose(ref.Aab[:, 4], Acmp[:, 4], atol=.005, err_msg='H2H1')
    

    def test_polarizability_nobonds(self):

        Aab = self.m.Aab[0] + self.m.dAab[0]/2
        noa = self.m.noa

        Acmp = full.matrix((6, noa ))
        Aa = Aab.sum(axis=3).view(full.matrix)
        
        ab = 0
        for a in range(noa):
            Acmp[:, a,] = Aa[:, :, a].pack()

        # atoms
        self.assert_allclose(Acmp, ref.Aa, atol=0.07)

    def test_potfile_PAn0(self):
        PAn0 = self.m.output_potential_file(maxl=-1, pol=0, hyper=0)
        self.assert_str(PAn0, ref.PAn0)

    def test_potfile_PA00(self):
        PA00 = self.m.output_potential_file(maxl=0, pol=0, hyper=0)
        self.assert_str(PA00, ref.PA00)

    def test_potfile_P0A0B1(self):
        this = self.m.output_potential_file(maxl=0, pol=0, hyper=1)
        self.assert_str(this, ref.P0A0B1)

    def test_potfile_P0A0B1b(self):
        this = self.m.output_potential_file(maxl=0, pol=0, hyper=1, bond_centers=True)
        self.assert_str(this, ref.P0A0B1b)

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

    def test_outfile_PAn0_by_atom(self):
        self.m.max_l = -1
        Da = self.m.Da #use for beta internally and will be set in output
        self.m.output_by_atom(fmt="%12.5f", hyperpol=1)
        print_output = sys.stdout.getvalue().strip()
        self.assert_str(print_output, ref.OUTPUT_BY_ATOM_n0)

    def test_outfile_PAn0_by_bond(self):
        self.m.max_l = 1
        Da = self.m.Da #use for beta internally and will be set in output
        self.m.output_by_atom(fmt="%12.5f", max_l=1, hyperpol=1, bond_centers=True)
        print_output = sys.stdout.getvalue().strip()
        self.assert_str(print_output, ref.OUTPUT_BY_BOND_11)

if __name__ == "__main__":
    pass
