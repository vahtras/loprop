import os 
import numpy as np
from ..core import MolFrag, penalty_function, xtang, pairs
from ..daltools.util import full

import re
thisdir  = os.path.dirname(__file__)
case = "h2o_rot"
tmpdir=os.path.join(thisdir, case, 'tmp')
exec('from . import %s_data as ref'%case)


def assert_(this, ref, atol=1e-5, text=None):
    if text: print(text)
    print(this, ref)
    print("Max deviation", np.amax(this - ref))
    assert np.allclose(this, ref, atol=atol)

def assert_str(this, ref, text=None):
    if text: print(text)
    print(this, ref)
    assert this == ref


def setup():
    global m
    global ff
# modify Gagliardi penalty function to include unit conversion bug
    m = MolFrag(tmpdir, pf=penalty_function(2.0/xtang**2))
    ff = ref.ff

def test_nuclear_charge():
    Z = m.Z
    assert_(Z, ref.Z)

def test_coordinates_au():
    R = m.R
    assert_(R, ref.R)

def test_default_gauge():
    assert_(ref.Rc, m.Rc)

def test_total_charge():
    Qtot = m.Qab.sum()
    assert_(Qtot, ref.Qtot)

def test_charge():
    Qaa = m.Qa
    assert_(ref.Q, Qaa)

def test_total_dipole():
    assert_(m.Dtot, ref.Dtot)

def test_dipole_allbonds():
    D = full.matrix(ref.D.shape)
    Dab = m.Dab
    for ab, a, b in pairs(m.noa):
        D[:, ab] += Dab[:, a, b ] 
        if a != b: D[:, ab] += Dab[:, b, a] 
    assert_(D, ref.D)

def test_dipole_allbonds_sym():
    Dsym = m.Dsym
    assert_(Dsym, ref.D)

def test_dipole_nobonds():
    Daa = m.Dab.sum(axis=2).view(full.matrix)
    assert_(Daa, ref.Daa)

def test_quadrupole_total():
    rrab=full.matrix((6, m.noa, m.noa))
    rRab=full.matrix((6, m.noa, m.noa))
    RRab=full.matrix((6, m.noa, m.noa))
    Rabc = 1.0*m.Rab
    for a in range(m.noa):
        for b in range(m.noa):
            Rabc[a,b,:] -= m.Rc
    for a in range(m.noa):
        for b in range(m.noa):
            ij = 0
            for i in range(3):
                for j in range(i,3):
                    rRab[ij, a, b] = m.Dab[i, a, b]*Rabc[a, b, j]\
                                   + m.Dab[j, a, b]*Rabc[a, b, i]
                    RRab[ij, a, b] = m.Qab[a, b]*(m.R[a, i] - m.Rc[i])*(m.R[b, j] - m.Rc[j])
                    ij += 1
    QUcab = m.QUab + rRab + RRab
    QUc = QUcab.sum(axis=2).sum(axis=1).view(full.matrix)
    assert_(QUc, ref.QUc)
    
def test_nuclear_quadrupole():
    QUN = m.QUN
    assert_(QUN, ref.QUN)

def test_quadrupole_allbonds():
    QU = full.matrix(ref.QU.shape)
    QUab = m.QUab
    for ab, a, b in pairs(m.noa):
        QU[:, ab] += QUab[:, a, b ] 
        if a != b: QU[:, ab] += QUab[:, b, a] 
    assert_(QU, ref.QU)

def test_quadrupole_allbonds_sym():
    QUsym = m.QUsym
    assert_(QUsym, ref.QU)

def test_quadrupole_nobonds():

    QUaa = (m.QUab + m.dQUab).sum(axis=2).view(full.matrix)
    assert_(QUaa, ref.QUaa)


def test_Fab():

    Fab = m.Fab
    assert_(Fab, ref.Fab)

def test_molcas_shift():
    Fab = m.Fab
    Lab = Fab + m.sf(Fab)

    assert_(Lab, ref.Lab)

def test_total_charge_shift():
    dQ = m.dQa[0].sum(axis=0).view(full.matrix)
    dQref = [0., 0., 0.]
    assert_(dQref, dQ)

def test_atomic_charge_shift():
    dQa = m.dQa[0]
    dQaref = (ref.dQa[:, 1::2] - ref.dQa[:, 2::2])/(2*ff)

    assert_(dQa, dQaref, .006)

def test_lagrangian():
# values per "perturbation" as in atomic_charge_shift below
    la = m.la[0]
    laref = (ref.la[:,0:6:2] - ref.la[:,1:6:2])/(2*ff)
# The sign difference is because mocas sets up rhs with opposite sign
    assert_(-laref, la, atol=100)

def test_bond_charge_shift():
    dQab = m.dQab[0]
    noa = m.noa


    dQabref = (ref.dQab[:, 1:7:2] - ref.dQab[:, 2:7:2])/(2*ff)
    dQabcmp = full.matrix((3, 3))
    ab = 0
    for a in range(noa):
        for b in range(a):
            dQabcmp[ab, :] = dQab[a, b, :]
            ab += 1
# The sign difference is because mocas sets up rhs with opposite sign
    assert_(-dQabref, dQabcmp, 0.006)

def test_bond_charge_shift_sum():
    dQa  = m.dQab[0].sum(axis=1).view(full.matrix)
    dQaref = m.dQa[0]
    assert_(dQa, dQaref)


def test_polarizability_total():

    Am = m.Am[0]

    assert_(Am, ref.Am, 0.015)
        
def test_polarizability_allbonds_molcas_internal():

    O = ref.O
    H1O = ref.H1O
    H1 = ref.H1
    H2O = ref.H2O
    H2H1 = ref.H2H1
    H2 = ref.H2
    rMP = ref.rMP

    RO, RH1, RH2 = m.R
    ROx, ROy, ROz = RO
    RH1x, RH1y, RH1z = RH1
    RH2x, RH2y, RH2z = RH2
    

    ihff = 1/(2*ff)

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

    assert_(O[0], Oxx, text="Oxx")
    assert_(O[1], Oyx, text="Oyx")
    assert_(O[2], Oyy, text="Oyy")
    assert_(O[3], Ozx, text="Ozx")
    assert_(O[4], Ozy, text="Ozy")
    assert_(O[5], Ozz, text="Ozz")
    assert_(H1O[0], H1Oxx, text="H1Oxx")
    assert_(H1O[1], H1Oyx, text="H1Oyx")
    assert_(H1O[2], H1Oyy, text="H1Oyy")
    assert_(H1O[3], H1Ozx, text="H1Ozx")
    assert_(H1O[4], H1Ozy, text="H1Ozy")
    assert_(H1O[5], H1Ozz, text="H1Ozz")
    assert_(H1[0], H1xx, text="H1xx")
    assert_(H1[1], H1yx, text="H1yx")
    assert_(H1[2], H1yy, text="H1yy")
    assert_(H1[3], H1zx, text="H1zx")
    assert_(H1[4], H1zy, text="H1zy")
    assert_(H1[5], H1zz, text="H1zz")
    assert_(H2O[0], H2Oxx, text="H2Oxx")
    assert_(H2O[1], H2Oyx, text="H2Oyx")
    assert_(H2O[2], H2Oyy, text="H2Oyy")
    assert_(H2O[3], H2Ozx, text="H2Ozx")
    assert_(H2O[4], H2Ozy, text="H2Ozy")
    assert_(H2O[5], H2Ozz, text="H2Ozz")
    assert_(H2H1[0], H2H1xx, text="H2H1xx")
    assert_(H2H1[1], H2H1yx, text="H2H1yx")
    assert_(H2H1[2], H2H1yy, text="H2H1yy")
    assert_(H2H1[3], H2H1zx, text="H2H1zx")
    assert_(H2H1[4], H2H1zy, text="H2H1zy")
    assert_(H2H1[5], H2H1zz, text="H2H1zz")
    assert_(H2[0], H2xx, text="H2xx")
    assert_(H2[1], H2yx, text="H2yx")
    assert_(H2[2], H2yy, text="H2yy")
    assert_(H2[3], H2zx, text="H2zx")
    assert_(H2[4], H2zy, text="H2zy")
    assert_(H2[5], H2zz, text="H2zz")

def test_altint():
    R = m.R
    rMP = ref.rMP
    diff = [(1, 2), (3, 4), (5, 6)]
    atoms = (0, 2, 5) 
    bonds = (1, 3, 4)
    ablab = ("O", "H1O", "H1", "H2O", "H2H1", "H2")
    ijlab = ("xx", "yx", "yy", "zx", "zy", "zz")

    pol = np.zeros((6, m.noa*(m.noa+1)//2))
    for ab, a, b in pairs(m.noa):
        for ij, i, j in pairs(3):
            #from pdb import set_trace; set_trace()
            i1, i2 = diff[i]
            j1, j2 = diff[j]
            pol[ij, ab] += (rMP[i+1, j1, ab] - rMP[i+1, j2, ab]
                        +   rMP[j+1, i1, ab] - rMP[j+1, i2, ab])/(4*ff)
            if ab in bonds:
                pol[ij, ab] -= (R[a][i]-R[b][i])*(rMP[0, j1, ab] - rMP[0, j2, ab])/(2*ff)
            assert_(ref.Aab[ij, ab], pol[ij, ab], text="%s%s"%(ablab[ab], ijlab[ij]))

def test_polarizability_allbonds_atoms():

    Aab = m.Aab[0] #+ m.dAab
    noa = m.noa

    Acmp=full.matrix(ref.Aab.shape)
    
    ab = 0
    for a in range(noa):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    # atoms
    assert_(ref.Aab[:, 0], Acmp[:, 0], .005)
    assert_(ref.Aab[:, 2], Acmp[:, 2], .005)
    assert_(ref.Aab[:, 5], Acmp[:, 5], .005)

def test_polarizability_allbonds_bonds():

    Aab = m.Aab[0] + m.dAab[0]/2
    noa = m.noa

    Acmp=full.matrix(ref.Aab.shape)
    
    ab = 0
    for a in range(noa):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    # atoms
    assert_(ref.Aab[:, 1], Acmp[:, 1], .150, 'H1O')
    assert_(ref.Aab[:, 3], Acmp[:, 3], .150, 'H2O')
    assert_(ref.Aab[:, 4], Acmp[:, 4], .005, 'H2H1')
    

def test_polarizability_nobonds():

    Aab = m.Aab[0] + m.dAab[0]/2
    noa = m.noa

    Acmp = full.matrix((6, noa ))
    Aa = Aab.sum(axis=3).view(full.matrix)
    
    ab = 0
    for a in range(noa):
        Acmp[:, a,] = Aa[:, :, a].pack()

    # atoms
    assert_(Acmp, ref.Aa, 0.07)

def test_potfile_PAn0():
    PAn0 = m.output_potential_file(maxl=-1, pol=0, hyper=0)
    assert_str(PAn0, ref.PAn0)

def test_potfile_PA00():
    PA00 = m.output_potential_file(maxl=0, pol=0, hyper=0)
    assert_str(PA00, ref.PA00)

def test_potfile_PA10():
    PA10 = m.output_potential_file(maxl=1, pol=0, hyper=0)
    assert_str(PA10, ref.PA10)

def test_potfile_PA20():
    PA20 = m.output_potential_file(maxl=2, pol=0, hyper=0)
    assert_str(PA20, ref.PA20)

def test_potfile_PA21():
    PA21 = m.output_potential_file(maxl=2, pol=1, hyper=0)
    assert_str(PA21, ref.PA21)

def test_potfile_PA22():
    PA22 = m.output_potential_file(maxl=2, pol=2, hyper=0)
    assert_str(PA22, ref.PA22)

if __name__ == "__main__":
    import sys
    setup()
    exec("%s()"%sys.argv[1])
