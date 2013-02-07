import os 
import numpy as np
import loprop
from util import full

import re
case = re.sub('test_', '', __name__)
tmpdir=os.path.join(case, 'tmp')

# modify Gagliardi penalty function to include unit conversion bug
from loprop import penalty_function, xtang
mcpf = lambda *args : penalty_function(*args, alpha=2/xtang**2)
mcsf = lambda Fab : 2*np.max(np.abs(Fab))

def assert_(ref, this, atol=1e-5, text=None):
    if text: print text,
    print ref, this
    print "Max deviation", np.amax(ref - this)
    assert np.allclose(ref, this, atol=atol)

def pairs(n):
    li = []
    mn = 0
    for m in range(n):
        for n in range(m+1):
            li.append((mn, m, n))
            mn += 1
    return li

from h2o_data \
    import Aref, ff, rMP, Rref, Rcref, Dtotref, Daaref, QUcref, QUaaref, Fabref, Labref, \
        laorig, Aaref, Amref

def test_nuclear_charge():
    m = loprop.MolFrag(tmpdir)
    Z = m.Z
    Zref = [8., 1., 1.]
    assert_(Zref, Z)

def test_coordinates_au():
    m = loprop.MolFrag(tmpdir)
    R = m.R
    assert_(Rref, R)

def test_default_gauge():
    m = loprop.MolFrag(tmpdir)
    assert_(Rcref, m.Rc)

def test_total_charge():
    m = loprop.MolFrag(tmpdir, maxl=0)
    assert_(m.Qab.sum(), -10.0)

def test_charge():
    M = loprop.MolFrag(tmpdir, maxl=0)
    Qref = rMP[0, 0, (0, 2, 5)]
    Qaa = M.Qab.diagonal()
    print Qref, Qaa
    assert np.allclose(Qref, Qaa)

def test_total_dipole():
    m = loprop.MolFrag(tmpdir)
    # molecular dipole moment wrt gauge center gc
    Dtot = m.Dab.sum(axis=2).sum(axis=1).view(full.matrix)
    Qa = m.Qab.diagonal()
    Q = Qa.sum()
    Dtot += Qa*m.R - Q*m.Rc
    assert_(Dtot, Dtotref)

def test_dipole_allbonds():
    m = loprop.MolFrag(tmpdir)
    Dref = rMP[1:4, 0, :]
    D = full.matrix(Dref.shape)
    Dab = m.Dab
    print Dab
    for ab, a, b in pairs(m.noa):
        D[:, ab] += Dab[:, a, b ] 
        if a != b: D[:, ab] += Dab[:, b, a] 
    assert_(Dref, D)

def test_dipole_nobonds():
    m = loprop.MolFrag(tmpdir, maxl = 1) 

    Daa = m.Dab.sum(axis=2).view(full.matrix)
    assert_(Daaref, Daa)

def test_quadrupole_total():
    m = loprop.MolFrag(tmpdir)
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
    assert_(QUcref, QUc)
    


def test_quadrupole_allbonds():
    m = loprop.MolFrag(tmpdir)
    QUref = rMP[4:, 0, :]
    QU = full.matrix(QUref.shape)
    QUab = m.QUab
    for ab, a, b in pairs(m.noa):
        QU[:, ab] += QUab[:, a, b ] 
        if a != b: QU[:, ab] += QUab[:, b, a] 
    assert_(QUref, QU)

def test_quadrupole_nobonds():
    M = loprop.MolFrag(tmpdir)

    QUaa = (M.QUab + M.dQUab).sum(axis=2).view(full.matrix)
    assert_(QUaaref, QUaa)


def test_Fab():

    m = loprop.MolFrag(tmpdir, pf=mcpf)
    Fab = m.Fab

    assert_(Fabref, Fab)

def test_molcas_shift():

    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    Fab = m.Fab
    Lab = Fab + m.sf(Fab)

    print Labref, Lab
    assert np.allclose(Labref, Lab, atol=1e-5, rtol=1e-2)

def test_total_charge_shift():
    m = loprop.MolFrag(tmpdir)
    dQ = m.dQa.sum(axis=0).view(full.matrix)
    dQref = [0., 0., 0.]
    assert_(dQref, dQ)

def test_atomic_charge_shift():
    m = loprop.MolFrag(tmpdir)
    dQa = m.dQa
    dQaorig = rMP[0, :, (0,2,5)]
    dQaref = dQaorig[:, 1::2]
    dQaref -= dQaorig[:, 2::2]
    dQaref /= 2*ff

    assert_(dQaref, dQa, 0.006)

def test_lagrangian():
# values per "perturbation" as in atomic_charge_shift below
    laref = laorig[:,0:6:2]
    laref -= laorig[:,1:6:2]
    laref /= 2*ff
#...>
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    la = m.la

    print laref, la
# The sign difference is because mocas sets up rhs with opposite sign
    assert np.allclose(-laref, la, atol=100)

def test_bond_charge_shift():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    dQab = m.dQab
    noa = m.noa

    dQaborig = rMP[0, :, (1, 3, 4)]

    dQabref = dQaborig[:, 1:7:2]
    dQabref -= dQaborig[:, 2:7:2]
    dQabref /= (2*ff)
    
    dQabcmp = full.matrix((3, 3))
    ab = 0
    for a in range(noa):
        for b in range(a):
            dQabcmp[ab, :] = dQab[a, b, :]
            ab += 1

    assert_(-dQabref, dQabcmp, 0.006)

def test_bond_charge_shift_sum():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    dQaref = m.dQa
    dQa  = m.dQab.sum(axis=1).view(full.matrix)
    print dQaref, dQa
    assert np.allclose(dQaref, dQa)


def test_polarizability_total():

    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    dQa = m.dQa
    Rab = m.Rab
    Aab = m.Aab
    noa = m.noa
    
    Am = Aab.sum(axis=3).sum(axis=2).view(full.matrix)
    for i in range(3):
        for j in range(3):
            for a in range(noa):
                Am[i, j] += Rab[a, a, i]*dQa[a, j]

    assert_(Am, Amref)
        
def test_polarizability_allbonds_molcas_internal():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    ff = .001

    from h2o_data import O, H1O, H1, H2O, H2H1, H2
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
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    R = m.R
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
            assert_(Aref[ij, ab], pol[ij, ab], text="%s%s"%(ablab[ab], ijlab[ij]))

def test_polarizability_allbonds_atoms():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

    Aab = m.Aab #+ m.dAab
    noa = m.noa

    Acmp=full.matrix(Aref.shape)
    
    ab = 0
    for a in range(noa):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    # atoms
    assert_(Aref[:, 0], Acmp[:, 0], .005)
    assert_(Aref[:, 2], Acmp[:, 2], .005)
    assert_(Aref[:, 5], Acmp[:, 5], .005)

def test_polarizability_allbonds_bonds():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

    Aab = m.Aab + m.dAab/2
    noa = m.noa

    Acmp=full.matrix(Aref.shape)
    
    ab = 0
    for a in range(noa):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    # atoms
    assert_(Aref[:, 1], Acmp[:, 1], .150, 'H1O')
    assert_(Aref[:, 3], Acmp[:, 3], .150, 'H2O')
    assert_(Aref[:, 4], Acmp[:, 4], .005, 'H2H1')
    

def test_polarizability_nobonds():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

    Aab = m.Aab + m.dAab/2
    noa = m.noa

    Acmp = full.matrix((6, noa ))
    Aa = Aab.sum(axis=3).view(full.matrix)
    
    ab = 0
    for a in range(noa):
        Acmp[:, a,] = Aa[:, :, a].pack()

    # atoms
    assert_(Acmp, Aaref, 0.07)


if __name__ == "__main__":
    test_default_gauge()
