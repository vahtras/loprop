import os 
import numpy as np
import loprop
from util import full

tmpdir=os.path.join('h2o', 'tmp')
origin=[0., 0., 0.]

def test_total_charge():
    m = loprop.MolFrag(tmpdir, maxl=0)
    print m.Qab
    assert np.allclose(m.Qab.sum(), -10.0)

def test_charge():
    M = loprop.MolFrag(tmpdir, maxl=0)
    Qref = [-8.70343886, -0.64828057, -0.64828057] 
    Qaa = M.Qab.diagonal()
    print Qref, Qaa
    assert np.allclose(Qref, Qaa)

def test_total_dipole():
    m = loprop.MolFrag(tmpdir, maxl = 1, gc=origin)
    Dref = [0, 0, 5.65148934]
    # molecular dipole moment wrt gauge center gc
    Dtot = m.Dab.sum(axis=2).sum(axis=1).view(full.matrix)
    Qa = m.Qab.diagonal()
    Q = Qa.sum()
    Dtot += Qa*m.R - Q*m.Rc
    print Dref, Dtot
    assert np.allclose(-Dtot, Dref)

def test_dipole_allbonds():
    m = loprop.MolFrag(tmpdir, maxl = 1)
    O    = [ 0.00000000, 0.00000000, -0.39827574]
    H1   = [ 0.10330994, 0.00000000,  0.07188960]
    H2   = [-0.10330994, 0.00000000,  0.07188960]
    OH1  = [ 0.10023328/2, 0.00000000,   0.11470275/2] 
    OH2  = [-0.10023328/2, 0.00000000,   0.11470275/2]
    H1H2 = [ 0.00000000/2,  0.00000000, -0.00378789/2]

    Dref = full.init([[O, OH1, OH2], [OH1, H1, H1H2], [OH2, H1H2, H2]])
    D = m.Dab
    print Dref, D
    assert np.allclose(Dref, D)

def test_dipole_nobonds():
    m = loprop.MolFrag(tmpdir, maxl = 1) 
    O  = [ 0.00000000, 0.00000000, -0.28357300]
    H1 = [ 0.15342658, 0.00000000,  0.12734703]
    H2 = [-0.15342658, 0.00000000,  0.12734703]

    Dref = full.init([O, H1, H2])
    Daa = m.Dab.sum(axis=2).view(full.matrix)
    print Dref, Daa
    assert np.allclose(Dref, Daa)

def test_quadrupole_total():
    QUref = [ 7.31176355, 0., 0., 5.43243241, 0., 9.49794117 ]
    m = loprop.MolFrag(tmpdir, maxl=2, gc=origin)
    print m.gc
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
                    RRab[ij, a, b] = m.Qab[a, b]*m.R[a, i]*m.R[b, j]
                    ij += 1
    QUcab = m.QUab + rRab + RRab
    QUc = QUcab.sum(axis=2).sum(axis=1)
    print "QUcab", QUcab
    print "QUc", QUc, QUref
    assert np.allclose(QUref, -QUc)
    

def test_quadrupole_allbonds():
    M = loprop.MolFrag(tmpdir, maxl = 2)
    O    = [-3.68114747, 0.00000000,  0.00000000, -4.58632761, 0.00000000, -4.24741556]
    H1   = [-0.47568174, 0.00000000, -0.03144252, -0.46920879, 0.00000000, -0.50818752]
    H2   = [-0.47568174, 0.00000000,  0.03144252, -0.46920879, 0.00000000, -0.50818752]
    OH1  = [0.53710687/2, 0.00000000/2,  0.43066796/2, 0.04316104/2, 0.00000000/2, 0.36285790/2]
    OH2  = [0.53710687/2, 0.00000000/2, -0.43066796/2, 0.04316104/2, 0.00000000/2, 0.36285790/2]
    H1H2 = [0.00148694/2, 0.00000000/2,  0.00000000/2, 0.00599079/2, 0.00000000/2, 0.01223822/2]

    QUref = full.init([[O, OH1, OH2], [OH1, H1, H1H2], [OH2, H1H2, H2]])
    QU = M.QUab
    print QUref-QU
    assert np.allclose(QUref, QU, atol=1e-5)

def test_quadrupole_nobonds():
    M = loprop.MolFrag(tmpdir, maxl = 2)
    O =  [-3.29253618, 0.00000000, 0.00000000, -4.54316657, 0.00000000, -4.00465380]
    H1 = [-0.13213704, 0.00000000, 0.24980518, -0.44463288, 0.00000000, -0.26059139]
    H2 = [-0.13213704, 0.00000000,-0.24980518, -0.44463288, 0.00000000, -0.26059139]


    QUref = full.init([O, H1, H2])
    QUaa = (M.QUab + M.dQUab).sum(axis=2).view(full.matrix)
    print QUref, QUaa, QUref-QUaa
    assert np.allclose(QUref, QUaa, atol=1e-5)

# Test of polarizability

def test_Fab():
    Fabref = full.init([
        [-0.11E-03,  0.55E-04,  0.55E-04],
        [ 0.55E-04, -0.55E-04,  0.16E-30],
        [ 0.55E-04,  0.16E-30, -0.55E-04]
        ])

    m = loprop.MolFrag(tmpdir)
    m.set_Fab(alpha=2/loprop.xtang**2)
    
    print Fabref, m.Fab
    assert np.allclose(Fabref, m.Fab, atol=1e-6)

    #
    # shift
    #
def molcas_shift(M):
    maxM = np.max(np.abs(M))
    return 2*maxM

def test_molcas_shift():
    Fabref = full.init([
        [0.11E-03, 0.28E-03, 0.28E-03],
        [0.28E-03, 0.17E-03, 0.22E-03],
        [0.28E-03, 0.22E-03, 0.17E-03]
        ])

    m = loprop.MolFrag(tmpdir)
    m.set_Fab(alpha=2/loprop.xtang**2, shift=molcas_shift)

    print Fabref, m.Fab 
    assert np.allclose(Fabref, m.Fab, atol=1e-5, rtol=1e-2)

    #
    # shift
    #


def test_total_charge_shift():
    m = loprop.MolFrag(tmpdir, maxl=0, pol=False)
    m.pol()
    dQ = m.dQab.sum(axis=2).sum(axis=1).view(full.matrix)
    dQref = [0., 0., 0.]
    print dQref, dQ
    assert np.allclose(dQref, dQ)

def test_atomic_charge_shift():
    ff = 0.001
    m = loprop.MolFrag(tmpdir, maxl=0, pol=False)
    m.pol()
# dQ
# -0.00000650  0.00150789 -0.00150139
# -0.00000595 -0.00150230  0.00150826
# -0.00000201  0.00000331 -0.00000130
# -0.00000201  0.00000100  0.00000101
# -0.00177384  0.00088692  0.00088692
#  0.00176668 -0.00088334 -0.00088334
    dQa = m.dQab.sum(axis=2).view(full.matrix).T
    dQaorig = full.init([
        [-0.00000650,  0.00150789, -0.00150139],
        [-0.00000595, -0.00150230,  0.00150826],
        [-0.00000201,  0.00000331, -0.00000130],
        [-0.00000201,  0.00000100,  0.00000101],
        [-0.00177384,  0.00088692,  0.00088692],
        [ 0.00176668, -0.00088334, -0.00088334],
        ])
    print dQaorig
    dQaref = dQaorig[:,0:6:2]
    print "1", dQaref
    dQaref -= dQaorig[:,1:6:2]
    print "2", dQaref
    dQaref /= 2*ff
    print "3"
    rtol=1e-3
    atol=2e-3
    print dQaref, dQa, 
    print "abs(dQaref - dQa)",abs(dQaref - dQa) 
    print "test 1", atol + rtol*abs(dQa)
    print "test 2", atol + rtol*abs(dQaref)
    print "abs(dQaref - dQa)<test",abs(dQaref - dQa) < atol + rtol*abs(dQa)

    assert np.allclose(dQaref, dQa, rtol=rtol, atol=atol)
    #assert np.allclose(dQaref, dQa)

def test_lagrangian():
# values per "perturbation" as in atomic_charge_shift below
    ff = 0.001
    laorig = full.init([
      [  0.0392366, -27.2474016,  27.2081650  ],
      [  0.0358964,  27.2214515, -27.2573479  ],
      [  0.01211180, -0.04775576,  0.03564396 ],
      [  0.01210615, -0.00594030, -0.00616584 ],
      [ 10.69975088, -5.34987556, -5.34987532 ],
      [-10.6565582,   5.3282791,   5.3282791  ],
      ])
    laref = laorig[:,0:6:2]
    laref -= laorig[:,1:6:2]
    laref /= 2*ff
#...>
    m = loprop.MolFrag(tmpdir, pol=True)
    m.set_Fab(alpha=2/loprop.xtang**2, shift=molcas_shift)
    m.set_la()

    #dQa = m.dQab.sum(axis=2).view(full.matrix).T
    #la = dQa/m.Fab
    #print dQa, m.Fab
    print laref, m.la
    #assert np.allclose(laref, la, rtol=1e-3, atol=1e-3)

def test_DQ():
    DQref = full.init([
        [ 0.18E-02, 0.28E-03, 0.28E-03],
        [-0.88E-03, 0.17E-03, 0.22E-03],
        [-0.88E-03, 0.22E-03, 0.17E-03]
        ])

    m = loprop.MolFrag(tmpdir, maxl=0, pol=False)
    m.pol()
    dQa = m.dQab.sum(axis=2).view(full.matrix).T
    print dQa

    print "DQref", DQref, DQref.sum(axis=0), DQref.sum(axis=1)
    assert False


def test_polarizability_total():
    m = loprop.MolFrag(tmpdir, maxl=0, pol=True)
    Aref = [8.186766009140, 0., 5.102747935447, 0., 0., 6.565131856389]
    # symmetrize over components
    A = full.matrix((6, m.noa, m.noa))
    for a in range(m.noa):
        for b in range(m.noa):
            ij = 0
            for i in range(3):
                for j in range(i):
                    A[ij, a, b] = .5*(m.Aab[i, j, a, b] + m.Aab[j, i, a, b])
                    A[ij, a, b] += .5*(m.dQab[i, a, b]*m.Rab[a, b, j]
                                     + m.dQab[j, a, b]*m.Rab[a, b, i])
                    ij += 1
                A[ij, a, b] = m.Aab[i, i, a, b] 
                A[ij, a, b] += m.dQab[ i, a, b] *m.Rab[a, b, i]
                ij += 1
    Am = A.sum(axis=2).sum(axis=1).view(full.matrix)
    print Am, Aref
    assert np.allclose(Am, Aref)
        

def test_polarizability_allbonds():
    M = loprop.MolFrag(tmpdir, pol=True)

    O = [
    0.76145382,
   -0.00001648, 1.75278523,
   -0.00007538, 0.00035773, 1.39756345
    ]
    H1O = [
    3.11619527,
    0.00019911, 1.25132346,
    2.11363325, 0.00111442, 2.12790474
    ]

    H1 = [
    0.57935224,
    0.00018083, 0.43312326,
    0.11495546, 0.00004222, 0.45770123
    ]

    H2O = [
    3.11568759,
    0.00019821,  1.25132443,
   -2.11327482, -0.00142746, 2.12790473
    ]
    H2H1 = [
    0.04078206,
   -0.00008380, -0.01712262,
   -0.00000098,  0.00000084, -0.00200285
    ]
    H2 = [
    0.57930522,
    0.00018221,  0.43312149,
   -0.11493635, -0.00016407,  0.45770123
    ]



    Aref = full.init([O, H1O, H1, H2O, H2H1, H2])
    #double symmetrize to match
    Asym=full.matrix(Aref.shape)
    #a > b, i>j
    ab = 0
    for a in range(3):
        for b in range(a):
            Asym[:, ab] = (M.Aab[:, :, a, b] + M.Aab[:, :, b, a]).pack()
            ab += 1
        Asym[:, ab] = M.Aab[:, :, a, a].pack()
        ab += 1
    print Aref, Asym, Aref - Asym
    assert np.allclose(Aref, Asym, atol=1e-5)
    

def zest_polarizability_nobonds():
    M = loprop.MolFrag(tmpdir, pol=True)
    O = [
    3.87739525,
    0.00018217, 3.00410918,
    0.00010384, 0.00020122, 3.52546819
    ]

    H1 = [
    2.15784091,
    0.00023848, 1.05022368,
    1.17177159, 0.00059985, 1.52065218
    ]

    H2 = [
    2.15754005,
    0.00023941,  1.05022240,
   -1.17157425, -0.00087738,  1.52065217
    ]

    Aref = full.init([O, H1, H2])

    Asym = full.matrix(Aref.shape)
    #a > b, i>j
    Aa = M.Aab.sum(axis=2).view(full.matrix)
    for a in range(3):
        Asym[:, a] = Aa[:,:,a].pack()
    print Aref, Asym, Aref - Asym
    assert np.allclose(Aref, Asym, atol=1e-5)

    print M.Aab
    Aaa = M.Aab.sum(axis=2).view(full.matrix)
    #symmetry packed
    Asp = full.matrix((6, 3))
    for a in range(3):
        Asp[:, a] = Aaa[a].pack()

    print Aref, Asp, Aref-Asp
    #assert np.allclose(Aref, Asp, atol=1e-5)
    assert True


if __name__ == "__main__":
    test_polarizability_total()
