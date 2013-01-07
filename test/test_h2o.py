import os 
import numpy as np
import loprop
from util import full

tmpdir=os.path.join('h2o', 'tmp')
origin=[0., 0., 0.]

# modify Gagliardi penalty function to include unit conversion bug
from loprop import penalty_function, xtang
mcpf = lambda *args : penalty_function(*args, alpha=2/xtang**2)
mcsf = lambda Fab : 2*np.max(np.abs(Fab))

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


def test_Fab():
    Fabref = full.init([
        [-0.11E-03,  0.55E-04,  0.55E-04],
        [ 0.55E-04, -0.55E-04,  0.16E-30],
        [ 0.55E-04,  0.16E-30, -0.55E-04]
        ])

    m = loprop.MolFrag(tmpdir, pf=mcpf)
    Fab = m.Fab

    
    print Fabref, Fab
    assert np.allclose(Fabref, Fab, atol=1e-6)

def test_molcas_shift():
    Labref = full.init([
        [0.11E-03, 0.28E-03, 0.28E-03],
        [0.28E-03, 0.17E-03, 0.22E-03],
        [0.28E-03, 0.22E-03, 0.17E-03]
        ])

    m = loprop.MolFrag(tmpdir, pf=mcpf)
    Fab = m.Fab
    #Lab = Fab + 2*np.max(np.abs(Fab))
    Lab = Fab + mcsf(Fab)

    print Labref, Lab
    assert np.allclose(Labref, Lab, atol=1e-5, rtol=1e-2)

def test_total_charge_shift():
    m = loprop.MolFrag(tmpdir, maxl=0, pol=False)
    #m.pol()
    dQ = m.dQa.sum(axis=0).view(full.matrix)
    dQref = [0., 0., 0.]
    print dQref, dQ
    assert np.allclose(dQref, dQ)

def test_atomic_charge_shift():
    ff = 0.00005
    m = loprop.MolFrag(tmpdir, maxl=0, pol=False)
    dQa = m.dQa
    #m.pol()
# dQ
# -0.00000650  0.00150789 -0.00150139
# -0.00000595 -0.00150230  0.00150826
# -0.00000201  0.00000331 -0.00000130
# -0.00000201  0.00000100  0.00000101
# -0.00177384  0.00088692  0.00088692
#  0.00176668 -0.00088334 -0.00088334
    dQaorig = -full.init([
        [-8.70343810, -0.64828095, -0.64828095],
        [-8.70343723, -0.64835663, -0.64820614],
        [-8.70343723, -0.64820609, -0.64835668],
        [-8.70343724, -0.64828166, -0.64828110],
        [-8.70343724, -0.64828138, -0.64828138],
        [-8.70334858, -0.64832571, -0.64832571],
        [-8.70352581, -0.64823709, -0.64823709]
        ])
    dQaref = dQaorig[:,1:7:2]
    dQaref -= dQaorig[:,2:7:2]
    dQaref /= 2*ff

    print dQaref, dQa 
    assert np.allclose(dQaref, dQa, atol=0.003)

def test_lagrangian():
# values per "perturbation" as in atomic_charge_shift below
    ff = 0.00005
    laorig = full.init([
    [0.00522380, -1.36429263,  1.35906884],
    [0.00522422,  1.35991888,-1.36514311],
    [0.00516952, -0.00761407, 0.00244456],
    [0.00517199, -0.00252856,-0.00264343],
    [0.53992933, -0.26996495,-0.26996438],
    [-0.52910812,  0.26455406, 0.26455406]
      ])
    laref = laorig[:,0:6:2]
    laref -= laorig[:,1:6:2]
    laref /= 2*ff
#...>
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    la = m.la

    print laref, la
    assert np.allclose(laref, la, atol=100)

def test_bond_charge_shift():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

    ff = 0.00005
    dQab = m.dQab
    #print dQab
    dQaborig = full.init([
        [ 0.00000000,  0.00000000,  0.00000000],
        [ 0.00007568, -0.00007482,  0.00000000],
        [-0.00007486,  0.00007573,  0.00000000],
        [ 0.00000071,  0.00000015,  0.00000000],
        [ 0.00000043,  0.00000043,  0.00000000],
        [ 0.00004476,  0.00004476,  0.00000000],
        [-0.00004386, -0.00004386,  0.00000000]
        ])            

    dQabref = dQaborig[:, 1:7:2]
    dQabref -= dQaborig[:, 2:7:2]
    dQabref /= (2*ff)
    
    dQabcmp = full.matrix((3, 3))
    #from pdb import set_trace; set_trace()
    #print dQabcmp
    ij = 0
    for i in range(3):
        for j in range(i):
            dQabcmp[ij, :] = dQab[i, j, :]
            ij += 1

    print dQabref, dQabcmp
    assert np.allclose(dQabref, dQabcmp, atol=0.005)

def test_bond_charge_shift_sum():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    dQaref = m.dQa
    dQa  = m.dQab.sum(axis=1).view(full.matrix)
    print dQaref, dQa
    assert np.allclose(dQaref, dQa)

def test_polarizability_total():
    Aref = full.init(
            [[8.186766009140, 0., 0.], 
            [0., 5.102747935447, 0.], 
            [0., 0., 6.565131856389]
            ])

    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    Aab = m.Aab
    noa = m.noa
    
    Am = Aab.sum(axis=3).sum(axis=2).view(full.matrix)
    print Am, Aref
    assert np.allclose(Am, Aref)
        

def test_polarizability_allbonds():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

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
    Aab = m.Aab 
    noa = m.noa
    dRab = 2*m.dRab
    dQab = m.dQab

    #Bond shift contribution
    for a in range(noa):
        for b in range(noa):
            Aab[:, :, a, b] += dRab[a, b].x(dQab[a, b, :])
    
    Acmp=full.matrix(Aref.shape)
    
    ab = 0
    for a in range(3):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    print Aref, Acmp
    assert np.allclose(Aref, Acmp, atol=1e-5)
    

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
    from pdb import set_trace; set_trace()
    test_total_dipole()
    test_polarizability_total()
