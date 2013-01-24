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

def assert_(ref, here, atol=1e-5, text=None):
    if text: print text
    print ref, here
    assert np.allclose(ref, here, atol=atol)

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

rMP = full.init([
#O
    [
    [-8.70343886, 0.00000000,  0.00000000, -0.39827574, -3.68114747,  0.00000000,  0.00000000, -4.58632761,  0.00000000, -4.24741556],
    [-8.70343235, 0.00076124,  0.00000000, -0.39827535, -3.68114147,  0.00000000,  0.00193493, -4.58631888,  0.00000000, -4.24741290],
    [-8.70343291,-0.00076166,  0.00000000, -0.39827505, -3.68114128,  0.00000000, -0.00193603, -4.58631789,  0.00000000, -4.24741229],
    [-8.70343685,-0.00000006,  0.00175241, -0.39827457, -3.68114516,  0.00000000,  0.00000161, -4.58632717,  0.00053363, -4.24741642],
    [-8.70343685, 0.00000000, -0.00175316, -0.39827456, -3.68114514,  0.00000000,  0.00000000, -4.58632711, -0.00053592, -4.24741639],
    [-8.70166502, 0.00000000,  0.00000144, -0.39688042, -3.67884999,  0.00000000,  0.00000000, -4.58395384,  0.00000080, -4.24349307],
    [-8.70520554, 0.00000000,  0.00000000, -0.39967554, -3.68344246,  0.00000000,  0.00000000, -4.58868836,  0.00000000, -4.25134640],
    ],
#H1O                                                                                                                    
    [
    [ 0.00000000, 0.10023328,  0.00000000,  0.11470275,  0.53710687,  0.00000000,  0.43066796,  0.04316104,  0.00000000,  0.36285790],
    [ 0.00150789, 0.10111974,  0.00000000,  0.11541803,  0.53753360,  0.00000000,  0.43120945,  0.04333774,  0.00000000,  0.36314215],
    [-0.00150230, 0.09934695,  0.00000000,  0.11398581,  0.53667861,  0.00000000,  0.43012612,  0.04298361,  0.00000000,  0.36257249],
    [ 0.00000331, 0.10023328,  0.00125017,  0.11470067,  0.53710812, -0.00006107,  0.43066944,  0.04316020,  0.00015952,  0.36285848],
    [ 0.00000100, 0.10023249, -0.00125247,  0.11470042,  0.53710716,  0.00006135,  0.43066837,  0.04316018, -0.00015966,  0.36285788],
    [ 0.00088692, 0.10059268, -0.00000064,  0.11590322,  0.53754715, -0.00000006,  0.43071206,  0.04334198, -0.00000015,  0.36330053],
    [-0.00088334, 0.09987383,  0.00000000,  0.11350091,  0.53666602,  0.00000000,  0.43062352,  0.04297910,  0.00000000,  0.36241326],
    ],
#H1                                                                                                                     
    [
    [-0.64828057, 0.10330994,  0.00000000,  0.07188960, -0.47568174,  0.00000000, -0.03144252, -0.46920879,  0.00000000, -0.50818752],
    [-0.64978846, 0.10389186,  0.00000000,  0.07204462, -0.47729337,  0.00000000, -0.03154159, -0.47074619,  0.00000000, -0.50963693],
    [-0.64677827, 0.10273316,  0.00000000,  0.07173584, -0.47408263,  0.00000000, -0.03134407, -0.46768337,  0.00000000, -0.50674873],
    [-0.64828388, 0.10331167,  0.00043314,  0.07189029, -0.47568875, -0.00023642, -0.03144270, -0.46921635, -0.00021728, -0.50819386],
    [-0.64828157, 0.10331095, -0.00043311,  0.07188988, -0.47568608,  0.00023641, -0.03144256, -0.46921346,  0.00021729, -0.50819095],
    [-0.64916749, 0.10338629, -0.00000024,  0.07234862, -0.47634698,  0.00000013, -0.03159569, -0.47003679,  0.00000011, -0.50936853],
    [-0.64739723, 0.10323524,  0.00000000,  0.07143322, -0.47502412,  0.00000000, -0.03129003, -0.46838912,  0.00000000, -0.50701656],
    ],
#H2O                                                                                                                    
    [
    [ 0.00000000,-0.10023328,  0.00000000,  0.11470275,  0.53710687,  0.00000000, -0.43066796,  0.04316104,  0.00000000,  0.36285790],
    [-0.00150139,-0.09934749,  0.00000000,  0.11398482,  0.53667874,  0.00000000, -0.43012670,  0.04298387,  0.00000000,  0.36257240],
    [ 0.00150826,-0.10112008,  0.00000000,  0.11541676,  0.53753350,  0.00000000, -0.43120982,  0.04333795,  0.00000000,  0.36314186],
    [-0.00000130,-0.10023170,  0.00125018,  0.11470018,  0.53710620,  0.00006107, -0.43066732,  0.04316017,  0.00015952,  0.36285728],
    [ 0.00000101,-0.10023249, -0.00125247,  0.11470042,  0.53710716, -0.00006135, -0.43066838,  0.04316018, -0.00015966,  0.36285788],
    [ 0.00088692,-0.10059268, -0.00000064,  0.11590322,  0.53754715,  0.00000006, -0.43071206,  0.04334198, -0.00000015,  0.36330053],
    [-0.00088334,-0.09987383,  0.00000000,  0.11350091,  0.53666602,  0.00000000, -0.43062352,  0.04297910,  0.00000000,  0.36241326],
    ],
#H2H1                                                                                                                   
    [
    [ 0.00000000, 0.00000000,  0.00000000, -0.00378789,  0.00148694,  0.00000000,  0.00000000,  0.00599079,  0.00000000,  0.01223822],
    [ 0.00000000, 0.00004089,  0.00000000, -0.00378786,  0.00148338,  0.00000000, -0.00004858,  0.00599281,  0.00000000,  0.01224094],
    [ 0.00000000,-0.00004067,  0.00000000, -0.00378785,  0.00148341,  0.00000000,  0.00004861,  0.00599277,  0.00000000,  0.01224093],
    [ 0.00000000,-0.00000033, -0.00001707, -0.00378763,  0.00149017,  0.00000000,  0.00000001,  0.00599114, -0.00001229,  0.01223979],
    [ 0.00000000, 0.00000000,  0.00001717, -0.00378763,  0.00149019,  0.00000000,  0.00000000,  0.00599114,  0.00001242,  0.01223980],
    [ 0.00000000, 0.00000000,  0.00000000, -0.00378978,  0.00141897,  0.00000000,  0.00000000,  0.00590445,  0.00000002,  0.01210376],
    [ 0.00000000, 0.00000000,  0.00000000, -0.00378577,  0.00155694,  0.00000000,  0.00000000,  0.00607799,  0.00000000,  0.01237393],
    ],
#H2                                                                                                                     
    [
    [-0.64828057,-0.10330994,  0.00000000,  0.07188960, -0.47568174,  0.00000000,  0.03144252, -0.46920879,  0.00000000, -0.50818752],
    [-0.64677918,-0.10273369,  0.00000000,  0.07173576, -0.47408411,  0.00000000,  0.03134408, -0.46768486,  0.00000000, -0.50674986],
    [-0.64978883,-0.10389230,  0.00000000,  0.07204446, -0.47729439,  0.00000000,  0.03154159, -0.47074717,  0.00000000, -0.50963754],
    [-0.64827927,-0.10331022,  0.00043313,  0.07188947, -0.47568340,  0.00023642,  0.03144242, -0.46921057, -0.00021727, -0.50818804],
    [-0.64828158,-0.10331095, -0.00043311,  0.07188988, -0.47568609, -0.00023641,  0.03144256, -0.46921348,  0.00021729, -0.50819097],
    [-0.64916749,-0.10338629, -0.00000024,  0.07234862, -0.47634698, -0.00000013,  0.03159569, -0.47003679,  0.00000011, -0.50936853],
    [-0.64739723,-0.10323524,  0.00000000,  0.07143322, -0.47502412,  0.00000000,  0.03129003, -0.46838912,  0.00000000, -0.50701656]
    ]
    ])

def test_default_gauge():
    m = loprop.MolFrag(tmpdir)
    Rcref = full.init([0.00000000,  0.00000000,  0.48860959])
    assert_(Rcref, m.Rc)

def test_total_charge():
    m = loprop.MolFrag(tmpdir, maxl=0)
    assert_(m.Qab.sum(), -10.0)

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
    QUref = full.init([-7.31176220, 0., 0., -5.43243232, 0., -6.36258665])
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
    print "QUc", QUc, QUref
    assert np.allclose(QUref, QUc)
    

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
    dQaorig = rMP[0, :, (0,2,5)]; ff=0.001
# sign headaches:
# normally ff calc complements h1 with f*op (field times operator)
# if integrals are <z> the interaction hamiltonian is really -f*<z>
# so dQ/df = (Q[-f<z>] - Q[f<z>])/(2f)
    dQaref = dQaorig[:,2:7:2]
    dQaref -= dQaorig[:,1:7:2]
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
    dQab = m.dQab
    noa = m.noa

    ff = 0.001
    dQaborig = full.init([
        [ 0.00000000,  0.00000000,  0.00000000],
        [ 0.00150789, -0.00150139,  0.00000000],
        [-0.00150230,  0.00150826,  0.00000000],
        [ 0.00000331, -0.00000130,  0.00000000],
        [ 0.00000100,  0.00000101,  0.00000000],
        [ 0.00088692,  0.00088692,  0.00000000],
        [-0.00088334, -0.00088334,  0.00000000]
        ])            
    dQaborig = rMP[0, :, (1, 3, 4)]
#   ff = 0.00005
#   dQaborig = full.init([
#       [ 0.00000000,  0.00000000,  0.00000000],
#       [ 0.00007568, -0.00007482,  0.00000000],
#       [-0.00007486,  0.00007573,  0.00000000],
#       [ 0.00000071,  0.00000015,  0.00000000],
#       [ 0.00000043,  0.00000043,  0.00000000],
#       [ 0.00004476,  0.00004476,  0.00000000],
#       [-0.00004386, -0.00004386,  0.00000000]
#       ])            

    dQabref = dQaborig[:, 1:7:2]
    dQabref -= dQaborig[:, 2:7:2]
    dQabref /= (2*ff)
    
    dQabcmp = full.matrix((3, 3))
    #from pdb import set_trace; set_trace()
    #print dQabcmp
    ab = 0
    for a in range(noa):
        for b in range(a):
            dQabcmp[ab, :] = dQab[a, b, :]
            ab += 1

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
    dQa = m.dQa
    Rab = m.Rab
    Aab = m.Aab
    noa = m.noa
    
    Am = Aab.sum(axis=3).sum(axis=2).view(full.matrix)
    for i in range(3):
        for j in range(3):
            for a in range(noa):
                Am[i, j] -= Rab[a, a, i]*dQa[a, j]

    print Am, Aref
    assert np.allclose(Am, Aref)
        
def test_polarizability_allbonds_molcas_internal():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    ff = .001

# polarizabilities
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

    RO = [0.0000000000000, 0.0000000000000, 0.6980137021426]
    RH1 = [-1.4814997377056, 0.0000000000000, -0.3490068510713]
    RH2 = [1.4814997377056, 0.0000000000000, -0.3490068510713]

    ihff = 1/(2*ff)
    Oxx = ihff*(rMP[1, 1, 0] - rMP[1, 2, 0])
    Oyy = ihff*(rMP[2, 3, 0] - rMP[2, 4, 0])
    Ozz = ihff*(rMP[3, 5, 0] - rMP[3, 6, 0])
    H1Oxx = ihff*(rMP[1, 1, 1] - rMP[1, 2, 1] - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[0]-RO[0]))
    H1Oyy = ihff*(rMP[2, 3, 1] - rMP[2, 4, 1] - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[1]-RO[1]))
    H1Ozz = ihff*(rMP[3, 5, 1] - rMP[3, 6, 1] - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[2]-RO[2]))
    H1xx = ihff*(rMP[1, 1, 2] - rMP[1, 2, 2])
    H1yy = ihff*(rMP[2, 3, 2] - rMP[2, 4, 2])
    H1zz = ihff*(rMP[3, 5, 2] - rMP[3, 6, 2])
    H2Oxx = ihff*(rMP[1, 1, 3] - rMP[1, 2, 3] - (rMP[0, 1, 3] - rMP[0, 2, 3])*(RH2[0]-RO[0]))
    H2Oyy = ihff*(rMP[2, 3, 3] - rMP[2, 4, 3] - (rMP[0, 3, 3] - rMP[0, 4, 3])*(RH2[1]-RO[1]))
    H2Ozz = ihff*(rMP[3, 5, 3] - rMP[3, 6, 3] - (rMP[0, 5, 3] - rMP[0, 6, 3])*(RH2[2]-RO[2]))
    H2H1xx = ihff*(rMP[1, 1, 4] - rMP[1, 2, 4] - (rMP[0, 1, 4] - rMP[0, 2, 4])*(RH2[0]-RH1[0]))
    H2H1yy = ihff*(rMP[2, 3, 4] - rMP[2, 4, 4] - (rMP[0, 3, 4] - rMP[0, 4, 4])*(RH2[1]-RH1[1]))
    H2H1zz = ihff*(rMP[3, 5, 4] - rMP[3, 6, 4] - (rMP[0, 5, 4] - rMP[0, 6, 4])*(RH2[2]-RH1[2]))
    H2xx = ihff*(rMP[1, 1, 5] - rMP[1, 2, 5])
    H2yy = ihff*(rMP[2, 3, 5] - rMP[2, 4, 5])
    H2zz = ihff*(rMP[3, 5, 5] - rMP[3, 6, 5])

    print "Oxx", O[0], Oxx
    assert np.allclose(O[0], Oxx)
    print "Oyy", O[2], Oyy
    assert np.allclose(O[2], Oyy)
    print "Ozz", O[5], Ozz
    assert np.allclose(O[5], Ozz)
    print "H1Oxx", H1O[0], H1Oxx
    assert np.allclose(H1O[0], H1Oxx)
    print "H1Oyy", H1O[2], H1Oyy
    assert np.allclose(H1O[2], H1Oyy)
    print "H1Ozz", H1O[5], H1Ozz
    assert np.allclose(H1O[5], H1Ozz)
    print "H1xx", H1[0], H1xx
    assert np.allclose(H1[0], H1xx)
    print "H1yy", H1[2], H1yy
    assert np.allclose(H1[2], H1yy)
    print "H1zz", H1[5], H1zz
    assert np.allclose(H1[5], H1zz)
    print "H2Oxx", H2O[0], H2Oxx
    assert np.allclose(H2O[0], H2Oxx)
    print "H2Oyy", H2O[2], H2Oyy
    assert np.allclose(H2O[2], H2Oyy)
    print "H2Ozz", H2O[5], H2Ozz
    assert np.allclose(H2O[5], H2Ozz)
    print "H2H1xx", H2H1[0], H2H1xx
    assert np.allclose(H2H1[0], H2H1xx, atol=1e-5)
    print "H2H1yy", H2H1[2], H2H1yy
    assert np.allclose(H2H1[2], H2H1yy, atol=1e-5)
    print "H2H1zz", H2H1[5], H2H1zz
    assert np.allclose(H2H1[5], H2H1zz, atol=1e-5)
    print "H2xx", H2[0], H2xx
    assert np.allclose(H2[0], H2xx)
    print "H2yy", H2[2], H2yy
    assert np.allclose(H2[2], H2yy)
    print "H2zz", H2[5], H2zz
    assert np.allclose(H2[5], H2zz)

def test_polarizability_allbonds_atoms():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)

    Aab = m.Aab + m.dAab
    noa = m.noa

    Acmp=full.matrix(Aref.shape)
    
    ab = 0
    for a in range(3):
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

    Aab = m.Aab - m.dAab/2
    noa = m.noa

    Acmp=full.matrix(Aref.shape)
    
    ab = 0
    for a in range(3):
        for b in range(a):
            Acmp[:, ab] = (Aab[:, :, a, b] + Aab[:, :, b, a]).pack()
            ab += 1
        Acmp[:, ab] = Aab[:, :, a, a].pack()
        ab += 1
    # atoms
    assert_(Aref[:, 1], Acmp[:, 1], .150, 'H1O')
    assert_(Aref[:, 3], Acmp[:, 3], .150, 'H2O')
    assert_(Aref[:, 4], Acmp[:, 4], .005, 'H2H1')
    

def nozest_polarizability_nobonds():
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
    test_default_gauge()
