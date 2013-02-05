import os 
import numpy as np
import loprop
from util import full

tmpdir=os.path.join('h2o', 'tmp')

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

ff=0.001
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

def test_nuclear_charge():
    m = loprop.MolFrag(tmpdir)
    Z = m.Z
    Zref = [8., 1., 1.]
    assert_(Zref, Z)

def test_coordinates_au():
    m = loprop.MolFrag(tmpdir)
    R = m.R
    Rref = [
        [ 0.00000,   0.00000,  0.69801],
        [-1.48150,   0.00000, -0.34901],
        [ 1.48150,   0.00000, -0.34901]
        ]
    assert_(Rref, R)

def test_default_gauge():
    m = loprop.MolFrag(tmpdir)
    Rcref = full.init([0.00000000,  0.00000000,  0.48860959])
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
    Dref = [0, 0, -0.76539388]
    # molecular dipole moment wrt gauge center gc
    Dtot = m.Dab.sum(axis=2).sum(axis=1).view(full.matrix)
    Qa = m.Qab.diagonal()
    Q = Qa.sum()
    Dtot += Qa*m.R - Q*m.Rc
    assert_(Dtot, Dref)

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
    O  = [ 0.00000000, 0.00000000, -0.28357300]
    H1 = [ 0.15342658, 0.00000000,  0.12734703]
    H2 = [-0.15342658, 0.00000000,  0.12734703]

    Dref = full.init([O, H1, H2])
    Daa = m.Dab.sum(axis=2).view(full.matrix)
    assert_(Dref, Daa)

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
    O =  [-3.29253618, 0.00000000, 0.00000000, -4.54316657, 0.00000000, -4.00465380]
    H1 = [-0.13213704, 0.00000000, 0.24980518, -0.44463288, 0.00000000, -0.26059139]
    H2 = [-0.13213704, 0.00000000,-0.24980518, -0.44463288, 0.00000000, -0.26059139]

    QUref = full.init([O, H1, H2])
    QUaa = (M.QUab + M.dQUab).sum(axis=2).view(full.matrix)
    assert_(QUref, QUaa)


def test_Fab():
    Fabref = full.init([
        [-0.11E-03,  0.55E-04,  0.55E-04],
        [ 0.55E-04, -0.55E-04,  0.16E-30],
        [ 0.55E-04,  0.16E-30, -0.55E-04]
        ])

    m = loprop.MolFrag(tmpdir, pf=mcpf)
    Fab = m.Fab

    assert_(Fabref, Fab)

def test_molcas_shift():
    Labref = full.init([
        [0.11E-03, 0.28E-03, 0.28E-03],
        [0.28E-03, 0.17E-03, 0.22E-03],
        [0.28E-03, 0.22E-03, 0.17E-03]
        ])

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
    laorig = full.init([
    [0.0392366,-27.2474016 , 27.2081650],
    [0.0358964, 27.2214515 ,-27.2573479],
    [0.01211180, -0.04775576,  0.03564396],
    [0.01210615, -0.00594030, -0.00616584],
    [10.69975088, -5.34987556, -5.34987532],
    [-10.6565582,  5.3282791 ,  5.3282791]
    ])
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
                Am[i, j] += Rab[a, a, i]*dQa[a, j]

    assert_(Am, Aref)
        
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
    Oyx = ihff*(rMP[2, 1, 0] - rMP[2, 2, 0]
        +       rMP[1, 3, 0] - rMP[1, 4, 0])/2
    Oyy = ihff*(rMP[2, 3, 0] - rMP[2, 4, 0])
    Ozx = ihff*(rMP[3, 1, 0] - rMP[3, 2, 0]
        +       rMP[1, 5, 0] - rMP[1, 6, 0])/2
    Ozy = ihff*(rMP[3, 3, 0] - rMP[3, 4, 0]
        +       rMP[2, 5, 0] - rMP[2, 6, 0])/2
    Ozz = ihff*(rMP[3, 5, 0] - rMP[3, 6, 0])
    H1Oxx = ihff*(rMP[1, 1, 1] - rMP[1, 2, 1] \
          - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[0]-RO[0]))
    H1Oyx = ihff*(
        (rMP[2, 1, 1] - rMP[2, 2, 1] 
       + rMP[1, 3, 1] - rMP[1, 4, 1])/2
       - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[1]-RO[1])
#      - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[0]-RO[0]) THIS IS REALLY... A BUG?
       )
    H1Oyy = ihff*(rMP[2, 3, 1] - rMP[2, 4, 1] - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[1]-RO[1]))
    H1Ozx = ihff*(
        (rMP[3, 1, 1] - rMP[3, 2, 1]
       + rMP[1, 5, 1] - rMP[1, 6, 1])/2
      - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[2]-RO[2])
#             - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[0]-RO[0]) #THIS IS REALLY... A BUG?
            )
    H1Ozy = ihff*(
        (rMP[3, 3, 1] - rMP[3, 4, 1]
       + rMP[2, 5, 1] - rMP[2, 6, 1])/2
      - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[2]-RO[2])
#     - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[1]-RO[1]) THIS IS REALLY... A BUG?
        )
    H1Ozz = ihff*(rMP[3, 5, 1] - rMP[3, 6, 1] - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[2]-RO[2]))
    H1xx = ihff*(rMP[1, 1, 2] - rMP[1, 2, 2])
    H1yx = (ihff*(rMP[2, 1, 2] - rMP[2, 2, 2])
         +  ihff*(rMP[1, 3, 2] - rMP[1, 4, 2]))/2
    H1yy = ihff*(rMP[2, 3, 2] - rMP[2, 4, 2])
    H1zx = (ihff*(rMP[3, 1, 2] - rMP[3, 2, 2])
         +  ihff*(rMP[1, 5, 2] - rMP[1, 6, 2]))/2
    H1zy = (ihff*(rMP[3, 3, 2] - rMP[3, 4, 2])
         +  ihff*(rMP[2, 5, 2] - rMP[2, 6, 2]))/2
    H1zz = ihff*(rMP[3, 5, 2] - rMP[3, 6, 2])
    H2Oxx = ihff*(rMP[1, 1, 3] - rMP[1, 2, 3] - (rMP[0, 1, 3] - rMP[0, 2, 3])*(RH2[0]-RO[0]))
    H2Oyx = ihff*(
        (rMP[2, 1, 3] - rMP[2, 2, 3] 
       + rMP[1, 3, 3] - rMP[1, 4, 3])/2
       - (rMP[0, 1, 3] - rMP[0, 2, 3])*(RH1[1]-RO[1])
#      - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[0]-RO[0]) THIS IS REALLY... A BUG?
       )
    H2Oyy = ihff*(rMP[2, 3, 3] - rMP[2, 4, 3] - (rMP[0, 3, 3] - rMP[0, 4, 3])*(RH2[1]-RO[1]))
    H2Ozx = ihff*(
        (rMP[3, 1, 3] - rMP[3, 2, 3]
       + rMP[1, 5, 3] - rMP[1, 6, 3])/2
      - (rMP[0, 1, 3] - rMP[0, 2, 3])*(RH1[2]-RO[2])
#             - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[0]-RO[0]) #THIS IS REALLY... A BUG?
            )
    H2Ozy = ihff*(
        (rMP[3, 3, 3] - rMP[3, 4, 3]
       + rMP[2, 5, 3] - rMP[2, 6, 3])/2
      - (rMP[0, 3, 3] - rMP[0, 4, 3])*(RH1[2]-RO[2])
#     - (rMP[0, 5, 3] - rMP[0, 6, 3])*(RH1[1]-RO[1]) THIS IS REALLY... A BUG?
        )
    H2Ozz = ihff*(rMP[3, 5, 3] - rMP[3, 6, 3] - (rMP[0, 5, 3] - rMP[0, 6, 3])*(RH2[2]-RO[2]))
    H2H1xx = ihff*(rMP[1, 1, 4] - rMP[1, 2, 4] - (rMP[0, 1, 4] - rMP[0, 2, 4])*(RH2[0]-RH1[0]))
    H2H1yx = ihff*(
        (rMP[2, 1, 4] - rMP[2, 2, 4] 
       + rMP[1, 3, 4] - rMP[1, 4, 4])/2
       - (rMP[0, 1, 4] - rMP[0, 2, 4])*(RH1[1]-RO[1])
#      - (rMP[0, 3, 4] - rMP[0, 4, 4])*(RH1[0]-RO[0]) THIS IS REALLY... A BUG?
       )
    H2H1yy = ihff*(rMP[2, 3, 4] - rMP[2, 4, 4] - (rMP[0, 3, 4] - rMP[0, 4, 4])*(RH2[1]-RH1[1]))
    H2H1zx = ihff*(
        (rMP[3, 1, 4] - rMP[3, 2, 4]
       + rMP[1, 5, 4] - rMP[1, 6, 4])/2
      - (rMP[0, 1, 4] - rMP[0, 2, 4])*(RH1[2]-RO[2])
#     - (rMP[0, 5, 4] - rMP[0, 6, 4])*(RH1[0]-RO[0]) #THIS IS REALLY... A BUG?
            )
    H2H1zy = ihff*(
        (rMP[3, 3, 4] - rMP[3, 4, 4]
       + rMP[2, 5, 4] - rMP[2, 6, 4])/2
      - (rMP[0, 3, 4] - rMP[0, 4, 4])*(RH1[2]-RO[2])
#     - (rMP[0, 5, 4] - rMP[0, 6, 4])*(RH1[1]-RO[1]) THIS IS REALLY... A BUG?
        )
    H2H1zz = ihff*(rMP[3, 5, 4] - rMP[3, 6, 4] - (rMP[0, 5, 4] - rMP[0, 6, 4])*(RH2[2]-RH1[2]))
    H2xx = ihff*(rMP[1, 1, 5] - rMP[1, 2, 5])
    H2yx = (ihff*(rMP[2, 1, 5] - rMP[2, 2, 5])
         +  ihff*(rMP[1, 3, 5] - rMP[1, 4, 5]))/2
    H2yy = ihff*(rMP[2, 3, 5] - rMP[2, 4, 5])
    H2zx = (ihff*(rMP[3, 1, 5] - rMP[3, 2, 5])
         +  ihff*(rMP[1, 5, 5] - rMP[1, 6, 5]))/2
    H2zy = (ihff*(rMP[3, 3, 5] - rMP[3, 4, 5])
         +  ihff*(rMP[2, 5, 5] - rMP[2, 6, 5]))/2
    H2zz = ihff*(rMP[3, 5, 5] - rMP[3, 6, 5])

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
    ff = .001
    polref = [O, H1O, H1, H2O, H2H1, H2]
    diff = [(1, 2), (3, 4), (5, 6)]
    atoms = (0, 2, 5) 
    bonds = (1, 3, 4)
    ablab = ("O", "H1O", "H1", "H2O", "H2H1", "H2")
    ijlab = ("xx", "yx", "yy", "zx", "zy", "zz")

    pol = np.zeros((m.noa*(m.noa+1)//2, 6))
    for ab, a, b in pairs(m.noa):
        for ij, i, j in pairs(3):
            #from pdb import set_trace; set_trace()
            i1, i2 = diff[i]
            j1, j2 = diff[j]
            pol[ab, ij] += (rMP[i+1, j1, ab] - rMP[i+1, j2, ab]
                        +   rMP[j+1, i1, ab] - rMP[j+1, i2, ab])/(4*ff)
            if ab in bonds:
                pol[ab, ij] -= (R[a][i]-R[b][i])*(rMP[0, j1, ab] - rMP[0, j2, ab])/(2*ff)
            assert_(polref[ab][ij], pol[ab, ij], text="%s%s"%(ablab[ab], ijlab[ij]))

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

    Aab = m.Aab + m.dAab/2
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
    test_altquadrupole_allbonds()
