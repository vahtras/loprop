import os 
import numpy as np
import loprop
from util import full

tmpdir=os.path.join('h2o_rot', 'tmp')

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
    0.52702211,
   -0.04998028, 0.59526807,
    0.09132377, 0.04262747, 0.41525801
]

H1O = [
    0.72842854,
    0.49069835, 2.57026115,
   -0.07274981,-1.65456138, 2.38918515
]

H1 = [
    0.26784425,
    0.05213792, 0.37318941,
   -0.04806460,-0.15594795, 0.44996726
]

H2O = [
    2.57576906,
   -0.88177757, 1.01304153,
   -1.38837953, 0.85225504, 2.09689902,
]

H2H1 = [
   -0.01290768,
   -0.00347111,-0.00709622,
    0.01479145, 0.00726970,-0.02842193
]
H2 = [
    0.40673759,
   -0.03533002, 0.25637547,
   -0.15873042, 0.05106665, 0.42844188,
]
Aref = full.init([O, H1O, H1, H2O, H2H1, H2])

ff=0.001
rMP = full.init([
#O
    [
    [-8.65876991,   0.23513101,   -0.41152423,    0.02182444,   -4.10259557, -0.02640338,   -0.35701148,   -4.11454656,   -0.18737011,   -3.77627245 ],
    [-8.65966380,   0.23565694,   -0.41157439,    0.02191561,   -4.10430219, -0.02582351,   -0.35678918,   -4.11537968,   -0.18769353,   -3.77728460 ],
    [-8.65786747,   0.23460290,   -0.41147422,    0.02173069,   -4.10087541, -0.02698333,   -0.35723318,   -4.11370532,   -0.18704504,   -3.77525274 ],
    [-8.65719904,   0.23508187,   -0.41092870,    0.02186809,   -4.10094135, -0.02660224,   -0.35701455,   -4.11183148,   -0.18761695,   -3.77441606 ],
    [-8.66033231,   0.23518161,   -0.41211923,    0.02178468,   -4.10424535, -0.02620166,   -0.35700650,   -4.11725066,   -0.18712341,   -3.77812281 ],
    [-8.65884440,   0.23522235,   -0.41148098,    0.02224001,   -4.10227926, -0.02665554,   -0.35755568,   -4.11496604,   -0.18636217,   -3.77646405 ],
    [-8.65867458,   0.23504197,   -0.41156808,    0.02140949,   -4.10290754, -0.02615241,   -0.35646904,   -4.11411892,   -0.18837868,   -3.77607070 ]
    ],
#H1O                                                                                                                    
    [
    [ 0.00000000,   0.00898855,    0.15409012,   -0.13486969,    0.02281518, 0.03832377 ,  -0.00815698 ,   0.54574628 ,  -0.45416412 ,   0.41359357 ],
    [ 0.00029358,   0.00967204,    0.15428477,   -0.13464363,    0.02264624, 0.03842412 ,  -0.00814072 ,   0.54589004 ,  -0.45429250 ,   0.41369137 ],
    [-0.00029351,   0.00830816,    0.15389694,   -0.13509349,    0.02298329, 0.03822374 ,  -0.00817385 ,   0.54560189 ,  -0.45403534 ,   0.41349554 ],
    [ 0.00114660,   0.00901179,    0.15517595,   -0.13498968,    0.02290713, 0.03828688 ,  -0.00819508 ,   0.54600002 ,  -0.45416857 ,   0.41375544 ],
    [-0.00113670,   0.00896454,    0.15300617,   -0.13475104,    0.02272220, 0.03836065 ,  -0.00811942 ,   0.54549053 ,  -0.45415956 ,   0.41343123 ],
    [-0.00128954,   0.00934855,    0.15374695,   -0.13409685,    0.02275339, 0.03828862 ,  -0.00819605 ,   0.54551246 ,  -0.45387473 ,   0.41338506 ],
    [ 0.00130093,   0.00862582,    0.15443447,   -0.13564629,    0.02287684, 0.03835912 ,  -0.00811837 ,   0.54597985 ,  -0.45445480 ,   0.41380294 ]
    ],                                                                                                                                 
#H1                                                                                                                     
    [                                                                                                                                 
    [-0.67061347,   0.01485867,    0.07708189,   -0.08099761,   -0.45800547, 0.00291480 ,  -0.00728464 ,  -0.50930014 ,   0.03415076 ,  -0.47957835 ],
    [-0.67090705,   0.01512686,    0.07714872,   -0.08105741,   -0.45833067, 0.00283302 ,  -0.00723319 ,  -0.50962785 ,   0.03413846 ,  -0.47987638 ],
    [-0.67031996,   0.01459117,    0.07701531,   -0.08093789,   -0.45768170, 0.00299626 ,  -0.00733588 ,  -0.50897362 ,   0.03416297 ,  -0.47928152 ],
    [-0.67176007,   0.01489654,    0.07745701,   -0.08113657,   -0.45907862, 0.00292065 ,  -0.00732500 ,  -0.51051507 ,   0.03416558 ,  -0.48057979 ],
    [-0.66947677,   0.01482139,    0.07671063,   -0.08086232,   -0.45694802, 0.00290889 ,  -0.00724490 ,  -0.50810118 ,   0.03413549 ,  -0.47859185 ],
    [-0.66932393,   0.01482266,    0.07690846,   -0.08054991,   -0.45674408, 0.00290179 ,  -0.00722820 ,  -0.50805689 ,   0.03412463 ,  -0.47826215 ],
    [-0.67191440,   0.01489540,    0.07725800,   -0.08144985,   -0.45928282, 0.00292779 ,  -0.00734175 ,  -0.51055885 ,   0.03417666 ,  -0.48091062 ]
    ],                                                                                                                                 
#H2O                                                                                                                    
    [                                                                                                                                 
    [ 0.00000000,  -0.14318840,    0.08078556,    0.12241327,    0.47959403, -0.25138711,   -0.37309624,    0.16064536,    0.23185608,    0.34191355 ],
    [-0.00118748,  -0.14211002,    0.08055718,    0.12245764,    0.47934395, -0.25125576,   -0.37302609,    0.16058346,    0.23182826,    0.34172530 ],
    [ 0.00119595,  -0.14426817,    0.08101485,    0.12236948,    0.47984212, -0.25151786,   -0.37316630,    0.16070571,    0.23188326,    0.34210108 ],
    [ 0.00042427,  -0.14324667,    0.08153516,    0.12266724,    0.47956668, -0.25135214,   -0.37304195,    0.16087371,    0.23177055,    0.34194448 ],
    [-0.00042570,  -0.14313188,    0.08003592,    0.12215842,    0.47962012, -0.25142131,   -0.37314973,    0.16041621,    0.23194039,    0.34188185 ],
    [ 0.00121505,  -0.14328103,    0.08126381,    0.12312580,    0.47982473, -0.25146790,   -0.37332022,    0.16072795,    0.23204649,    0.34206106 ],
    [-0.00120560,  -0.14309554,    0.08030939,    0.12170269,    0.47936364, -0.25130625,   -0.37287344,    0.16056250,    0.23166576,    0.34176665 ]
    ],                                                                                                                                
#H2H1                                                                                                                   
    [                                                                                                                                 
    [ 0.00000000,   0.00116808,   -0.00204368,    0.00010787,    0.00322826, 0.00004172 ,   0.00466455 ,   0.00374013 ,   0.00241150 ,  -0.00119167 ],
    [ 0.00000000,   0.00115526,   -0.00204698,    0.00012261,    0.00325125, 0.00003145 ,   0.00465378 ,   0.00375345 ,   0.00241662 ,  -0.00117449 ],
    [ 0.00000000,   0.00118108,   -0.00204007,    0.00009273,    0.00320606, 0.00005173 ,   0.00467541 ,   0.00372820 ,   0.00240627 ,  -0.00120814 ],
    [ 0.00000000,   0.00116432,   -0.00205081,    0.00011552,    0.00320722, 0.00004836 ,   0.00467208 ,   0.00370151 ,   0.00242192 ,  -0.00122184 ],
    [ 0.00000000,   0.00117129,   -0.00203661,    0.00010084,    0.00325005, 0.00003496 ,   0.00465637 ,   0.00377916 ,   0.00240094 ,  -0.00116040 ],
    [ 0.00000000,   0.00118271,   -0.00203651,    0.00007950,    0.00322006, 0.00004708 ,   0.00467464 ,   0.00375135 ,   0.00239442 ,  -0.00118938 ],
    [ 0.00000000,   0.00115342,   -0.00205091,    0.00013635,    0.00323736, 0.00003542 ,   0.00465698 ,   0.00373120 ,   0.00242982 ,  -0.00119605 ]
    ],                                                                                                                                
#H2
    [                                                                                                                                 
    [-0.67061662,  -0.07751699,    0.03258355,    0.07518067,   -0.49574976, 0.02685145 ,   0.02286702 ,  -0.47748424 ,  -0.02253188 ,  -0.47365821 ],
    [-0.66942914,  -0.07711206,    0.03253374,    0.07503459,   -0.49449392, 0.02681520 ,   0.02287738 ,  -0.47634812 ,  -0.02252606 ,  -0.47259734 ],
    [-0.67181258,  -0.07792554,    0.03263432,    0.07532985,   -0.49702021, 0.02688748 ,   0.02285602 ,  -0.47863437 ,  -0.02253777 ,  -0.47473269 ],
    [-0.67104090,  -0.07753729,    0.03283993,    0.07521228,   -0.49609000, 0.02691830 ,   0.02284438 ,  -0.47793364 ,  -0.02259034 ,  -0.47399829 ],
    [-0.67019092,  -0.07749655,    0.03232717,    0.07514851,   -0.49540844, 0.02678459 ,   0.02288970 ,  -0.47703386 ,  -0.02247336 ,  -0.47331702 ],
    [-0.67183168,  -0.07768819,    0.03265421,    0.07561105,   -0.49691993, 0.02684496 ,   0.02286599 ,  -0.47869380 ,  -0.02259253 ,  -0.47490843 ],
    [-0.66941103,  -0.07734852,    0.03251371,    0.07475417,   -0.49459304, 0.02685764 ,   0.02286759 ,  -0.47628835 ,  -0.02247162 ,  -0.47242185 ]
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
        [-0.27439,  0.48018, -0.35773],
        [-0.43275, -0.82090,  0.88874],
        [ 0.98152, -0.13965, -1.50233]
        ]
    assert_(Rref, R)

def test_default_gauge():
    m = loprop.MolFrag(tmpdir)
    Rcref = full.init([-0.16463294,   0.28808875,  -0.34753953])
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
    Dref = [0.40095611, -0.70174560, 0.03721264]
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
    O =  [ 0.16803109,  -0.29408639,   0.01559623 ]
    H1 = [ 0.01993698,   0.15310511,  -0.14837852 ]
    H2 = [-0.14852716,   0.07195449,   0.13644124 ]

    Dref = full.init([O, H1, H2])
    Daa = m.Dab.sum(axis=2).view(full.matrix)
    assert_(Dref, Daa)

def test_quadrupole_total():
    QUref = full.init([-5.97224464, 0.24966841, 0.77110470, -6.17226877, 0.42783060,-6.57765870] )
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
    O =  [-3.94201869,  -0.09440612,  -0.46008896,  -3.88662892,  -0.24872405, -3.55263150 ]
    H1 = [-0.44344604,   0.03059799,  -0.01783139,  -0.13501155,  -0.28237140, -0.18945131 ]
    H2 = [-0.16524860,  -0.14585061,  -0.24009742,  -0.36955871,   0.13544768, -0.23311075 ]

    QUref = full.init([O, H1, H2])
    QUaa = (M.QUab + M.dQUab).sum(axis=2).view(full.matrix)
    assert_(QUref, QUaa)


def test_Fab():
    Fabref = full.init([
        [-0.12E-03,  0.58E-04, 0.58E-04],
        [ 0.58E-04, -0.58E-04, 0.19E-28],
        [ 0.58E-04,  0.19E-28,-0.58E-04]
        ])

    m = loprop.MolFrag(tmpdir, pf=mcpf)
    Fab = m.Fab

    assert_(Fabref, Fab)

def test_molcas_shift():
    Labref = full.init([
        [0.12E-03, 0.29E-03, 0.29E-03],
        [0.29E-03, 0.17E-03, 0.23E-03],
        [0.29E-03, 0.23E-03, 0.17E-03]
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
    [-5.10951420, -10.14238550,  15.25189980],
    [ 5.15832880,  10.18998130, -15.3483100],
    [ 8.97693540, -10.67897570,   1.7020403],
    [-8.92856407,  10.55769081,  -1.6291267],
    [-0.42410210,  21.68234150, -21.2582394],
    [ 0.54318080, -21.75839020,  21.2152094],
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
    Aref = full.init([
         [ 4.49289385,  -0.42772271, -1.56180915],
         [-0.42772271,   4.80103940, -0.85729045],
         [-1.56180915,  -0.85729045,  5.75132939]
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

    assert_(Am, Aref, 0.015)
        
def notest_polarizability_allbonds_molcas_internal():
    m = loprop.MolFrag(tmpdir, pf=mcpf, sf=mcsf)
    ff = .001

# polarizabilities
    RO, RH1, RH2 = m.R

    ihff = 1/(2*ff)
    Oxx =  ihff*(rMP[1, 1, 0] - rMP[1, 2, 0])
    Oyx = (ihff*(rMP[2, 1, 0] - rMP[2, 2, 0]) 
         + ihff*(rMP[1, 3, 0] - rMP[1, 4, 0]))/2
    Oyy =  ihff*(rMP[2, 3, 0] - rMP[2, 4, 0])
    Ozx = (ihff*(rMP[3, 1, 0] - rMP[3, 2, 0]) 
         + ihff*(rMP[1, 5, 0] - rMP[1, 6, 0]))/2
    Ozy = (ihff*(rMP[3, 3, 0] - rMP[3, 4, 0]) 
         + ihff*(rMP[2, 5, 0] - rMP[2, 6, 0]))/2
    Ozz =  ihff*(rMP[3, 5, 0] - rMP[3, 6, 0])

    H1Oxx =  ihff*(rMP[1, 1, 1] - rMP[1, 2, 1] - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[0]-RO[0]))
    H1Oyx = (ihff*(rMP[2, 1, 1] - rMP[2, 2, 1] - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[1]-RO[1])) \
          +  ihff*(rMP[1, 3, 1] - rMP[1, 4, 1] - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[0]-RO[0])))/2
    H1Oyy = ihff*(rMP[2, 3, 1] - rMP[2, 4, 1] - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[1]-RO[1]))
    H1Ozx = (ihff*(rMP[3, 1, 1] - rMP[3, 2, 1] - (rMP[0, 1, 1] - rMP[0, 2, 1])*(RH1[2]-RO[2])) \
          +  ihff*(rMP[1, 5, 1] - rMP[1, 6, 1] - (rMP[0, 5, 1] - rMP[0, 4, 1])*(RH1[0]-RO[0])))
    H1Ozy = (ihff*(rMP[3, 3, 1] - rMP[3, 4, 1] - (rMP[0, 3, 1] - rMP[0, 4, 1])*(RH1[2]-RO[2])) \
          +  ihff*(rMP[2, 5, 1] - rMP[2, 6, 1] - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[1]-RO[1])))
    H1Ozz = ihff*(rMP[3, 5, 1] - rMP[3, 6, 1] - (rMP[0, 5, 1] - rMP[0, 6, 1])*(RH1[2]-RO[2]))

    H1xx =  ihff*(rMP[1, 1, 2] - rMP[1, 2, 2])
    H1yx = (ihff*(rMP[2, 1, 2] - rMP[2, 2, 2]) 
          + ihff*(rMP[1, 3, 2] - rMP[1, 4, 2]))/2
    H1yy =  ihff*(rMP[2, 3, 2] - rMP[2, 4, 2])
    H1zx = (ihff*(rMP[3, 1, 2] - rMP[3, 2, 2]) 
          + ihff*(rMP[1, 5, 2] - rMP[1, 6, 2]))/2
    H1zy = (ihff*(rMP[3, 3, 2] - rMP[3, 4, 2]) 
          + ihff*(rMP[2, 5, 2] - rMP[2, 6, 2]))/2
    H1zz =  ihff*(rMP[3, 5, 2] - rMP[3, 6, 2])

    H2Oxx = ihff*(rMP[1, 1, 3] - rMP[1, 2, 3] - (rMP[0, 1, 3] - rMP[0, 2, 3])*(RH2[0]-RO[0]))
    H2Oyy = ihff*(rMP[2, 3, 3] - rMP[2, 4, 3] - (rMP[0, 3, 3] - rMP[0, 4, 3])*(RH2[1]-RO[1]))
    H2Ozz = ihff*(rMP[3, 5, 3] - rMP[3, 6, 3] - (rMP[0, 5, 3] - rMP[0, 6, 3])*(RH2[2]-RO[2]))
    H2H1xx = ihff*(rMP[1, 1, 4] - rMP[1, 2, 4] - (rMP[0, 1, 4] - rMP[0, 2, 4])*(RH2[0]-RH1[0]))
    H2H1yy = ihff*(rMP[2, 3, 4] - rMP[2, 4, 4] - (rMP[0, 3, 4] - rMP[0, 4, 4])*(RH2[1]-RH1[1]))
    H2H1zz = ihff*(rMP[3, 5, 4] - rMP[3, 6, 4] - (rMP[0, 5, 4] - rMP[0, 6, 4])*(RH2[2]-RH1[2]))

    H2xx =  ihff*(rMP[1, 1, 5] - rMP[1, 2, 5])
    H2yx = (ihff*(rMP[2, 1, 5] - rMP[2, 2, 5]) 
          + ihff*(rMP[1, 3, 5] - rMP[1, 4, 5]))/2
    H2yy =  ihff*(rMP[2, 3, 5] - rMP[2, 4, 5])
    H2zx = (ihff*(rMP[3, 1, 5] - rMP[3, 2, 5]) 
          + ihff*(rMP[1, 5, 5] - rMP[1, 6, 5]))/2
    H2zy = (ihff*(rMP[3, 3, 5] - rMP[3, 4, 5]) 
          + ihff*(rMP[2, 5, 5] - rMP[2, 6, 5]))/2
    H2zz =  ihff*(rMP[3, 5, 5] - rMP[3, 6, 5])

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
    #assert_(H2O[0], H2Oyx, text="H2Oyx")
    assert_(H2O[2], H2Oyy, text="H2Oyy")
    #assert_(H2O[5], H2Ozx, text="H2Ozx")
    #assert_(H2O[5], H2Ozy, text="H2Ozy")
    assert_(H2O[5], H2Ozz, text="H2Ozz")
    assert_(H2H1[0], H2H1xx, text="H2H1xx")
    #assert_(H2H1[0], H2H1yx, text="H2H1yx")
    assert_(H2H1[2], H2H1yy, text="H2H1yy")
    #assert_(H2H1[5], H2H1zx, text="H2H1zx")
    #assert_(H2H1[5], H2H1zy, text="H2H1zy")
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
    test_default_gauge()
