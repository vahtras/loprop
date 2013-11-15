#!/usr/bin/env python
"""
Loprop model implementation (J. Chem. Phys. 121, 4494 (2004))
"""
import os, sys, math, numpy
from daltools import one, mol, dens, prop, lr
from util import full, blocked, subblocked, timing

full.matrix.fmt = "%14.6f"
xtang = 0.5291772108
angtx = 1.0/xtang
mc = False

def verbose2(func):
    import inspect
    def wrapper(self, *args, **kwargs):
        print "before %s from %s" % (func.__name__, inspect.stack()[1][3])
        print "with arguments", args, kwargs
        result = func(self, *args, **kwargs)
        print "after %s backto %s" % (func.__name__, inspect.stack()[1][3])
        return result
    return wrapper


# Bragg-Slater radii () converted from Angstrom to Bohr
rbs = numpy.array([0, 
      0.25,                                     0.25, 
      1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,
      1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.00, 
      ])*angtx

def penalty_function(alpha=2):
    """Returns function object """
    def pf(Za, Ra, Zb, Rb):
        """Inverse half of penalty function defined in Gagliardi"""

        from math import exp
        ra = rbs[int(round(Za))]
        rb = rbs[int(round(Zb))]

        xa, ya, za = Ra
        xb, yb, zb = Rb
        rab2 = (xa - xb)**2 + (ya - yb)**2 + (za - zb)**2

        f = 0.5*exp(-alpha*(rab2/(ra+rb)**2))
        return f
    return pf

def pairs(n):
    """Generate index pairs for triangular packed matrices up to n """
    ij = 0
    for i in range(n):
        for j in range(i+1):
            yield (ij, i, j)
            ij += 1

def shift_function(*args):
    """Return value twice max value of F"""
    F, = args
    return 2*numpy.max(numpy.abs(F))

def header(string):
    """Pretty print header"""
    border = '-'*len(string)
    print "\n%s\n%s\n%s" % (border, string, border)


class MolFrag:
    """An instance of the MolFrag class is created and populated with
    data from a Dalton runtime scratch directory"""

    def __init__(
        self, tmpdir, freqs=None, pf=penalty_function, sf=shift_function, gc=None
        ):
        """Constructur of MolFrac class objects
        input: tmpdir, scratch directory of Dalton calculation
        """
        self.tmpdir = tmpdir
        if freqs is None:
            self.freqs = (0,)
            self.nfreqs = 1
        else:
            self.freqs = freqs
            self.nfreqs = len(freqs)
        self.rfreqs = range(self.nfreqs)

        self.pf = pf
        self.sf = sf
        self.gc = gc
        #
        # Dalton files
        #
        self.aooneint = os.path.join(tmpdir,'AOONEINT')
        self.dalton_bas = os.path.join(tmpdir,'DALTON.BAS')
        self.sirifc = os.path.join(tmpdir,'SIRIFC')

        self._S = None
        self._T = None
        self._D = None
        self._Dk = None
        self.get_basis_info()
        #self.get_overlap()
        self.get_isordk()
        #self.get_density()
        #self.transformation()
        self._x = None

        self._Qab = None
        self._Dab = None
        self._Dsym = None
        self._QUab = None
        self._QUsym = None
        self._QUN = None
        self._QUc = None
        self._dQa = None
        self._dQab = None
        self._Fab = None
        self._la = None
        self._Aab = None
        self._Am = None
        self._dAab = None
        #if maxl >= 0: self.charge()
        #if maxl >= 1: self.dipole()
        #if maxl >= 2: self.quadrupole()
        #if pol: self.pol()

    def get_basis_info(self, debug=False):
        """ Obtain basis set info from DALTON.BAS """
        molecule = mol.readin(self.dalton_bas)
        self.cpa = mol.contracted_per_atom(molecule)
        self.cpa_l = mol.contracted_per_atom_l(molecule)
        self.opa = mol.occupied_per_atom(molecule)
        self.noa = len(self.opa)
#
# Total number of basis functions and occpied orbitals
#
        self.nbf = sum(self.cpa)
        self.noc = 0
        for o in self.opa:
            self.noc += len(o)

        if debug:
            print "Orbitals/atom", self.cpa, "\nTotal", self.nbf
            print "Occupied/atom", self.opa, "\nTotal", self.noc

    @property
    @verbose2
    def S(self):
        """
        Get overlap, nuclear charges and coordinates from AOONEINT
        """
        if self._S is None:
            self._S = one.read("OVERLAP", self.aooneint).unpack().unblock()
        return self._S
    #S = property(fget=get_overlap)
        

    def get_isordk(self):
        """
        Get overlap, nuclear charges and coordinates from AOONEINT
        """
        #
        # Data from the ISORDK section in AOONEINT
        #
        isordk = one.readisordk(filename=self.aooneint)
        #
        # Number of nuclei
        #
        N = isordk["nucdep"]
        #
        # MXCENT , Fix dimension defined in nuclei.h
        #
        mxcent = len(isordk["chrn"]) 
        #
        # Nuclear charges
        #
        self.Z = full.matrix((N,))
        self.Z[:] = isordk["chrn"][:N]
        #
        # Nuclear coordinates
        #
        R = full.matrix((mxcent*3,))
        R[:] = isordk["cooo"][:]
        self.R = R.reshape((mxcent, 3), order='F')[:N, :]
#
# Form Rc  molecular gauge origin, default nuclear center of charge
#
        if self.gc is None:
            self.Rc = self.Z*self.R/self.Z.sum()
        else: 
            self.Rc = numpy.array(self.gc).view(full.matrix)
       #
       # Bond center matrix and half bond vector
       #
        noa = self.noa
        self.Rab = full.matrix((noa, noa, 3))
        self.dRab = full.matrix((noa, noa, 3))
        for a in range(noa):
            for b in range(noa):
                self.Rab[a, b, :] = (self.R[a, :] + self.R[b, :])/2
                self.dRab[a, b, :] = (self.R[a, :] - self.R[b, :])/2

    @property
    def D(self, debug=False):
        """ 
        Density from SIRIFC 
        """
        if self._D is None:
            Di, Dv = dens.ifc(filename=self.sirifc)
            self._D = Di + Dv
            if debug:
                print "main:Di", Di
                print "main:Dv", Dv
                print "main:D&S", self._D&self.S
        return self._D


    @property
    def T(self, debug=False):
        """
        Generate loprop transformation matrix according to the
        following steps 
        Given atomic overlap matrix:
        1. orthogonalize in each atomic block
        2. a) Lowdin orthogonalize occupied subspace
           b) Lowdin orthogonalize virtual subspace
        3. project occupied out of virtual
        4. Lowdin orthogonalize virtual
        Input: overlap S (matrix)
               contracted per atom (list)
               occupied per atom (nested list)
        Returns: transformation matrix T
                 such that T+ST = 1 (unit) """

        if self._T is not None: return self._T

        S = self.S
        cpa = self.cpa
        opa = self.opa
    #
    # 1. orthogonalize in each atomic block
    #
        #t1=timing("step 1")
        if debug:
            print "Initial S", S
        nbf = S.shape[0]
        #
        # obtain atomic blocking
        #
        #assert(len(cpa) == len(opa))
        noa = len(opa)
        nocc = 0
        for at in range(noa):
            nocc += len(opa[at])
        if debug:
            print "nocc", nocc
        Satom = S.block(cpa, cpa)
        Ubl = full.unit(nbf).subblocked((nbf,), cpa)
        if debug:
            print "Ubl", Ubl
        #
        # Diagonalize atomwise
        #
        GS = 1
        if GS:
            T1 = blocked.matrix(cpa, cpa)
            for at in range(noa):
                T1.subblock[at] = Ubl.subblock[0][at].GST(S)
            if debug:
                print "T1", T1
            T1 = T1.unblock()
        else:
            u, v = Satom.eigvec()
            T1 = v.unblock()
        if debug:
            print "T1", T1
        #
        # Full transformration
        #
        S1 = T1.T*S*T1
        if debug:
            print "Overlap after step 1", S1
        #t1.stop()
       
        # 2. a) Lowdin orthogonalize occupied subspace
        #
        # Reorder basis (permute)
        #
        #t2=timing("step 2")
        vpa = []
        adim = []
        for at in range(noa):
            vpa.append(cpa[at]-len(opa[at]))
            adim.append(len(opa[at]))
            adim.append(vpa[at])
        if debug:
            print "Blocking: Ao Av Bo Bv...", adim
        #
        # dimensions for permuted basis
        #
        pdim = []
        if debug:
            print "opa", opa
        for at in range(noa):
            pdim.append(len(opa[at]))
        for at in range(noa):
            pdim.append(vpa[at])
        if debug:
            print "Blocking: Ao Bo... Av Bv...", pdim
        #
        # within atom permute occupied first
        #
        P1 = subblocked.matrix(cpa, cpa)
        for at in range(noa):
            P1.subblock[at][at][:, :] = full.permute(opa[at], cpa[at])
        n = len(adim)
        if debug:
            print "P1", P1
        P1 = P1.unblock()
       
       
        P2 = subblocked.matrix(adim, pdim)
        for i in range(0, len(adim), 2):
            P2.subblock[i][i/2] = full.unit(adim[i])
        for i in range(1, len(adim), 2):
            P2.subblock[i][noa+(i-1)/2] = full.unit(adim[i])
        if debug:
            print "P2", P2
        P2 = P2.unblock()
       
        #
        # new permutation scheme
        #
       
        P = P1*P2
        if debug:
            print "P", P
            if not numpy.allclose(P.inv(), P.T):
                print "P not unitary"
                sys.exit(1)
       
       
        S1P = P.T*S1*P
        if debug:
            print "Overlap in permuted basis", S1P
       
       
        #invsq=lambda x: 1.0/math.sqrt(x)
        occdim = (nocc, sum(vpa))
        S1Pbl = S1P.block(occdim, occdim)
        ### SYM ### S1Pbl += S1Pbl.T; S1Pbl *= 0.5 ###SYM###
        #T2bl=S1Pbl.func(invsq)
        T2bl = S1Pbl.invsqrt()
        T2 = T2bl.unblock()
       
       
        S2 = T2.T*S1P*T2
        if debug:
            print "Overlap after step 2", S2
        #t2.stop()
       
        #
        # Project occupied out of virtual
        #
        #t3=timing("step 3")
        if 0:
            T3 = full.unit(nbf).GST(S2)
        else:
            S2sb = S2.subblocked(occdim, occdim)
            T3sb = full.unit(nbf).subblocked(occdim, occdim)
            T3sb.subblock[0][1] = -S2sb.subblock[0][1]
            T3 = T3sb.unblock()
        S3 = T3.T*S2*T3
        #
        if debug:
            print "T3", T3
            print "Overlap after step 3", S3
        #t3.stop()
        #
        # 4. Lowdin orthogonalize virtual 
        #
        #t4=timing("step 4")
        T4b = blocked.unit(occdim)
        S3b = S3.block(occdim, occdim)
        if debug:
            print "S3b", S3b
            print "T4b", T4b
        ### SYM ### S3b += S3b.T; S3b *= 0.5 ###SYM###
        T4b.subblock[1] = S3b.subblock[1].invsqrt()
        T4 = T4b.unblock()
        S4 = T4.T*S3*T4
        #S4=S3
        if debug:
            print "T4", T4
            print "Overlap after step 4", S4
        #t4.stop()
       
        #
        # permute back to original basis
        #
        S4 = P*S4*P.T
        if debug:
            print "Final overlap ", S4
       
        #
        # Return total transformation
        #
        T = T1*P*T2*T3*T4*P.T
        #
        # Test
        #
        if debug:
            print "Transformation determinant", T.det()
            print "original S", S, "final", T.T*S*T
        self._T = T
        return self._T

    #T = property(fget=transformation)

    #def charge(self, debug=False):
    @property
    def Qab(self, debug=False):
        """ set charge/atom property"""
        if self._Qab is not None: return self._Qab

        D = self.D
        T = self.T
        cpa = self.cpa
        
        Ti = T.I
        Dlop = Ti*D*Ti.T
        Dlopsb = Dlop.subblocked(cpa, cpa)
        noa = self.noa
        qa = full.matrix((noa, noa))
        for a in range(noa):
            qa[a, a] = Dlopsb.subblock[a][a].tr()
        self._Qab = -qa
        return self._Qab

    #Qab = property(fget=charge)

    #def dipole(self, debug=False):
    @property
    def Dab(self, debug=False):
        """Set dipole property"""

        if self._Dab is not None: return self._Dab

        D = self.D
        T = self.T
        cpa = self.cpa
        Z = self.Z
        Rab = self.Rab
        Qab = self.Qab

        lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
       
        prp = os.path.join(self.tmpdir,"AOPROPER")
        nbf = D.shape[0]
        #x = [prop.read(l, prp).unpack() for l in lab]
        x = prop.read(*lab, filename=prp, unpack=True)
        #for l, xi in zip(lab, x): print l, xi
        xlop = [T.T*xi*T for xi in x]
        xlopsb = [xli.subblocked(cpa, cpa) for xli in xlop]

        Ti = T.I
        Dlop = Ti*D*Ti.T
        Dlopsb = Dlop.subblocked(cpa, cpa)
        if debug:
            print "dipole:Dlopsb", Dlopsb
        noa = len(cpa)
        dab = full.matrix((3, noa, noa))
        for i in range(3):
            for a in range(noa):
                for b in range(noa):
                    dab[i, a, b] = -(
                         xlopsb[i].subblock[a][b]&Dlopsb.subblock[a][b]
                         ) \
                         -Qab[a, b]*Rab[a, b, i]
        if debug:
            print "dipole:dab", dab
        
        self._Dab = dab
        return self._Dab

    #Dab = property(fget=dipole)

    #def dipole_sym(self):
    @property
    def Dsym(self):
        """Symmetrize density contributions from atom pairs """
        if self._Dsym is not None: return self._Dsym

        Dab = self.Dab
        noa = self.noa
        dsym = full.matrix((3, noa*(noa+1)//2))
        ab = 0
        for a in range(noa):
            for b in range(a):
                dsym[:, ab] = Dab[:, a, b] + Dab[:, b, a]
                ab += 1
            dsym[:, ab] = Dab[:, a, a]
            ab += 1

        self._Dsym = dsym
        return self._Dsym

    #Dsym = property(fget=dipole_sym)

    #def quadrupole(self, debug=False):
    @property
    def QUab(self, debug=False):
        """Quadrupole moment"""
        if self._QUab is not None: return self._QUab

        D = self.D
        T = self.T
        cpa = self.cpa
        Z = self.Z
        R = self.R
        Rc = self.Rc
        dRab = self.dRab
        Qab = self.Qab
        Dab = self.Dab

        lab = ("XXSECMOM", "XYSECMOM", "XZSECMOM", 
                           "YYSECMOM", "YZSECMOM", 
                                       "ZZSECMOM")


        prp = os.path.join(self.tmpdir,"AOPROPER")
        C = "XYZ"
        nbf = D.shape[0]
        #x = [prop.read(l, prp).unpack() for l in lab]
        x = prop.read(*lab, filename=prp, unpack=True)
        xlop = [T.T*xi*T for xi in x]
        xlopsb = [xli.subblocked(cpa, cpa) for xli in xlop]

        Ti = T.I
        Dlop = Ti*D*Ti.T
        Dlopsb = Dlop.subblocked(cpa, cpa)
        if debug:
            print "prop:Dlopsb", Dlopsb
        noa = len(cpa)
        QUab = full.matrix((6, noa, noa))
        rrab = full.matrix((6, noa, noa))
        rRab = full.matrix((6, noa, noa))
        RRab = full.matrix((6, noa, noa))
        for a in range(noa):
            for b in range(noa):
                Rab = (R[a, :]+R[b, :])/2
                # using b as expansion center
                #Rab=R[b,:]
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        #ij=i*(i+1)/2+j
                        #print "i j ij lab",C[i],C[j],ij,lab[ij]
                        rrab[ij, a, b] = -(
                            xlopsb[ij].subblock[a][b]&Dlopsb.subblock[a][b]
                            ) 
                        rRab[ij, a, b] = Dab[i, a, b]*Rab[j]+Dab[j, a, b]*Rab[i]
                        RRab[ij, a, b] = Rab[i]*Rab[j]*Qab[a, b]
                        ij += 1
        QUab = rrab-rRab-RRab
        #print "rrab", rrab
        self._QUab = QUab
        #
        # Addition term - gauge correction summing up bonds
        #
        dQUab = full.matrix(self.QUab.shape)
        for a in range(noa):
            for b in range(noa):
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        dQUab[ij, a, b] = dRab[a, b, i]*Dab[j, a, b] \
                                      +dRab[a, b, j]*Dab[i, a, b]
                        ij += 1
        if False:
            QUaa = QUab.sum(axis=2).view(full.matrix)
            dQUaa = dQUab.sum(axis=2).view(full.matrix)
            print 'QUab', QUab.T
            print "dQUab", dQUab.T
            print "dQUaa", dQUaa.T
            print "QUaa+dQUaa", (QUaa + dQUaa).T
            dQUsum = dQUaa.sum(axis=1).view(full.matrix)
            print "dQUsum", dQUsum
            print \
                "Electronic quadrupole moment by atomic domain, summed(B):",\
                " Q(A,B)", QUaa+dQUaa
        self.dQUab = - dQUab

        return self._QUab

    #QUab = property(fget=quadrupole)

    #def quadrupole_sym(self):
    @property
    def QUsym(self):
        """Quadrupole moment symmetrized over atom pairs"""
        if self._QUsym is not None: return self._QUsym

        QUab = self.QUab
        noa = self.noa
        qusym = full.matrix((6, noa*(noa+1)//2))
        ab = 0
        for a in range(noa):
            for b in range(a):
                qusym[:, ab] = QUab[:, a, b] + QUab[:, b, a]
                ab += 1
            qusym[:, ab] = QUab[:, a, a]
            ab += 1

        self._QUsym = qusym
        return self._QUsym

    #QUsym = property(fget=quadrupole_sym)

    #def nuclear_quadrupole(self):
    @property
    def QUN(self):
        """Nuclear contribution to quadrupole"""
        if self._QUN is not None: return self._QUN

        qn = full.matrix(6)
        Z = self.Z
        R = self.R
        Rc = self.Rc
        for a in range(len(Z)):
            ij = 0
            for i in range(3):
                for j in range(i, 3):
                    qn[ij] += Z[a]*(R[a, i]-Rc[i])*(R[a, j]-Rc[j])
                    ij += 1
        self._QUN = qn
        return self._QUN

    #QUN = property(fget=nuclear_quadrupole)

    @property
    def QUc(self):
        if self._QUc is not None: return self._QUc

        rrab=full.matrix((6, self.noa, self.noa))
        rRab=full.matrix((6, self.noa, self.noa))
        RRab=full.matrix((6, self.noa, self.noa))
        Rabc = 1.0*self.Rab
        for a in range(self.noa):
            for b in range(self.noa):
                Rabc[a,b,:] -= self.Rc
        for a in range(self.noa):
            for b in range(self.noa):
                ij = 0
                for i in range(3):
                    for j in range(i,3):
                        rRab[ij, a, b] = self.Dab[i, a, b]*Rabc[a, b, j]\
                                       + self.Dab[j, a, b]*Rabc[a, b, i]
                        RRab[ij, a, b] = self.Qab[a, b]*(self.R[a, i] - self.Rc[i])*(self.R[b, j] - self.Rc[j])
                        ij += 1
        QUcab = self.QUab + rRab + RRab
        self._QUc = QUcab.sum(axis=2).sum(axis=1).view(full.matrix)
        return self._QUc

    @property
    def Fab(self, **kwargs):
        """Penalty function"""
        if self._Fab is not None: return self._Fab

        Fab = full.matrix((self.noa, self.noa))
        for a in range(self.noa):
            Za = self.Z[a]
            Ra = self.R[a]
            for b in range(a):
                Zb = self.Z[b]
                Rb = self.R[b]
                Fab[a, b] = self.pf(Za, Ra, Zb, Rb, **kwargs)
                Fab[b, a] = Fab[a, b]
        for a in range(self.noa):
            Fab[a, a] += - Fab[a, :].sum()
        self._Fab = Fab
        return self._Fab
    #Fab = property(fget=get_Fab)

    @property
    def la(self):
        """Lagrangian for local poplarizabilities"""
        #
        # The shift should satisfy
        #   sum(a) sum(b) (F(a,b) + C)l(b) = sum(a) dq(a) = 0
        # =>sum(a, b) F(a, b) + N*C*sum(b) l(b) = 0
        # => C = -sum(a, b)F(a,b) / sum(b)l(b)
        #

        if self._la is not None: return self._la
        #
        dQa = self.dQa
        Fab = self.Fab
        Lab = Fab + self.sf(Fab)
        self._la = [rhs/Lab for rhs in dQa]
        return self._la
    #la = property(fget=get_la)

    #def get_linear_response_density(self):
    @property
    def Dk(self):
        """Read perturbed densities"""

        if self._Dk is None:
            lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
            prp = os.path.join(self.tmpdir,"AOPROPER")
            self._Dk = [[lr.Dk(l, freq=w, tmpdir=self.tmpdir) for l in lab] for w in self.freqs]

        return self._Dk

    #Dk = property(fget=get_linear_response_density)

    #def get_dipole_property(self):
    @property
    def x(self):
        """Read dipole matrices """

        if self._x is None:
            lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
            prp = os.path.join(self.tmpdir,"AOPROPER")
            #self._x = [prop.read(l, prp).unpack() for l in lab]
            self._x = prop.read(*lab, filename=prp, unpack=True)

        return self._x

    #x = property(fget=get_dipole_property)
        
    #def get_dQa(self):
    @property
    def dQa(self):
        """Charge shift per atom"""
        if self._dQa is not None: return self._dQa

        S = self.S
        T = self.T
        cpa = self.cpa
        noa = self.noa

        Dk = self.Dk
        print "Dk"
        Dklop = [[T.I*d*T.I.T for d in Dkw] for Dkw in Dk]
        Dklopsb = [[d.subblocked(cpa, cpa) for d in Dkwlop] for Dkwlop in Dklop]

        Slop = T.T*S*T
        Slopsb = Slop.subblocked(cpa, cpa)

        dQa = full.matrix((self.nfreqs, noa, 3))
        for a in range(noa):
            for i in range(3):
                for w in self.rfreqs:
                    dQa[w, a, i] = \
                  - Slopsb.subblock[a][a]&Dklopsb[w][i].subblock[a][a]
        self._dQa = dQa
        return self._dQa

    #dQa = property(fget = get_dQa)

    #def get_dQab(self):
    @property
    def dQab(self):
        """Charge transfer matrix"""
        if self._dQab is not None: return self._dQab

        dQa = self.dQa
        #la = self.dQa/self.Fab
        la = self.la
        noa = self.noa

        dQab = full.matrix((self.nfreqs, noa, noa, 3))
        for field in range(3):
            for a in range(noa):
                Za = self.Z[a]
                Ra = self.R[a]
                for b in range(a):
                    Zb = self.Z[b]
                    Rb = self.R[b]
                    for w in self.rfreqs:
                        dQab[w, a, b, field] =  \
                        - (la[w][a, field]-la[w][b, field]) * \
                        self.pf(Za, Ra, Zb, Rb)
                        dQab[w, b, a, field] = -dQab[w, a, b, field]

        self._dQab = dQab
        return self._dQab


    #dQab = property(fget = get_dQab)

    #def get_Aab(self, debug=False):
    @property
    def Aab(self):
        """Localized polariziabilities"""
        if self._Aab is not None: return self._Aab

        D = self.D
        Dk = self.Dk
        T = self.T
        cpa = self.cpa
        Z = self.Z
        Rab = self.Rab
        Qab = self.Qab
        dQa = self.dQa
        x = self.x

        #Transform property/density to loprop basis
        Slop = T.T*self.S*T
        xlop = [T.T*p*T for p in x]
        Dklop = [[T.I*d*T.I.T for d in Dkw] for Dkw in Dk]
        #to subblocked
        xlopsb = [p.subblocked(cpa, cpa) for p in xlop]
        Dklopsb = [[d.subblocked(cpa, cpa) for d in Dkwlop] for Dkwlop in Dklop]
        Slopsb = Slop.subblocked(cpa, cpa)
           
        noa = len(cpa)
        Aab = full.matrix((self.nfreqs, 3, 3, noa, noa))

        # correction term for shifting origin from O to Rab
        for i in range(3):
            for j in range(3):
                for a in range(noa):
                    for b in range(noa):
                        for w in self.rfreqs:
                            Aab[w, i, j, a, b] = (
                           -xlopsb[i].subblock[a][b]&Dklopsb[w][j].subblock[a][b]
                           )
                    for w in self.rfreqs:
                        Aab[w, i, j, a, a] -= dQa[w, a, j]*Rab[a, a, i]

        self._Aab = Aab
        return self._Aab

    #Aab = property(fget=get_Aab)

    #def get_dAab(self):
    @property
    def dAab(self):
        """Charge transfer contribution to bond polarizability"""
        if self._dAab is not None: return self._dAab

        dQa = self.dQa
        dQab = self.dQab
        dRab = self.dRab
        noa = self.noa
        dAab = full.matrix((self.nfreqs, 3, 3, noa, noa))
        for a in range(noa):
            for b in range(noa):
                for i in range(3):
                    for j in range(3):
                        if  mc:
                            dAab[:, i, j, a, b] = 2*dRab[a, b, i]*dQab[:, a, b, j]
                        else:
                            dAab[:, i, j, a, b] = (
                                dRab[a, b, i]*dQab[:, a, b, j]+
                                dRab[a, b, j]*dQab[:, a, b, i]
                                )
        self._dAab = dAab
        return self._dAab

    #dAab = property(fget=get_dAab)

    @property
    def Am(self):
        "Molecular polarizability"

        if self._Am is not None: return self._Am

        dQa = self.dQa
        Rab = self.Rab
        Aab = self.Aab
        noa = self.noa

        self._Am = Aab.sum(axis=4).sum(axis=3).view(full.matrix)
        for i in range(3):
            for j in range(3):
                for a in range(noa):
                    for w in self.rfreqs:
                        self._Am[w, i, j] += Rab[a, a, i]*dQa[w, a, j]
        return self._Am

    def output_by_atom(self, fmt="%9.5f", max_l=0, pol=0, bond_centers=False):
        """Print nfo"""

        if max_l >= 0: Qab = self.Qab
        if max_l >= 1:
            Dab = self.Dab
            Dsym = self.Dsym
        if max_l >= 2:
            QUab = self.QUab
            QUN = self.QUN
            dQUab = self.dQUab
        if  pol:
            Aab = self.Aab + self.dAab

        Z = self.Z
        R = self.R
        Rc = self.Rc
        noa = Qab.shape[0]
    #
    # Form net atomic properties P(a) = sum(b) P(a,b)
    #
        Qa = Qab.diagonal()
        if self._Dab is not None : Da = Dab.sum(axis=2)
        if self._QUab is not None : 
            QUa = QUab.sum(axis=2) + dQUab.sum(axis=2)
        if self._Aab is not None: 
            Aab = self.Aab + 0.5*self.dAab
            Aa = Aab.sum(axis=3)
        if bond_centers:
            for a in range(noa):
                for b in range(a):
                    header("Bond    %d %d" % (a+1, b+1))
                    print "Bond center:       " + \
                        (3*fmt) % tuple(0.5*(R[a, :]+R[b, :]))
                    print "Electronic charge:   "+fmt % Qab[a, b]
                    print "Total charge:        "+fmt % Qab[a, b]
                    if self._Dab is not None:
                        print "Electronic dipole    " + \
                            (3*fmt) % tuple(Dab[:, a, b]+Dab[:, b, a])
                        print "Electronic dipole norm" + \
                            fmt % (Dab[:, a, b]+Dab[:, b, a]).norm2()
                    if self._QUab is not None:
                        print "Electronic quadrupole" + \
                            (6*fmt) % tuple(QUab[:, a, b]+QUab[:, b, a])
                    if self._Aab is not None:
                        Asym = Aab[:, :, a, b] + Aab[:, :, b, a]
                        if pol > 0:
                            print "Isotropic polarizability", fmt % Asym.trace()
                        if pol > 1:
                            print "Polarizability       ", (6*fmt) % tuple(Asym.pack().view(full.matrix))
                header("Atom    %d"%(a+1))
                print "Atom center:       " + \
                    (3*fmt) % tuple(R[a,:])
                print "Nuclear charge:    "+fmt % Z[a]
                print "Electronic charge:   "+fmt % Qab[a, a]
                print "Total charge:        "+fmt % (Z[a]+Qab[a, a])
                if self._Dab is not None:
                    print "Electronic dipole    " + \
                        (3*fmt) % tuple(Dab[:, a, a])
                    print "Electronic dipole norm" + \
                        fmt % Dab[:, a, a].norm2()
                if self._QUab is not None:
                    print "Electronic quadrupole" + \
                        (6*fmt) % tuple(QUab[:, a, a])
                if self._Aab is not None:
                    Asym = Aab[:, :, a, a] 
                    if pol > 0:
                        print "Isotropic polarizability", fmt % (Asym.trace()/3)
                    if pol > 1:
                        print "Polarizability       ", (6*fmt) % tuple(Asym.pack().view(full.matrix))
        else:
            for a in range(noa):
                header("Atomic domain %d" % (a+1))
                line = " 0"
                print "Domain center:       "+(3*fmt) % tuple(R[a, :])
                line += (3*"%17.10f") % tuple(xtang*R[a, :])
                print "Nuclear charge:      "+fmt % Z[a]
                print "Electronic charge:   "+fmt % Qa[a]
                print "Total charge:        "+fmt % (Z[a]+Qa[a])
                line += "%12.6f" % (Z[a]+Qa[a])
                if self._Dab is not None:
                    print "Electronic dipole    "+(3*fmt) % tuple(Da[:, a])
                    line += (3*"%12.6f") % tuple(Da[:, a])
                if self._QUab is not None:
                    #print "QUab", QUab
                    print "Electronic quadrupole"+(6*fmt) % tuple(QUa[:, a])
                    line += (6*"%12.6f") % tuple(QUa[:, a])
                if self._Aab is not None:
                    Asym = Aa[:, :, a].view(full.matrix)
                    #print "Polarizability       ", Asym.pack()
                    print "Isotropic polarizablity"+fmt % (Aa[:, :, a].trace()/3)
                    print "Electronic polarizability" + \
                        (6*fmt) % tuple(Asym.pack().view(full.matrix))
    #
    # Total molecular properties
    #
        Ztot = Z.sum()
        Qtot = Qa.sum()
        if self._Dab is not None:
            Dm = Da.sum(axis=1) 
            Dc = Qa*(R-Rc)
            DT = Dm+Dc
        if self._QUab is not None:
            QUm = self.QUc
            QUT = QUm+QUN

        header("Molecular")
        print "Domain center:       "+(3*fmt) % tuple(Rc)
        print "Nuclear charge:      "+fmt % Ztot
        print "Electronic charge:   "+fmt % Qtot
        print "Total charge:        "+fmt % (Ztot+Qtot)
        if self._Dab is not None: 
            print "Electronic dipole    "+(3*fmt) % tuple(Dm)
            print "Gauge   dipole       "+(3*fmt) % tuple(Dc)
            print "Total   dipole       "+(3*fmt) % tuple(DT)
        if self._QUab is not None:
            print "Electronic quadrupole"+(6*fmt) % tuple(QUm)
            print "Nuclear    quadrupole"+(6*fmt) % tuple(QUN)
            print "Total      quadrupole"+(6*fmt) % tuple(QUT)
        if self._Aab is not None:
            Am = self.Am
            #print "Polarizability       ",Am
            print "Polarizability av    ", fmt % (Am.trace()/3)
            print "Polarizability       ", (6*fmt) % tuple(Am.pack().view(full.matrix))

    def output_potential_file(
            self, maxl, pol, bond_centers=False, angstrom=False, nfreq=1
            ):
        """Output potential file"""
        fmt = "%10.3f"
        lines = []
        if angstrom: 
            unit = "AA" 
        else: 
            unit = "AU"
        lines.append(unit)

        noa = self.noa
        if bond_centers:
            noc = noa*(noa + 1)/2
        else:
            noc = self.noa

        lines.append("%d %d %d %d"%(noc, maxl, pol, 1))

        if maxl >= 0: Qab = self.Qab
        if maxl >= 1: 
            Dab = self.Dab
            Dsym = self.Dsym
        if maxl >= 2: 
            QUab = self.QUab
            dQUab = self.dQUab
        if pol > 0: Aab = self.Aab + 0.5*self.dAab

        if bond_centers:
            ab = 0
            for a in range(noa):
                for b in range(a):
                    line  = ("1" + 3*fmt) % tuple(self.Rab[a, b, :])
                    if maxl >= 0: line += fmt % Qab[a, b]
                    if maxl >= 1: line += (3*fmt) % tuple(Dsym[:, ab])
                    if maxl >= 2: line += (6*fmt) % \
                        tuple(QUab[:, a, b] +QUab[:, b, a])
                    if pol > 0:
                        Asym = Aab[:, :, a, b] + Aab[:, :, b, a]
                        if pol == 1: line += fmt % Asym.trace()
                        if pol == 2: 
                            line += (6*fmt)%tuple(Asym.pack().view(full.matrix))
                    ab += 1
                        

                    lines.append(line)

                line  = ("1" + 3*fmt) % tuple(self.Rab[a, a, :])
                if maxl >= 0: line += fmt % (self.Z[a]+Qab[a, a])
                if maxl >= 1: line += (3*fmt) % tuple(Dsym[:, ab])
                if maxl >= 2: line += (6*fmt) % tuple(QUab[:, a, a])
                if pol > 0:
                    Asym = Aab[:, :, a, a]
                    if pol == 1: line += fmt % (Asym.trace()/3)
                    if pol == 2: 
                        line += (6*fmt) % tuple(Asym.pack().view(full.matrix))
                ab += 1
                    
                lines.append(line)
        else:
            for a in range(noa):
                line  = ("1" + 3*fmt) % tuple(self.Rab[a, a, :])
                if maxl >= 0: line += fmt % (self.Z[a] + Qab[a, a])
                if maxl >= 1: line += (3*fmt) % tuple(Dab.sum(axis=2)[:, a])
                if maxl >= 2: 
                    line += (6*fmt) % tuple((QUab+dQUab).sum(axis=2)[:,  a])
                if pol > 0:
                    for iw in range(nfreq):
                        Asym = Aab.sum(axis=4)[iw, :, :, a].view(full.matrix)
                        if pol == 1: 
                            line += fmt % (Asym.trace()/3)
                        if pol == 2: 
                            line += (6*fmt) % tuple(Asym.pack().view(full.matrix))
                    
                lines.append(line)
            

        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import optparse

    OP = optparse.OptionParser()
    OP.add_option(
          '-d', '--debug',
          dest='debug', action='store_true', default=False,
          help='print for debugging [False]'
          )
    OP.add_option(
          '-v', '--verbose',
          dest='verbose', action='store_true', default=False,
          help='print details [False]'
          )
    OP.add_option(
          '-t','--tmpdir',
          dest='tmpdir', default='/tmp',
          help='scratch directory [/tmp]'
          )
    OP.add_option(
          '-f','--daltgz',
          dest='daltgz', default=None,
          help='Dalton restart tar ball [None]'
          )
    OP.add_option(
          '-p', '--potfile',
          dest='potfile', default='LOPROP.POT',
          help='Potential input file [LOPROP.POT]'
          )
    OP.add_option(
          '-b','--bond',
          dest='bc', action='store_true',default=False,
          help='include bond centers [False]'
          )
    OP.add_option(
          '-g','--gauge-center',
          dest='gc', default=None,
          help='gauge center'
          )

    OP.add_option(
          '-l', '--angular-momentum',
          dest='max_l', type='int', default=2,
          help='Max angular momentum [2]'
          )

    OP.add_option(
          '-A', '--Anstrom',
          dest='angstrom', action='store_true', default=False,
          help="Output in Angstrom"
          )

    OP.add_option(
          '-a','--polarizabilities',
          dest='pol', type='int', default=0,
          help='Localized polarizabilities (1=isotropic, 2=full)'
          )

    OP.add_option(
          '-s','--screening (alpha)',
          dest='alpha', type='float', default=2.0,
          help='Screening parameter for penalty function'
          )

    o, a = OP.parse_args(sys.argv[1:])

    #
    # Check consistency: present Dalton files
    #
    if not os.path.isdir(o.tmpdir):
        print "%s: Directory not found: %s" % (sys.argv[0], o.tmpdir)
        raise SystemExit

    import tarfile
    if o.daltgz:
        tgz = tarfile.open(o.daltgz, 'r:gz')
        tgz.extractall(path=o.tmpdir)
        
    needed_files = ["AOONEINT", "DALTON.BAS", "SIRIFC", "AOPROPER", "RSPVEC"]
    for file_ in needed_files:
        df = os.path.join(o.tmpdir, file_)
        if not os.path.isfile(df):
            print "%s: %s does not exists" % (sys.argv[0], df)
            print "Needed Dalton files to run loprop.py:"
            print "\n".join(needed_files)
            raise SystemExit

    if o.gc is not None: 
        #Gauge center
        try:
            #gc = map(float, o.gc.split())
            gc = [float(i) for i in o.gc.split()]
        except(ValueError):
            sys.stderr.write("Gauge center incorrect:%s\n" % o.gc)
            sys.exit(1)
    else:
        gc = None


    t = timing.timing('Loprop')
    molfrag = MolFrag(
        o.tmpdir, pf=penalty_function(o.alpha), gc=gc
        )
    print molfrag.output_potential_file(
        o.max_l, o.pol, o.bc, o.angstrom
        )
        
        
    if o.verbose:
        molfrag.output_by_atom(fmt="%9.5f", max_l=o.max_l, pol=o.pol, bond_centers=o.bc)
    print t
     
