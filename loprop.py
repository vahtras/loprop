#!/usr/bin/env python
#
# Loprop model implementation (J. Chem. Phys. 121, 4494 (2004))
#
import os, sys, math, numpy, pdb
from daltools import one, mol, dens, prop, lr
from util import full, blocked, subblocked, timing

full.matrix.fmt="%14.6f"
xtang = 0.5291772108
angtx = 1.0/xtang
mc = False

rbs=numpy.array([0, 
      0.25,                                     0.25, 
      1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45
      ])*angtx

def penalty_function(alpha=2):
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

def shift_function(*args):
    F, = args
    return 2*numpy.max(numpy.abs(F))

def header(str):
    border='-'*len(str)
    print "\n%s\n%s\n%s"%(border,str,border)


class MolFrag:
    """An instance of the MolFrag class is created and populated with
    data from a Dalton runtime scratch directory"""

    def __init__(self, tmpdir, pf=penalty_function, sf=shift_function, gc=None, maxl=2, pol=False, debug=False):
        """Constructur of MolFrac class objects
        input: tmpdir, scratch directory of Dalton calculation
        """
        self.tmpdir = tmpdir
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
        self._QUab = None
        self._QUN = None
        self._dQa = None
        self._dQab = None
        self._Fab = None
        self._la = None
        self._Aab = None
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
            print "Orbitals/atom",cpa,"\nTotal",nbf
            print "Occupied/atom",opa,"\nTotal",noc

    def get_overlap(self):
        """
        Get overlap, nuclear charges and coordinates from AOONEINT
        """
        if self._S is None:
            self._S = one.read("OVERLAP", self.aooneint).unpack().unblock()
        return self._S
    S = property(fget=get_overlap)
        

    def get_isordk(self):
        """
        Get overlap, nuclear charges and coordinates from AOONEINT
        """
        #
        # Data from the ISORDK section in AOONEINT
        #
        isordk = one.readisordk(file=self.aooneint)
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
        self.R = R.reshape((mxcent,3),order='F')[:N,:]
#
# Form Rc  molecular gauge origin, default nuclear center of charge
#
        if self.gc is None:
            self.Rc=self.Z*self.R/self.Z.sum()
        else: 
            self.Rc = numpy.array(self.gc).view(full.matrix)
       #
       # Bond center matrix and half bond vector
       #
        noa = self.noa
        self.Rab=full.matrix((noa, noa, 3))
        self.dRab=full.matrix((noa, noa, 3))
        for a in range(noa):
           for b in range(noa):
              self.Rab[a, b, :] = (self.R[a,:] + self.R[b,:])/2
              self.dRab[a, b, :] = (self.R[a,:] - self.R[b,:])/2

    def get_density(self, debug=False):
        """ 
        Density from SIRIFC 
        """
        if self._D is None:
            Di, Dv = dens.ifc(filename=self.sirifc)
            self._D = Di + Dv
            if debug:
                print "main:Di",Di
                print "main:Dv",Dv
                print "main:D&S",D&S
        return self._D
    D = property(fget=get_density)


    def transformation(self, debug=False):
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
           print "Initial S",S
        nbf=S.shape[0]
        #
        # obtain atomic blocking
        #
        #assert(len(cpa) == len(opa))
        noa=len(opa)
        nocc=0
        for at in range(noa):
           nocc+=len(opa[at])
        if debug:
           print "nocc",nocc
        Satom=S.block(cpa,cpa)
        Ubl=full.unit(nbf).subblocked((nbf,),cpa)
        if debug:
           print "Ubl",Ubl
        #
        # Diagonalize atomwise
        #
        GS=1
        if GS:
           T1=blocked.matrix(cpa,cpa)
           for at in range(noa):
              T1.subblock[at]=Ubl.subblock[0][at].GST(S)
           if debug:
              print "T1",T1
           T1=T1.unblock()
        else:
           u,v=Satom.eigvec()
           T1=v.unblock()
        if debug:
           print "T1",T1
        #
        # Full transformration
        #
        S1=T1.T*S*T1
        if debug:
           print "Overlap after step 1",S1
        #t1.stop()
       
        # 2. a) Lowdin orthogonalize occupied subspace
        #
        # Reorder basis (permute)
        #
        #t2=timing("step 2")
        vpa=[]
        adim=[]
        for at in range(noa):
           vpa.append(cpa[at]-len(opa[at]))
           adim.append(len(opa[at]))
           adim.append(vpa[at])
        if debug:
           print "Blocking: Ao Av Bo Bv...",adim
        #
        # dimensions for permuted basis
        #
        pdim=[]
        if debug:
           print "opa",opa
        for at in range(noa):
           pdim.append(len(opa[at]))
        for at in range(noa):
           pdim.append(vpa[at])
        if debug:
           print "Blocking: Ao Bo... Av Bv...",pdim
        #
        # within atom permute occupied first
        #
        P1=subblocked.matrix(cpa,cpa)
        for at in range(noa):
           P1.subblock[at][at][:,:]=full.permute(opa[at],cpa[at])
        n=len(adim)
        if debug:
           print "P1",P1
        P1=P1.unblock()
       
       
        P2=subblocked.matrix(adim,pdim)
        for i in range(0,len(adim),2):
           P2.subblock[i][i/2]=full.unit(adim[i])
        for i in range(1,len(adim),2):
           P2.subblock[i][noa+(i-1)/2]=full.unit(adim[i])
        if debug:
           print "P2",P2
        P2=P2.unblock()
       
        #
        # new permutation scheme
        #
       
        P=P1*P2
        if debug:
           print "P",P
           if not numpy.allclose(P.inv(),P.T):
              print "P not unitary"
              sys.exit(1)
       
       
        S1P=P.T*S1*P
        if debug:
           print "Overlap in permuted basis",S1P
       
       
        #invsq=lambda x: 1.0/math.sqrt(x)
        occdim=(nocc,sum(vpa))
        S1Pbl=S1P.block(occdim,occdim)
        ### SYM ### S1Pbl += S1Pbl.T; S1Pbl *= 0.5 ###SYM###
        #T2bl=S1Pbl.func(invsq)
        T2bl=S1Pbl.invsqrt()
        T2=T2bl.unblock()
       
       
        S2=T2.T*S1P*T2
        if debug:
           print "Overlap after step 2",S2
        #t2.stop()
       
        #
        # Project occupied out of virtual
        #
        #t3=timing("step 3")
        if 0:
           T3=full.unit(nbf).GST(S2)
        else:
           S2sb=S2.subblocked(occdim,occdim)
           T3sb=full.unit(nbf).subblocked(occdim,occdim)
           T3sb.subblock[0][1]=-S2sb.subblock[0][1]
           T3=T3sb.unblock()
        S3=T3.T*S2*T3
        #
        if debug:
           print "T3",T3
           print "Overlap after step 3",S3
        #t3.stop()
        #
        # 4. Lowdin orthogonalize virtual 
        #
        #t4=timing("step 4")
        T4b=blocked.unit(occdim)
        S3b=S3.block(occdim,occdim)
        if debug:
           print "S3b",S3b
           print "T4b",T4b
        ### SYM ### S3b += S3b.T; S3b *= 0.5 ###SYM###
        T4b.subblock[1]=S3b.subblock[1].invsqrt()
        T4=T4b.unblock()
        S4=T4.T*S3*T4
        #S4=S3
        if debug:
           print "T4",T4
           print "Overlap after step 4",S4
        #t4.stop()
       
        #
        # permute back to original basis
        #
        S4=P*S4*P.T
        if debug:
           print "Final overlap ",S4
       
        #
        # Return total transformation
        #
        T=T1*P*T2*T3*T4*P.T
        #
        # Test
        #
        if debug:
           print "Transformation determinant",T.det()
           print "original S",S,"final",T.T*S*T
        if False:
           print t1
           print t2
           print t3
           print t4
        self._T = T
        return self._T

    T = property(fget=transformation)

    def charge(self,debug=False):
       
        if self._Qab is not None: return self._Qab

        S = self.S
        D = self.D
        T = self.T
        cpa = self.cpa
        
        """Input: 
              overlap S, 
              density D, 
              transformation T,
              contracted per atom cpa (list)
           Returns:
              matrix with atomic and bond charges
              for an loprop transformation T this is diagonal
        """
        #
        Ti=T.I
        if debug:
           print "charge:Inverse transformation",Ti
        if 1:
           Slop=T.T*S*T
           Dlop=Ti*D*Ti.T
        elif 0: #dumb tests
           Slop=T.T*S*T
           Dlop=Ti*D*T
        else: #in loprop article
           Slop=Ti*S*T
           Dlop=Ti*D*T
        if debug:
           print "charge:Slop",Slop
           print "charge:Dlop",Dlop
           print "charge:Dlop&Slop",Dlop&Slop
           #print "charge:Dlop",Dlop
        Slopsb=Slop.subblocked(cpa,cpa)
        Dlopsb=Dlop.subblocked(cpa,cpa)
        if debug:
           print "charge:Slopsb",Slopsb
           print "charge:Dlopsb",Dlopsb
        noa=len(cpa)
        qa=full.matrix((noa,noa))
        for a in range(noa):
           qa[a,a]=Slopsb.subblock[a][a]&Dlopsb.subblock[a][a]
           if debug:
               for b in range(a):
                   qa[a,b]=Slopsb.subblock[a][b]&Dlopsb.subblock[a][b]
                   qa[b,a]=Slopsb.subblock[b][a]&Dlopsb.subblock[b][a]
        self._Qab = -qa
        return self._Qab

    Qab = property(fget=charge)

    def dipole(self, debug=False):

        if self._Dab is not None: return self._Dab

        D = self.D
        T = self.T
        cpa = self.cpa
        Z = self.Z
        Rab = self.Rab
        Qab = self.Qab

        lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
       
        prp=os.path.join(self.tmpdir,"AOPROPER")
        nbf=D.shape[0]
        x=[]
        xlop=[]
        xlopsb=[]
        for i in range(len(lab)):
           x.append(prop.read(nbf,lab[i],prp).unpack())
           xlop.append(T.T*x[i]*T)
           xlopsb.append(xlop[i].subblocked(cpa,cpa))
           if debug:
              print "dipole:",lab[i],xlopsb[i]
        #from pdb import set_trace; set_trace()
        Ti=T.I
        Dlop=Ti*D*Ti.T
        Dlopsb=Dlop.subblocked(cpa,cpa)
        if debug:
           print "dipole:Dlopsb",Dlopsb
        noa=len(cpa)
        dab=full.matrix((3, noa, noa))
        for i in range(3):
           for a in range(noa):
              for b in range(noa):
                 dab[i,a,b]=-(
                      xlopsb[i].subblock[a][b]&Dlopsb.subblock[a][b]
                      ) \
                      -Qab[a,b]*Rab[a,b,i]
        if debug:
           print "dipole:dab",dab
        
        self._Dab = dab
        return self._Dab

    Dab = property(fget=dipole)

    def quadrupole(self, debug=False):

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


        prp=os.path.join(self.tmpdir,"AOPROPER")
        C="XYZ"
        nbf=D.shape[0]
        x=[]
        xlop=[]
        xlopsb=[]
        for i in range(len(lab)):
           x.append(prop.read(nbf,lab[i],prp).unpack())
           xlop.append(T.T*x[i]*T)
           xlopsb.append(xlop[i].subblocked(cpa,cpa))
           if debug:
              print lab[i],xlopsb[i]
        Ti=T.I
        Dlop=Ti*D*Ti.T
        Dlopsb=Dlop.subblocked(cpa,cpa)
        if debug:
           print "prop:Dlopsb",Dlopsb
        noa=len(cpa)
        QUab=full.matrix((6, noa,noa))
        rrab=full.matrix((6, noa,noa))
        rRab=full.matrix((6, noa,noa))
        RRab=full.matrix((6, noa,noa))
        for a in range(noa):
           for b in range(noa):
              Rab=(R[a,:]+R[b,:])/2
              # using b as expansion center
              #Rab=R[b,:]
              ij=0
              for i in range(3):
                 for j in range(i,3):
                    #ij=i*(i+1)/2+j
                    #print "i j ij lab",C[i],C[j],ij,lab[ij]
                    rrab[ij,a,b]=-(xlopsb[ij].subblock[a][b]&Dlopsb.subblock[a][b]) 
                    rRab[ij,a,b]=Dab[i,a,b]*Rab[j]+Dab[j,a,b]*Rab[i]
                    RRab[ij,a,b]=Rab[i]*Rab[j]*Qab[a,b]
                    ij+=1
        QUab=rrab-rRab-RRab
        #print "rrab", rrab
        self._QUab = QUab
        #
        # Addition term - gauge correction summing up bonds
        #
        dQUab = full.matrix(self.QUab.shape)
        for a in range(noa):
           for b in range(noa):
              ij=0
              for i in range(3):
                 for j in range(i,3):
                    dQUab[ij, a,b] = dRab[a,b,i]*Dab[j,a,b] \
                                  +dRab[a,b,j]*Dab[i,a,b]
                    ij += 1
        if False:
            QUaa=QUab.sum(axis=2).view(full.matrix)
            dQUaa=dQUab.sum(axis=2).view(full.matrix)
            print 'QUab', QUab.T
            print "dQUab",dQUab.T
            print "dQUaa",dQUaa.T
            print "QUaa+dQUaa", (QUaa + dQUaa).T
            dQUsum=dQUaa.sum(axis=1).view(full.matrix)
            print "dQUsum",dQUsum
            print "Electronic quadrupole moment by atomic domain, summed(B): Q(A,B)",QUaa+dQUaa
        self.dQUab = - dQUab

        return self._QUab

    QUab = property(fget=quadrupole)

    def nuclear_quadrupole(self):
        """Nuclear contribution to quadrupole"""
        if self._QUN is not None: return self._QUN

        qn=full.matrix(6)
        Z = self.Z
        R = self.R
        Rc = self.Rc
        for a in range(len(Z)):
           ij=0
           for i in range(3):
              for j in range(i,3):
                 qn[ij]+=Z[a]*(R[a,i]-Rc[i])*(R[a,j]-Rc[j])
                 ij+=1
        self._QUN = qn
        return self._QUN

    QUN = property(fget=nuclear_quadrupole)


    def get_Fab(self, **kwargs):
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
    Fab = property(fget=get_Fab)

    def get_la(self):
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
        self._la = dQa/Lab
        return self._la
    la = property(fget=get_la)

    def get_linear_response_density(self):
        """Read perturbed densities"""

        if self._Dk is None:
            lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
            prp=os.path.join(self.tmpdir,"AOPROPER")
            Dk = []
            for l in lab:
                Dk.append(lr.Dk(l, tmpdir=self.tmpdir))
            self._Dk = Dk

        return self._Dk

    Dk = property(fget=get_linear_response_density)

    def get_dipole_property(self):
        """Read dipole matrices """

        if self._x is None:
            lab = ['XDIPLEN', "YDIPLEN", "ZDIPLEN"]
            prp=os.path.join(self.tmpdir,"AOPROPER")
            x = []
            for l in lab:
                x.append(prop.read(self.nbf, l, prp).unpack())
            self._x = x

        return self._x

    x = property(fget=get_dipole_property)
        
    def get_dQa(self):

        if self._dQa is not None: return self._dQa

        S = self.S
        T = self.T
        cpa = self.cpa
        noa = self.noa

        Dk = self.Dk
        Dklop = []
        for d in Dk:
           Dklop.append(T.I*d*T.I.T)

        Dklopsb=[]
        for d in Dklop:
            Dklopsb.append(d.subblocked(cpa, cpa))

        Slop = T.T*S*T
        Slopsb = Slop.subblocked(cpa, cpa)

        dQa = full.matrix((noa, 3))
        for a in range(noa):
            for i in range(3):
                dQa[a, i] = \
              - Slopsb.subblock[a][a]&Dklopsb[i].subblock[a][a]
        self._dQa = dQa
        return self._dQa

    dQa = property(fget = get_dQa)

    def get_dQab(self):
        if self._dQab is not None: return self._dQab

        dQa = self.dQa
        #la = self.dQa/self.Fab
        la = self.la
        noa = self.noa

        dQab = full.matrix((noa, noa, 3))
        #from pdb import set_trace; set_trace()
        for field in range(3):
            for a in range(noa):
                Za = self.Z[a]
                Ra = self.R[a]
                for b in range(a):
                    Zb = self.Z[b]
                    Rb = self.R[b]
                    dQab[a, b, field] =  \
                        - (la[a, field]-la[b, field]) * \
                        self.pf(Za, Ra, Zb, Rb)
                    dQab[b, a, field] = -dQab[a, b, field]

        self._dQab = dQab
        return self._dQab


    dQab = property(fget = get_dQab)

    def get_Aab(self, debug=False):

        if self._Aab is not None: return self._Aab

        #from pdb import set_trace; set_trace()
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
        xlop=[]
        Dklop = []
        Slop = T.T*self.S*T
        for p in x:
           xlop.append(T.T*p*T)
        for d in Dk:
           Dklop.append(T.I*d*T.I.T)
        #to subblocked
        xlopsb=[]
        Dklopsb=[]
        Slopsb = Slop.subblocked(cpa, cpa)
        for p in xlop:
            xlopsb.append(p.subblocked(cpa, cpa))
        for d in Dklop:
            Dklopsb.append(d.subblocked(cpa, cpa))
           
        noa=len(cpa)
        Aab=full.matrix((3, 3, noa,noa))

        if debug:
            print "full charge shift ", dQab
            print "atomic charge shift", dQa
            print "verify dQ=0", dQ

        # correction term for shifting origin from O to Rab
        for i in range(3):
            for j in range(3):
               for a in range(noa):
                  for b in range(noa):
                     Aab[i,j,a,b]= (
                        -xlopsb[i].subblock[a][b]&Dklopsb[j].subblock[a][b]
                        )
                  Aab[i,j,a,a] -= dQa[a, j]*Rab[a, a, i]

        self._Aab = Aab
        return self._Aab

    Aab = property(fget=get_Aab)

    def get_dAab(self):
        
        if self._dAab is not None: return self._dAab

        dQa = self.dQa
        dQab = self.dQab
        dRab = self.dRab
        noa = self.noa
        dAab = full.matrix((3, 3, noa, noa))
        for a in range(noa):
            for b in range(noa):
                for i in range(3):
                    for j in range(3):
                        if  mc:
                            dAab[i, j, a, b] = 2*dRab[a, b, i]*dQab[a, b, j]
                        else:
                            dAab[i, j, a, b] = (
                                dRab[a, b, i]*dQab[a, b, j]+
                                dRab[a, b, j]*dQab[a, b, i]
                                )
        self._dAab = dAab
        return self._dAab

    dAab = property(fget=get_dAab)

    def output_by_atom(self, fmt="%9.5f", bond_centers=False):
        Qab = self.Qab
        Dab = self.Dab
        QUab = self.QUab
        QUN = self.QUN
        dQUab = self.dQUab
        Aab = self.Aab
        Z = self.Z
        R = self.R
        Rc = self.Rc
        noa=Qab.shape[0]
    #
    # Form net atomic properties P(a) = sum(b) P(a,b)
    #
        Qa=Qab.diagonal()
        if Dab is not None : Da=Dab.sum(axis=2)
        if QUab is not None : 
            QUa = QUab.sum(axis=2) + dQUab.sum(axis=2)
        if Aab: Aa=Aab.sum(axis=2)
        if bond_centers:
            for i in range(noa):
                for j in range(i+1):
                    if i == j:
                        header("Atom    %d"%(j+1))
                        print "Atom center:       "+(3*fmt)%tuple(R[i,:])
                        print "Nuclear charge:    "+fmt%Z[i]
                        print "Electronic charge:   "+fmt%Qab[i,i]
                        print "Total charge:        "+fmt%(Z[i]+Qab[i,i])
                        if Dab is not None:
                            print "Electronic dipole    "+(3*fmt)%tuple(Dab[:, i,i])
                            print "Electronic dipole norm"+fmt%Dab[:, i,i].norm2()
                        if QUab is not None:
                            print "Electronic quadrupole"+(6*fmt)%tuple(QUab[:, i,i])
                        if Aab is not None:
                            #print "Polarizability       ",Aab[i,i,:,:]
                            #print "Polarizability       ",(9*fmt)%tuple(Aab[i,i,:,:].flatten('F'))
                            print "Polarizability       ",(3*fmt)%(
                                  Aab[i,i,0,0],Aab[i,i,1,1],Aab[i,i,2,2]
                                  )
                            print Aab[i,i,:,:]
                            print fmt%(Aab[i,i,:,:].trace()/3)
                    else:
                        header("Bond    %d %d"%(i+1,j+1))
                        print "Bond center:       "+(3*fmt)%tuple(0.5*(R[i,:]+R[j,:]))
                        print "Electronic charge:   "+fmt%Qab[i,j]
                        print "Total charge:        "+fmt%Qab[i,j]
                        if Dab is not None:
                            print "Electronic dipole    "+(3*fmt)%tuple(Dab[:, i,j]+Dab[:, j,i])
                            print "Electronic dipole norm"+fmt%(Dab[:, i,j]+Dab[:, j,i]).norm2()
                        if QUab is not None:
                            print "Electronic quadrupole"+(6*fmt)%tuple(QUab[:, i,j]+QUab[:, j,i])
                        if Aab is not None:
                            print "Polarizability       ",(3*fmt)%(
                                  Aab[i,j,0,0]+Aab[j,i,0,0],
                                  Aab[i,j,1,1]+Aab[j,i,1,1],
                                  Aab[i,j,2,2]+Aab[j,i,2,2]
                                  )
                            print Aab[i,j,:,:]+Aab[j,i,:,:]
                            print fmt%((Aab[i,j,:,:]+Aab[j,i,:,:]).trace()/3)
        else:
            for i in range(noa):
                header("Atomic domain %d"%(i+1))
                line=" 0"
                print "Domain center:       "+(3*fmt)%tuple(R[i,:])
                line+=(3*"%17.10f")%tuple(xtang*R[i,:])
                print "Nuclear charge:      "+fmt%Z[i]
                print "Electronic charge:   "+fmt%Qa[i]
                print "Total charge:        "+fmt%(Z[i]+Qa[i])
                line+="%12.6f"%(Z[i]+Qa[i])
                if Dab is not None:
                    print "Electronic dipole    "+(3*fmt)%tuple(Da[i,:])
                    line+=(3*"%12.6f")%tuple(Da[i,:])
                if QUab is not None:
                    print "Electronic quadrupole"+(6*fmt)%tuple(QUa[i,:])
                    line+=(6*"%12.6f")%tuple(QUa[i,:])
                if Aab is not None:
                    print "Polarizability       ",Aa[i,:,:].view(full.matrix).sym()
                    print "Electronic polarizability"+(9*fmt)%tuple(Aa[i,:,:].flatten())
                    line+=(6*"%12.6f")%(Aa[i,0,0],Aa[i,0,1],Aa[i,0,2],Aa[i,1,1],Aa[i,1,2],Aa[i,2,2])
                    print "Isotropic            "+fmt%(Aa[i,:,:].trace()/3)
    #
    # Total molecular properties
    #
        Ztot=Z.sum()
        Qtot=Qa.sum()
        if Dab is not None:
            Dm=Da.sum(axis=0) 
            Dc=Qa*(R-Rc)
            DT=Dm+Dc
        if QUab is not None:
            QUm=QUa.sum(axis=0)
            Rab=full.matrix((noa,noa,3))
            for a in range(noa):
                for b in range(noa):
                    #Rab[a,b,:]=(R[a,:]+R[b,:])/2-Rc[:]
                    Rab[a,b,:]=R[b,:]-Rc[:]
                    ij=0
                    for i in range(3):
                        for j in range(i,3):
                            QUm[ij]+=Dab[i, a,b]*Rab[a,b,j] + Dab[j, a,b]*Rab[a,b,i]
                            QUm[ij]+=Qab[a,b]*Rab[a,b,i]*Rab[a,b,j]
                            ij+=1
            QUT=QUm+QUN
        if Aab is not None: Am=Aa.sum(axis=0)

        header("Molecular")
        print "Domain center:       "+(3*fmt)%tuple(Rc)
        print "Nuclear charge:      "+fmt%Ztot
        print "Electronic charge:   "+fmt%Qtot
        print "Total charge:        "+fmt%(Ztot+Qtot)
        if Dab is not None: 
            print "Electronic dipole    "+(3*fmt)%tuple(Dm)
            print "Gauge   dipole       "+(3*fmt)%tuple(Dc)
            print "Total   dipole       "+(3*fmt)%tuple(DT)
        if QUab is not None:
            print "Electronic quadrupole"+(6*fmt)%tuple(QUm)
            print "Nuclear    quadrupole"+(6*fmt)%tuple(QUN)
            print "Total      quadrupole"+(6*fmt)%tuple(QUT)
        if Aab is not None:
            #print "Polarizability       ",Am
            print "Polarizability av    ",fmt%(Am.trace()/3)
            print "Polarizability       ",(9*fmt)%tuple(Am.flatten('F'))

    def output_potential_file(self, potfile, maxl, pol, bond_centers, angstrom):
        """Output potential file"""
        fmt = "%6.2f"
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
        if maxl >= 1: Dab = self.Dab
        if maxl >= 2: QUab = self.QUab
        if pol > 0: Aab = self.Aab + 0.5*self.dAab

        if bond_centers:
            for a in range(noa):
                for b in range(a):
                    line  = ("1" + 3*fmt)%tuple(self.Rab[a, b,:])
                    if maxl >= 0: line += fmt%Qab[a, b]
                    if maxl >= 1: line += (3*fmt)%tuple(Dab[:, a, b] + Dab[:, b, a])
                    if maxl >= 2: line += (6*fmt)%tuple(QUab[:, a, b] +QUab[:, b, a])
                    if pol > 0:
                        Asym = Aab[:, :, a, b] + Aab[:, :, b, a]
                        if pol == 1: line += fmt%Asym.trace()
                        if pol == 2: line += (6*fmt)%tuple(Asym.pack().view(full.matrix))
                        
                    lines.append(line)
                pass
                line  = ("1" + 3*fmt)%tuple(self.Rab[a, a,:])
                if maxl >= 0: line += fmt%(self.Z[a]+Qab[a, a])
                if maxl >= 1: line += (3*fmt)%tuple(Dab[:, a, a])
                if maxl >= 2: line += (6*fmt)%tuple(QUab[:, a, a])
                if pol > 0:
                    Asym = Aab[:, :, a, a]
                    if pol == 1: line += fmt%Asym.trace()
                    if pol == 2: line += (6*fmt)%tuple(Asym.pack().view(full.matrix))
                    
                lines.append(line)
        else:
            for a in range(noa):
                line  = ("1" + 3*fmt)%tuple(self.Rab[a, a,:])
                if maxl >= 0: line += fmt%(self.Z[a] + Qab[a, a])
                if maxl >= 1: line += (3*fmt)%tuple(Dab.sum(axis=2)[:, a])
                if maxl >= 2: line += (6*fmt)%tuple(QUab.sum(axis=2)[:,  a])
                if pol > 0:
                    Asym = Aab.sum(axis=3)[:, :, a].view(full.matrix)
                    if pol == 1: line += fmt%Asym.trace()
                    if pol == 2: line += (6*fmt)%tuple(Asym.pack().view(full.matrix))
                    
                lines.append(line)
            pass

        print "\n".join(lines)


if __name__ == "__main__":
    import sys, os, optparse

    OP = optparse.OptionParser()
    OP.add_option(
          '-d', '--debug',
          dest='debug', action='store_true', default=False,
          help='print for debugging [False]'
          )
    OP.add_option(
          '-t','--tmpdir',
          dest='tmpdir', default='/tmp',
          help='scratch directory [/tmp]'
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

    o,a=OP.parse_args(sys.argv[1:])

    #
    # Check consistency: present Dalton files
    #
    if not os.path.isdir(o.tmpdir):
        print "%s: Directory not found: %s"%(sys.argv[0], o.tmpdir)
        raise SystemExit
        
    needed_files=["AOONEINT","DALTON.BAS","SIRIFC","AOPROPER","RSPVEC","LUINDF"]
    for file_ in needed_files:
        df = os.path.join(o.tmpdir, file_)
        if not os.path.isfile(df):
           print "%s: %s does not exists"%(sys.argv[0], df)
           print "Needed Dalton files to run loprop.py:"
           print "\n".join(needed_files)
           raise SystemExit

    if o.gc is not None: 
       #Gauge center
       try:
          gc = map(float,o.gc.split())
       except:
          sys.stderr.write("Gauge center incorrect:%s\n"%o.gc)
          sys.exit(1)
    else:
       gc = None

    #main(debug=o.debug,tmpdir=o.tmpdir,potfile=o.potfile,bond_centers=o.bc,pf=o.pf, gc=gc, maxl=o.l, pol=o.alpha)

    t = timing.timing('Loprop')
    molfrag = MolFrag(o.tmpdir, pf=penalty_function(o.alpha), gc=gc, debug=o.debug )
    molfrag.output_potential_file(o.potfile, o.max_l, o.pol, o.bc, o.angstrom)
        
        
    #molfrag.output_by_atom(fmt="%9.5f", bond_centers=o.bc)
    print t
     
