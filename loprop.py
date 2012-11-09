#!/usr/bin/env python
#
# Loprop model implementation (J. Chem. Phys. 121, 4494 (2004))
#
import sys,math,numpy
from daltools import one, mol, dens
from util import full,blocked,subblocked,timing
full.matrix.fmt="%14.6f"
xtang = 0.5291772108
angtx = 1.0/xtang
rbs=numpy.array([0, 
      0.25,                                     0.25, 
      1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45
      ])*angtx

def transformation(S,cpa,opa,debug=False):
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


   invsq=lambda x: 1.0/math.sqrt(x)
   occdim=(nocc,sum(vpa))
   S1Pbl=S1P.block(occdim,occdim)
   ### SYM ### S1Pbl += S1Pbl.T; S1Pbl *= 0.5 ###SYM###
   T2bl=S1Pbl.func(invsq)
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
   T4b.subblock[1]=S3b.subblock[1].func(invsq)
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
   return T

def charge(S,D,T,cpa,debug=False):
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
   if 1:
      # only diagonal blocks are nonzero in loprop
      for i in range(noa):
         qa[i,i]=Slopsb.subblock[i][i]&Dlopsb.subblock[i][i]
   else:
      for i in range(noa):
         for j in range(noa):
            qa[i,j]=Slopsb.subblock[i][j]&Dlopsb.subblock[i][j]
   return -qa

def dipole(lab,D,T,cpa,Z,Rab,Qab,debug=False,tmpdir='/tmp'):
   #
   import os
   #
   prp=os.path.join(tmpdir,"AOPROPER")
   import prop
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
   Ti=T.I
   Dlop=Ti*D*Ti.T
   Dlopsb=Dlop.subblocked(cpa,cpa)
   if debug:
      print "dipole:Dlopsb",Dlopsb
   noa=len(cpa)
   dab=full.matrix((noa,noa,len(lab)))
   for i in range(len(lab)):
      for a in range(noa):
         for b in range(noa):
            dab[a,b,i]=-(xlopsb[i].subblock[a][b]&Dlopsb.subblock[a][b]) \
                 -Qab[a,b]*Rab[a,b,i]
   if debug:
      print "dipole:dab",dab
   return dab

def quadrupole(lab,D,T,cpa,Z,R,Qab,Dab,debug=0,tmpdir='/tmp'):
   #
   import os
   #
   prp=os.path.join(tmpdir,"AOPROPER")
   C="XYZ"
   import prop
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
   quab=full.matrix((noa,noa,len(lab)))
   rrab=full.matrix((noa,noa,len(lab)))
   rRab=full.matrix((noa,noa,len(lab)))
   RRab=full.matrix((noa,noa,len(lab)))
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
               rrab[a,b,ij]=-(xlopsb[ij].subblock[a][b]&Dlopsb.subblock[a][b]) 
               rRab[a,b,ij]=Dab[a,b,i]*Rab[j]+Dab[a,b,j]*Rab[i]
               RRab[a,b,ij]=Rab[i]*Rab[j]*Qab[a,b]
               ij+=1
   quab=rrab-rRab-RRab
   return quab

def nuclear_quadrupole(Z,R,Rc):
   nao=len(Z)
   qn=full.matrix(6)
   for a in range(nao):
      ij=0
      for i in range(3):
         for j in range(i,3):
            qn[ij]+=Z[a]*(R[a,i]-Rc[i])*(R[a,j]-Rc[i])
            ij+=1
   return qn

def header(str):
    border='-'*len(str)
    print "\n%s\n%s\n%s"%(border,str,border)

def output_by_atom(Qab,Dab,QUab,dQUab,Aab,Z,R,Rc,potfile,fmt="%9.5f",bond_centers=False):
    noa=Qab.shape[0]
#
# Potential file
#
    potfile=open(potfile,'w')
    potfile.write("AA\n")
    if bond_centers:
        potfile.write("%d "%(noa*(noa+1)/2)) # all bonds for now
    else:
        potfile.write("%d "%noa)
    maxl = 0
    if Dab is not None: maxl = 1
    if QUab is not None: maxl = 2
    potfile.write("%d "%maxl)
    if Aab is not None:
        potfile.write("2 1") # get correct 
    potfile.write("\n")
#
# Form net atomic properties P(a) = sum(b) P(a,b)
#
    Qa=Qab.diagonal()
    if Dab is not None : Da=Dab.sum(axis=0)
    if QUab is not None : 
        QUa = QUab.sum(axis=0) + dQUab.sum(axis=0)
    if Aab: Aa=Aab.sum(axis=0)
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
                        print "Electronic dipole    "+(3*fmt)%tuple(Dab[i,i,:])
                        print "Electronic dipole norm"+fmt%Dab[i,i,:].norm2()
                    if QUab is not None:
                        print "Electronic quadrupole"+(6*fmt)%tuple(QUab[i,i,:])
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
                        print "Electronic dipole    "+(3*fmt)%tuple(Dab[i,j,:]+Dab[j,i,:])
                        print "Electronic dipole norm"+fmt%(Dab[i,j,:]+Dab[j,i,:]).norm2()
                    if QUab is not None:
                        print "Electronic quadrupole"+(6*fmt)%tuple(QUab[i,j,:]+QUab[j,i,:])
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
            potfile.write(line+'\n')
        potfile.close()
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
                        QUm[ij]+=Dab[a,b,i]*Rab[a,b,j] + Dab[a,b,j]*Rab[a,b,i]
                        QUm[ij]+=Qab[a,b]*Rab[a,b,i]*Rab[a,b,j]
                        ij+=1
        QUN=nuclear_quadrupole(Z,R,Rc)
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

def main(debug=False, tmpdir='/tmp', potfile="LOPROP.POT", bond_centers=False, pf=0,
         gc=None, maxl=2, pol=True):
    """Main Loprop driver:
    Input:
    debug: Print debug info [False]
    tmpdir: Dalton scratch directory containing runtime files ['/tmp']
    potfile: Potential file to be read by Dalton QM/MM
    pf: Penalty function for localized polarizabilities: 0:gaussian n>0:r^n
    gc: Gauge center as list input [None: Get nuclear center of charge]]
    maxl: Max angular momentum [2(quadrupole)]
    pol: Localized polarizabilties from response vectors
    """
    import os
    import socket
    from util.timing import timing
    t=timing('loprop')
#
# Dalton scratch files
#
    aooneint=os.path.join(tmpdir,'AOONEINT')
    dalton_bas=os.path.join(tmpdir,'DALTON.BAS')
    sirifc=os.path.join(tmpdir,'SIRIFC')
#
# Obtain basis set info
#
    S = one.read("OVERLAP", aooneint).unpack().unblock()
    molecule = mol.readin(dalton_bas)
    cpa = mol.contracted_per_atom(molecule)
    cpa_l = mol.contracted_per_atom_l(molecule)
    opa = mol.occupied_per_atom(molecule)
    noa = len(opa)
#
# charge,coor from AOONEINT
#
    isordk = one.readisordk(file=aooneint)
    N = isordk["nucdep"]
    Mx = len(isordk["chrn"])
    Z = full.matrix((N,))
    Z[:] = isordk["chrn"][:N]
    R = full.matrix((Mx*3,))
    R[:] = isordk["cooo"][:]
    R = R.reshape((Mx,3),order='F')[:N,:]
#
# Form Rc  molecular gauge origin, default nuclear center of charge
#
    if gc is None:
        Rc=Z*R/Z.sum()
    else: 
        Rc = numpy.array(gc).view(full.matrix)
   #
   # Bond center matrix and half bond vector
   #
    Rab=full.matrix((noa,noa,3))
    DRab=full.matrix((noa,noa,3))
    for a in range(noa):
       for b in range(noa):
          Rab[a,b,:]=(R[a,:]+R[b,:])/2
          DRab[a,b,:]=(R[a,:]-R[b,:])/2
# 
# Density from SIRIFC
#
    import dens
    Di, Dv = dens.ifc(filename=sirifc)
    D = Di + Dv
    if debug:
        print "main:Di",Di
        print "main:Dv",Dv
        print "main:D&S",D&S
#
# Total number of basis functions and occpied orbitals
#
    nbf = sum(cpa)
    noc = 0
    for i in range(N):
        #print "%d(%d)"%(i,N)
        noc += len(opa[i])

    if debug:
        print "Number of atoms",N
        print "Nuclear charges",Z
        print "Coordinates/a.u",R
        print "Orbitals/atom",cpa,"\nTotal",nbf
        print "Occupied/atom",opa,"\nTotal",noc
        print "Molecular gauge origin", Rc
#
# Loprop transformation matrix
#
    t1=timing("transformation")
    T=transformation(S,cpa,opa,debug=debug)
    print t1
#
# Local charges
#
    t2 = timing("charge")
    Qab = charge(S,D,T,cpa,debug=debug)
    print t2
    Qaa = Qab.diagonal()
    
    if debug:
        print "Electronic Charge/atom (Loprop)",Qab,"\nElectronic Charge/total",sum(Qaa)
        print "Net charge (Loprop)",Z+Qaa
        print "Net total charge %f"%sum(Z+Qaa)
        print "Expansion point: nuclear center of charge",Rc
#
# Local dipole moments
#
    if maxl > 0:
        lab=("XDIPLEN","YDIPLEN","ZDIPLEN")
        t3=timing("dipole")
        Dab=dipole(lab,D,T,cpa,Z,Rab,Qab,debug=False,tmpdir=tmpdir)
        print t3
        if False:
           print "Electronic dipole moment mu(A,B,x)",Dab
           print "Electronic dipole moment by atomic domain"
           Daa=Dab.sum(axis=0).view(full.matrix)
           print Daa
    else:
        Dab = None
#
# Quadrupole
#
    if maxl > 1:
        lab=("XXSECMOM","XYSECMOM","XZSECMOM","YYSECMOM","YZSECMOM","ZZSECMOM")
        t4=timing("quadrupole")
        QUab=quadrupole(lab,D,T,cpa,Z,R,Qab,Dab,debug=False,tmpdir=tmpdir)
        print t4
        QUaa=QUab.sum(axis=0).view(full.matrix)
        #
        # Addition term - gauge correction summing up bonds
        #
        dQUab = 0*QUab
        for a in range(noa):
           for b in range(noa):
              ij=0
              for i in range(3):
                 for j in range(i,3):
                    dQUab[a,b,ij] += DRab[a,b,i]*Dab[a,b,j]+DRab[a,b,j]*Dab[a,b,i]
                    ij += 1
        dQUaa=dQUab.sum(axis=0).view(full.matrix)
        if False:
            print 'QUab', QUab
            print "dQUab",dQUab
            print "dQUaa",dQUaa
            print "QUaa+dQUaa", QUaa + dQUaa
            dQUsum=dQUaa.sum(axis=0).view(full.matrix)
            print "dQUsum",dQUsum
            print "Electronic quadrupole moment by atomic domain, summed(B): Q(A,B)",QUaa+dQUaa
    else:
        QUab = None
#
# Polarizability (as dipole with perturbed density)
#
    if pol:
        if bond_centers:
            print "Bond penalty function %d"%pf
            dRab=full.matrix((noa,noa,3)) #atom-atom matrix of bond vectors
            for a in range(noa):
                for b in range(noa):
                    dRab[a,b,:]=R[a,:]-R[b,:]
            print "dRab",dRab
            #
            # deifne penalty function
            #
            def Fab(dR,a,b,n=0):
                import math
                print 'Fab',a,b,dR[a,b,:]
                rab=dR[a,b,:].norm2()
                if n == 0: #default BS radius
                    #alpha = 7.1421297
                    alpha = 2
                    ra=rbs[int(Z[a])]
                    rb=rbs[int(Z[b])]
                    print "rab2", rab**2
                    print "(ra+rb)2", (ra+rb)**2
                    #return math.exp(alpha*(rab/(ra+rb))**2)
                    #reproduce molcas bug
                    return math.exp(alpha*(rab/(ra+rb)/xtang)**2)
                else:
                    return rab**n

            #
            # Set bond vector matrix and Lagrangian matrix
            #
            Lab=full.matrix((noa,noa))
            dQab=full.matrix((noa,noa))
            Qab.diagonal()
            for a in range(noa):
                for b in range(a):
                    Lab[a,b] = 1/(2*Fab(dRab,a,b,pf)) 
                    Lab[b,a] = Lab[a,b]
            for a in range(noa):
                Lab[a,a] = 0
                Lab[a,a] = -Lab[a,:].sum() 
            print "Lab", Lab
            #print "absmax", abs(Lab).max()
           
            #Shift
            C=2*abs(Lab).max()
            Lab += C
            print "Shifted", Lab
            print "Inverted", Lab.inv()
           
            #Lab+=1.0
            print "Lab + eigenvalues",Lab,Lab.eig()
            print "rbs",rbs
           
            import lr
            lab=("XDIPLEN","YDIPLEN","ZDIPLEN")
            aab=full.matrix((noa,noa,3,3))
            Qabk=full.matrix((noa,noa,3))
            t5=timing("polarizability")
            #
            # Loop over perturbation fields i.e. response vectors
            #
            for i in range(3):
                Dk=lr.Dk(lab[i],tmpdir=tmpdir)
                #print "Dk",Dk
                Qabk[:,:,i]=charge(S,Dk,T,cpa,debug=debug)
                dkab=dipole(lab,Dk,T,cpa,Z,Rab,Qabk[:,:,i],debug=0,tmpdir=tmpdir)
                aab[:,:,:,i]=dkab
                dQa=Qabk[:,:,i].diagonal()
                #print "dQa",dQa
                if bond_centers:
                   #
                   # Charge transfer matrix
                   #
                   try:
                      print "dQa", dQa
                      lmult  = dQa/Lab
                      print "lmult (lambda)", lmult
                   except:
                      print "Linear Algebra Error"
                      print "i", i
                      print "dQa",dQa
                      print "Lab",Lab, 'det', Lab.det()
                      sys.exit(1)
                   delta=1e-3
                   for a in range(noa):
                      for b in range(a+1,noa):
                         dQab[a,b] = -(lmult[a]-lmult[b])*Lab[a,b]
                         if 0:
                            dQab[a,b] = (dQa[a] - dQa[b])/noa
                         dQab[b,a] = -dQab[a,b]
                          
                      try:
                          assert abs(dQab[a,:].sum()-dQa[a]) < delta
                      except(AssertionError):
                          print "dQab[%d,:]"%a, dQab[a,:].sum()
                          print "dQa[%d]"%a, dQa[a]
                          raise SystemExit
                   #print "dQab",dQab
                   for a in range(noa):
                      for b in range(noa):
                         aab[a,b,:,i]+=dQab[a,b]*dRab[a,b,:]/2
                         #aab[a,b,i,:]+=dQab[a,b]*(R[a]-Rc)
                else:
                    #
                    # Only total charge shift matters; add to diagonal
                    # (revert subtraction in function dipole)
                    #
                    for a in range(noa):
                       if 1:
                          #aab[a,a,:,i]+=Qabk[a,a,i]*Rab[a,a,:]
                          #aab[a,a,:,i]+=dQa[a]*Rab[a,a,:]
                          pass
                       else:
                          for b in range(noa):
                             aab[a,b,:,i]+=Qabk[a,b,i]*Rab[a,b,:]
                    pass
                if True:
                    for j in range(i,3):
                       print "Electronic polarizabillity <<%s:%s>>(A,B)"%(lab[j],lab[i]),dkab[:,:,j]
                    da=dkab.sum(axis=1)
                    print "Electronic polarizability by atomic domain, sum(A): alhpa(A,B,x)",da

        #print "Qabk",Qabk
        print "aab", aab
        print "a -atom"
        for i in range(noa):
           ai = aab[i,i,:,:]
           print i, ai.sym()
        print "a -bond"
        for i in range(noa):
           for j in range(i+1, noa):
               aij = aab[i,j,:,:]
               print i,j,aij.sym()
        print t5
    else:
        aab = None
#
# Results
#
    output_by_atom(Qab,Dab,QUab,dQUab,aab,Z,R,Rc,potfile,fmt="%9.5f",bond_centers=bond_centers)
    print
    print t
    print "loprop finished normally on host :",socket.gethostname()

if __name__ == "__main__":
   import sys,os,optparse
   #
   # Check consistency: present Dalton files
   #
   OP=optparse.OptionParser()
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
         '-f','--penalty-function',
         dest='pf', type='int', default=0,
         help='penalty function'
         )
   OP.add_option(
         '-g','--gauge-center',
         dest='gc', default=None,
         help='gauge center'
         )

   OP.add_option(
         '-l', '--angular-momentum',
         dest='l', type='int', default=2,
         help='Max angular momentum [2]'
         )

   OP.add_option(
         '-a','--polarizabilities',
         dest='alpha', action='store_true', default=False,
         help='Localized polarizabilities [False]'
         )

   o,a=OP.parse_args(sys.argv[1:])

   needed_files=["AOONEINT","DALTON.BAS","SIRIFC","AOPROPER","RSPVEC","LUINDF"]
   try:
      for file_ in needed_files:
         f=open(os.path.join(o.tmpdir,file_),'r')
         f.close()
   except IOError:
      print "%s does not exists"%os.path.join(o.tmpdir,file_)
      print "Needed Dalton files to run loprop.py:"
      print "\n".join(needed_files)
      sys.exit(-1)

   if o.gc is not None: 
      #Gauge center
      try:
         gc = map(float,o.gc.split())
      except:
         sys.stderr.write("Gauge center incorrect:%s\n"%o.gc)
         sys.exit(1)
   else:
      gc = None

   main(debug=o.debug,tmpdir=o.tmpdir,potfile=o.potfile,bond_centers=o.bc,pf=o.pf, gc=gc, maxl=o.l, pol=o.alpha)
