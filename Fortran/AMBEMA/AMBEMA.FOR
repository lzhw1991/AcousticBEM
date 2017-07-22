C***************************************************************
C           Subroutine AMBEMA by Stephen Kirkup                     
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AMBEMA.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This subroutine computes the modal solutions to the axisymmetric
C Helmholtz equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C in the domain interior to a closed axisymmetric boundary.
C
C The boundary (S) is defined (approximated) by a set of straight line 
C elements. The domain of the equation is within the boundary.
C
C The nature of the boundary condition may be Dirichlet, Robin or 
C Neumann and it is specified by the boundary functions {\alpha} and
C {\beta}. It is assumed to have the following general form
C
C         {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = 0
C    
C where {\alpha} and {\beta} are complex-valued functions defined on S.
C Important examples are {\alpha}=1, {\beta}=0 which is equivalent to a
C Dirichlet  boundary condition and {\alpha}=0, {\beta}=1 which is 
C equivalent to a Neumann boundary condition. 
C
C
C How to use the subroutine
C -------------------------
C
C The following diagram shows how the subroutine is to be used. A main
C program is required.
C
C                                   ....................................
C                                   :                                  :
C                                   :                                  :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|      AMBEMA            |   :
C      |(e.g. ambema_t.for) |       :     |                        |   :
C      |                    |       :     --------------------------   :
C      ----------------------       :                 |                :
C                                   :                 >                :
C                                   :                 |                :
C                                   :      ------------------------    :
C          Package ---------------->:      | subordinate routines |    :
C                                   :      ------------------------    :
C                                   :                                  :
C                                   :      (this file)                 :  
C                                   :..................................:
C                                  /         |                 |
C                               |_           >                 >
C                              /             |                 |
C             ................       ................   ................  
C             :              :       :   --------   :   :  --------    : 
C             : (geom2d.for) :---<---:   | H3ALC |  :   :  | CGLS |    : 
C             :              :       :   --------   :   :  --------    :  
C             :..............:       : -------------:   : -------------:  
C                                    : |subordinate|:   : |subordinate|: 
C             ................       : | routines  |:   : | routines  |:  
C             :              :       : -------------:   : -------------: 
C             : (geom3d.for) :---<---:              :   :              : 
C             :              :       : (h3alc.for)  :   : (cgls.for)   :
C             :..............:       :..............:   :..............:
C                                    
C
C The contents of the main program must be linked to AMBEMA.FOR, 
C H3ALC.FOR, CGLS.FOR, GEOM2D.FOR and GEOM3D.FOR.
C
C Method of solution
C ------------------
C 
C In the main program, the boundary must be described as a set of
C  elements. The elements are defined by two indices (integers) which
C  label a node or vertex on the boundary generator (the (r,z)
C  coordinate). The data structure VERTEX lists and enumerates the 
C  (r,z) coordinates of the vertices, the data structure SELV defines 
C  each element by indicating the labels for the two edge nodes on the
C  generator and hence enumerates the elements.
C The boundary solution points (the points on the boundary at which 
C  {\phi} (SPHI) and d {\phi}/dn (SVEL) are returned) are at the centres
C  of the generator of the elements. The boundary functions {\alpha} 
C  (SALPHA), {\beta} (SBETA) and f (SF) are also defined by their values
C  at the centres of the elements.
C Normally a solution in the domain is required. By listing the (r,z)
C  coordinates of all the interior points in PINT, the subroutine
C  returns the value of {\phi} at these points in PIPHI.
C
C

C Notes on the geometric parameters
C ---------------------------------
C (1) Each of the vertices listed in VERTEX must be distinct points
C  with respect to EGEOM.
C (2) The boundary must be complete and closed. Thus 
C  SELV(i,2)=SELV(i+1,1) for i=1..NSE-1 and VERTEX(SELV(1,1),1)=0
C  and VERTEX(SELV(NSE,2),1)=0.
C (3) The indices of the nodes listed in SELV must be such that they
C  are ordered counter-clockwise around the generator of the boundary.
C (4) The generator of the largest element must be no more than 10x 
C  the length of the generator of the smallest element.

C Notes on the interior points 
C ----------------------------
C (1) The points in PINT should lie within the boundary, as defined
C  by the parameters VERTEX and SELV. Any point lying outside the 
C  boundary will return a corresponding value in PIPHI that is near
C  zero.

C Notes on the boundary condition
C -------------------------------
C (1) For each i=1..NSE, it must not be the case that both of SALPHA(i)
C  and SBETA(i) are zero

C External modules in external files
C ==================================
C subroutine H3ALC: Returns the individual discrete Helmholtz integral
C  operators. (in file H3ALC.FOR)
C subroutine INTEIG: Solves a general linear system of equations. 
C  (in file INTEIG.FOR)
C real function DIST2: Returns the distance between two 2-vectors. (in
C  file GEOM2D.FOR)

C External modules provided in the package (this file)
C ====================================================
C subroutine GL8: Returns the points and weights of the 8-point Gauss-
C  Legendre quadrature rule.
C real function FNSQRT(X): real X : Returns the square root of X.
C real function FNEXP(Z): real X : Returns the complex exponential of Z.


C The subroutine

      SUBROUTINE AMBEMA(KA,KB,MAXNK,NK,
     *                 MAXNEIG,
     *                 MAXNV,NV,VERTEX,MAXNSE,NSE,SELV,
     *                 MAXNPI,NPI,PINT,
     *                 SALPHA,SBETA,
     *                 LVALID,EGEOM,
     *                 NEIG,EIGVAL,SPHI,SVEL,PIPHI,
     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,
     *                 WKSPC5,WKSPC6,WKSPC7,
     *                 WKSP00,WKSP01,WKSP02,WKSP03,WKSP04,WKSP05,
     *                 WKSP06,WKSP07,WKSP08,WKSP09,WKSP10,WKSP11,
     *                 WKSP12)

      PARAMETER (MAXNGQ=32)
      PARAMETER (MAXNTQ=10000)

C  Wavenumber interpolation information
C   Lower limit of k-range
      REAL*8     KA
C   Upper limit of k-range
      REAL*8     KB
C   Limit on the number of interpolation points in the k-range
      INTEGER    MAXNK
C   Number of interpolation points in the k-range
      INTEGER    NK

C  Eigenvalue information
C   Limit on the number of eigenvalues determined in the k-range
      INTEGER    MAXNEIG

C  Boundary geometry
C   Limit on the number of vertices on S
      INTEGER    MAXNV
C   The number of vertices on S
      INTEGER    NV
C   The coordinates of the vertices on S
      REAL*8     VERTEX(MAXNV,2)
C   Limit on the number of elements describing S
      INTEGER    MAXNSE
C   The number of elements describing S
      INTEGER    NSE
C   The indices of the vertices describing each element
      INTEGER    SELV(MAXNSE,2)
      
C  Interior points at which the solution is to be observed
C   Limit on the number of points interior to the boundary where 
C    solution is sought
      INTEGER    MAXNPI
C   The number of interior points
      INTEGER    NPI
C   Coordinates of the interior points
      REAL*8     PINT(MAXNPI,2)

C  The boundary condition is such that {\alpha} {\phi} + {\beta} v = 0
C  where alpha, beta are complex valued functions over S.
C  The functions are set values at the collocation points.
C   function alpha
      COMPLEX*16 SALPHA(MAXNSE)
C   function beta
      COMPLEX*16 SBETA(MAXNSE)

C  Validation and control parameters
      LOGICAL    LVALID
      REAL*8     EGEOM

C  Solution 
C   Number of eigenvalues determined in the k-range
      INTEGER    NEIG
C   eigenvalues
      COMPLEX*16     EIGVAL(MAXNEIG)
C   function phi
      COMPLEX*16 SPHI(MAXNEIG,MAXNSE)
C   function vel
      COMPLEX*16 SVEL(MAXNEIG,MAXNSE)
C   domain solution
      COMPLEX*16 PIPHI(MAXNEIG,MAXNPI)

C  Working space 
      COMPLEX*16 WKSPC1(MAXNK,MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC2(MAXNK,MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC3(MAXNK,MAXNSE,MAXNSE)
      LOGICAL    WKSPC4(MAXNSE)

C  External function
      REAL*8     DIST2

      REAL*8     K
      COMPLEX*16 CK

C  Constants
C   Real scalars: 0, 1, 2, half, pi
      REAL*8 ZERO,ONE,TWO,HALF,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Geometrical description of the boundary
C   Elements counter
      INTEGER    ISE,JSE
C   The points interior to the boundary where the solution is sought 
      INTEGER    IIP
C   Parameters for H2LC
      REAL*8     P(2),PA(2),PB(2),QA(2),QB(2),VECP(2)
      LOGICAL    LPONEL


C  Quadrature rule information
C   [Note that in this program two quadrature rules are used: one for
C    the case when the point P lies on the element (LPONEL=.TRUE.) and
C    one for the case when P does not lie on the element.]
C   Quadrature rule used when LPONEL=.TRUE.
C    Number of quadrature points
      INTEGER    NGQON
C    Abscissae of the actual quadrature rule
      REAL*8     AGQON(MAXNGQ)
C    Weights of the actual quadrature rule
      REAL*8     WGQON(MAXNGQ)
C   Quadrature rule used when LPONEL=.FALSE.
C    Number of quadrature points
      INTEGER    NGQOFF
C    Abscissae of the actual quadrature rule
      REAL*8     AGQOFF(MAXNGQ)
C    Weights of the actual quadrature rule
      REAL*8     WGQOFF(MAXNGQ)

C   Quadrature rule parameters for H3ALC
C   Counter through the quadrature points
      INTEGER    IGQ
C    Abscissae of the actual quadrature rule in the theta direction
      REAL*8     AGQ(MAXNGQ)
C    Weights of the actual quadrature rule
      REAL*8     WGQ(MAXNGQ)
C    Abscissae of the actual quadrature rule in the theta direction
      REAL*8     ATQ(MAXNTQ)
C    Weights of the actual quadrature rule
      REAL*8     WTQ(MAXNTQ)


C  Validation and control parameters for subroutine H2LC
      LOGICAL    LVAL
      REAL*8     EK
      REAL*8     EQRULE
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C  Parameters for subroutine H2LC. 
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

      REAL*8     WKSPCE(2*MAXNTQ+MAXNGQ)

C  Parameters for subroutine INTEIG
C   The k-coordiate of the interpolation matrices
      REAL*8 WKSPC5(MAXNK)

C The list of eigenvectors. EIGVCT(i,j) gives the value of the 
C  j-th component of the i-th eigenvector. The eigenvectors are
C  normalised
      COMPLEX*16 WKSPC6((MAXNK-1)*MAXNSE,MAXNSE)
      LOGICAL    WKSPC7(MAXNSE)

C Work space
C ----------
      COMPLEX*16 WKSP00((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      COMPLEX*16 WKSP01((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      COMPLEX*16 WKSP02((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      REAL*8     WKSP03((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      REAL*8     WKSP04((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      REAL*8     WKSP05((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      REAL*8     WKSP06((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      REAL*8     WKSP07((MAXNK-1)*MAXNSE)
      REAL*8     WKSP08((MAXNK-1)*MAXNSE)
      REAL*8     WKSP09(MAXNK)
      INTEGER    WKSP10((MAXNK-1)*MAXNSE)
      COMPLEX*16 WKSP11((MAXNK-1)*MAXNSE,(MAXNK-1)*MAXNSE)
      COMPLEX*16 WKSP12((MAXNK-1)*MAXNSE)

C  Other variables
C   Failure flag
      LOGICAL    LFAIL
C   Accumulation of solution {\phi}
      COMPLEX*16 SUMPHI
C   The `diameter' of the boundary or the maximum distance between any
C    two vertices
      REAL*8     DIAM
     
      REAL*8     RADMID,SGLEN,GLEN,CIRMID,TDIV
      INTEGER    NDIV
      REAL*8     SUMMK


C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      HALF=ONE/TWO
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

C Validation
C ==========


C Validation of parameters of AMBEMA
C ---------------------------------

C Find the diameter DIAM of the boundary
      DIAM=0.0
      DO 100 IV=1,NV-1
        PA(1)=VERTEX(IV,1)
        PA(2)=VERTEX(IV,2)
        DO 110 JV=IV+1,NV
          PB(1)=VERTEX(JV,1)
          PB(2)=VERTEX(JV,2)
          DIAM=MAX(DIAM,DIST2(PA,PB))
110     CONTINUE
100   CONTINUE


C Set the k-values of the interpolation points
      DO 200 IKPT=1,NK
        WKSPC5(IKPT)=(KA+KB)/2.0D0-
     *   COS((2*IKPT-1)*PI/DFLOAT(2*NK))*(KB-KA)/2.0D0
200   CONTINUE


C Set up validation and control parameters
C  Switch off the validation of H3ALC
      LVAL=.FALSE.
C  Set EK
      EK=1.0D-6
C  Set EQRULE
      EQRULE=1.0D-6


C Set up the quadrature rule(s).
C  Set up quadrature rule for the case when P is not on the element.
C   Set up 8 point Gauss-Legendre rules
      CALL GL8(MAXNGQ,NGQOFF,WGQOFF,AGQOFF)
C  Set up quadrature rule for the case when P is on the element.
C   This is done by splitting the quadrature rule at the centre.
      NGQON=2*NGQOFF
      DO 330 IGQ=1,NGQOFF
        AGQON(IGQ)=AGQOFF(IGQ)/TWO
        AGQON(NGQOFF+IGQ)=0.5D0+AGQOFF(IGQ)/TWO
        WGQON(IGQ)=WGQOFF(IGQ)/TWO
        WGQON(NGQOFF+IGQ)=WGQOFF(IGQ)/TWO
330   CONTINUE



C Validation that the surface is closed
      IF (LVALID) THEN
        PA(1)=VERTEX(SELV(1,1),1)
        PA(2)=VERTEX(SELV(1,1),2)
        PB(1)=VERTEX(SELV(1,2),1)
        PB(2)=VERTEX(SELV(1,2),2)
        P(1)=(PA(1)+PB(1))/TWO
        P(2)=(PA(2)+PB(2))/TWO
        VECP(1)=0.0D0
        VECP(2)=1.0D0
        SUMMK=0.0D0
        DO 180 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
C     Set LPONEL
          LPONEL=(JSE.EQ.1)

C Quadrature rule in the theta direction is constructed out of individual
C Gauss rules so that the length of each is approximately equal to the
C length of the element at the generator.
          NGQ=NGQON
          RADMID=(QA(1)+QB(1))/TWO
          SGLEN=(QA(1)-QB(1))*(QA(1)-QB(1))+
     *     (QA(2)-QB(2))*(QA(2)-QB(2))
          GLEN=SQRT(SGLEN)
          CIRMID=PI*RADMID
          NDIV=1+CIRMID/GLEN
          TDIV=ONE/DBLE(NDIV)
          NTQ=NDIV*NGQ
          IF (NTQ.GT.MAXNTQ) THEN
            WRITE(*,*) 'ERROR(IBEM3A) - MAXNTQ is set too small'
            STOP
          END IF
          DO 146 IDIV=1,NDIV
            DO 156 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQON(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQON(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
156         CONTINUE
146       CONTINUE



C     Only the Mk operator is required. Set  LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3ALC.
          CALL H3ALC(CZERO,P,VECP,QA,QB,LPONEL,
     *     MAXNGQ,NGQON,AGQON,WGQON,MAXNTQ,NTQ,ATQ,WTQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *     WKSPCE)
          
          SUMMK=SUMMK+DISMK
180     CONTINUE
        IF (ABS(SUMMK-0.5D0).LT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(AMBEMA) - in geometry'
          WRITE(*,*) ' The boundary could be oriented wrongly'
          WRITE(*,*) '  On the outer boundary arrange panels'
     *     // ' in clockwise order'
          WRITE(*,*) '  If there are inner boundaries arrange the'
     *     // ' panels in anticlockwise order'
          STOP
        END IF
        IF (ABS(SUMMK+0.5D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AMBEMA) - in geometry'
          WRITE(*,*) ' The boundary panels may be arranged incorrectly'
        END IF
      END IF  


C Validation that the points in PINT are interior points
      IF (LVALID) THEN
        DO IPI=1,NPI
        P(1)=PINT(IPI,1)
        P(2)=PINT(IPI,2)
        VECP(1)=0.0D0
        VECP(2)=1.0D0
        SUMMK=0.0D0
        DO 210 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the ISEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
C     Set LPONEL
          LPONEL=.FALSE.

C Quadrature rule in the theta direction is constructed out of individual
C Gauss rules so that the length of each is approximately equal to the
C length of the element at the generator.
          NGQ=NGQON
          RADMID=(QA(1)+QB(1))/TWO
          SGLEN=(QA(1)-QB(1))*(QA(1)-QB(1))+
     *     (QA(2)-QB(2))*(QA(2)-QB(2))
          GLEN=SQRT(SGLEN)
          CIRMID=PI*RADMID
          NDIV=1+CIRMID/GLEN
          TDIV=ONE/DBLE(NDIV)
          NTQ=NDIV*NGQ
          IF (NTQ.GT.MAXNTQ) THEN
            WRITE(*,*) 'ERROR(AMBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 147 IDIV=1,NDIV
            DO 157 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQON(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQON(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
157         CONTINUE
147       CONTINUE

C     Only the Mk operator is required. Set LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3ALC.
          CALL H3ALC(CZERO,P,VECP,QA,QB,LPONEL,
     *     MAXNGQ,NGQON,AGQON,WGQON,MAXNTQ,NTQ,ATQ,WTQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *     WKSPCE)
          
          SUMMK=SUMMK+DISMK
210     CONTINUE

        IF (ABS(SUMMK+1.0D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AMBEMA) - The observation point'
          WRITE(*,*) ' (',P(1),',',P(2),')'
          WRITE(*,*) ' is may not be interior to the boundary'
        END IF
      END DO
      END IF




      DO 500 IKPT=1,NK
      K=WKSPC5(IKPT)
      CK=CMPLX(K,ZERO)

C  Compute the discrete Lk, Mk, Mkt and Nk matrices
C   Loop(ISE) through the points on the boundary
      DO 510 ISE=1,NSE
C    Set P
        PA(1)=VERTEX(SELV(ISE,1),1)
        PA(2)=VERTEX(SELV(ISE,1),2)
        PB(1)=VERTEX(SELV(ISE,2),1)
        PB(2)=VERTEX(SELV(ISE,2),2)
        P(1)=(PA(1)+PB(1))/TWO
        P(2)=(PA(2)+PB(2))/TWO
C    Set VECP to the normal on the boundary of the element at P
        CALL NORM2(PA,PB,VECP)
C    Loop(ISE) through the elements
        DO 520 JSE=1,NSE
C     Set QA and QB, the coordinates of the edges of the ISEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)

C     Set LPONEL
          IF (ISE.EQ.JSE) THEN
            LPONEL=.TRUE.
          ELSE
            LPONEL=.FALSE.
          END IF

C     Select quadrature rule for H3ALC
C   :  Select the quadrature rule AGQON-WGQON in the case when the
C   :   point p lies on the element, otherwise select AGQOFF-WGQOFF
C      [Note that the overall method would benefit from selecting from
C       a wider set of quadrature rules, and an appropriate method
C       of selection]
          IF (LPONEL) THEN
          NGQ=NGQON
          DO 600 IGQ=1,NGQ
            AGQ(IGQ)=AGQON(IGQ)
            WGQ(IGQ)=WGQON(IGQ)
600       CONTINUE
          ELSE 
          NGQ=NGQOFF
          DO 610 IGQ=1,NGQ
            AGQ(IGQ)=AGQOFF(IGQ)
            WGQ(IGQ)=WGQOFF(IGQ)
610       CONTINUE
          END IF


C Quadrature rule in the theta direction is constructed out of individual
C Gauss rules so that the length of each is approximately equal to the
C length of the element at the generator.
          RADMID=(QA(1)+QB(1))/TWO
          SGLEN=(QA(1)-QB(1))*(QA(1)-QB(1))+
     *     (QA(2)-QB(2))*(QA(2)-QB(2))
          GLEN=SQRT(SGLEN)
          CIRMID=PI*RADMID
          NDIV=1+CIRMID/GLEN
          TDIV=ONE/DBLE(NDIV)
          NTQ=NDIV*NGQ
          IF (NTQ.GT.MAXNTQ) THEN
            WRITE(*,*) 'ERROR(AMBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 145 IDIV=1,NDIV
            DO 155 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQ(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQ(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
155         CONTINUE
145       CONTINUE


C     Only Lk and Mk operators are required. Set LLK, LMK true, 
C      LMKT,LNK false. 
            LLK=.TRUE.
            LMK=.TRUE.
            LMKT=.FALSE.
            LNK=.FALSE.


C    Call of H3ALC routine to compute [Lk], [Mk]
            CALL H3ALC(CK,P,VECP,QA,QB,LPONEL,
     *       MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
     *       LVAL,EK,EGEOM,EQRULE,LFAIL,
     *       LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *       WKSPCE)


            WKSPC1(IKPT,ISE,JSE)=DISLK
            WKSPC2(IKPT,ISE,JSE)=DISMK
         
C    Close loop(JSE) 
520       CONTINUE

          WKSPC2(IKPT,ISE,ISE)=WKSPC2(IKPT,ISE,ISE)+0.5D0
          
C   Close loop(ISE) 
510     CONTINUE

500   CONTINUE

      DO 750 JSE=1,NSE
        IF (ABS(SALPHA(JSE)).GT.ABS(SBETA(JSE))) THEN
          WKSPC7(JSE)=.TRUE.
        ELSE
          WKSPC7(JSE)=.FALSE.
        END IF
750   CONTINUE

      DO 700 IKPT=1,NK
        DO 710 ISE=1,NSE
          DO 720 JSE=1,NSE
            IF (WKSPC7(JSE)) THEN
              CONST=SBETA(JSE)/SALPHA(JSE)
              WKSPC3(IKPT,ISE,JSE)=
     *         -CONST*WKSPC2(IKPT,ISE,JSE)-WKSPC1(IKPT,ISE,JSE)
            ELSE
              CONST=SALPHA(JSE)/SBETA(JSE)
              WKSPC3(IKPT,ISE,JSE)=
     *         WKSPC2(IKPT,ISE,JSE)+CONST*WKSPC1(IKPT,ISE,JSE)
            END IF
720       CONTINUE
710     CONTINUE
700   CONTINUE


      CALL INTEIG(MAXNSE,NSE,MAXNK,NK,WKSPC5,WKSPC3,NEIG,EIGVAL,WKSPC6,
     * WKSP00,WKSP01,WKSP02,WKSP03,WKSP04,WKSP05,WKSP06,WKSP07,
     * WKSP08,WKSP09,WKSP10,WKSP11,WKSP12)


      DO 810 IEIG=1,NEIG

C Set k to the eigenfrequency
        CK=CMPLX(DBLE(EIGVAL(IEIG)),ZERO)

        DO 820 ISE=1,NSE
          IF (WKSPC7(ISE)) THEN
            SVEL(IEIG,ISE)=WKSPC6(IEIG,ISE)
            SPHI(IEIG,ISE)=-SBETA(ISE)*SVEL(IEIG,ISE)/SALPHA(ISE)
          ELSE
            SPHI(IEIG,ISE)=WKSPC6(IEIG,ISE)
            SVEL(IEIG,ISE)=-SALPHA(ISE)*SPHI(IEIG,ISE)/SBETA(ISE)
          END IF
820     CONTINUE

        
C  SOLUTION IN THE DOMAIN

C   Compute sound pressures at the selected interior points.
C    Loop through the the points in the interior region
        DO 800 IIP=1,NPI
C   Set P
          P(1)=PINT(IIP,1)
          P(2)=PINT(IIP,2)
C    Set VECP, this is arbitrary as the velocity/intensity at P
C     is not sought.
          VECP(1)=ONE
          VECP(2)=ZERO

C    Initialise SUMPHI to zero
          SUMPHI=CZERO

C    Loop(ISE) through the elements
          DO 900 JSE=1,NSE
C     Compute the discrete Lk and Mk integral operators. 
            
C     Set QA and QB, the coordinates of the edges of the JSEth element
            QA(1)=VERTEX(SELV(JSE,1),1)
            QA(2)=VERTEX(SELV(JSE,1),2)
            QB(1)=VERTEX(SELV(JSE,2),1)
            QB(2)=VERTEX(SELV(JSE,2),2)

C     All the points do not lie on the boundary hence LPONEL=.FALSE.
            LPONEL=.FALSE.              

C     Only Lk, Mk operators are required. Set LLK,LMK true, 
C      LMKT,LNK false. 
            LLK=.TRUE.
            LMK=.TRUE.
            LMKT=.FALSE.
            LNK=.FALSE.
                

C    Call of H3ALC routine to compute [Lk], [Mk]
            CALL H3ALC(CK,P,VECP,QA,QB,LPONEL,
     *       MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
     *       LVAL,EK,EGEOM,EQRULE,LFAIL,
     *       LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *       WKSPCE)

C      Accumulate phi 
            SUMPHI=SUMPHI+DISLK*SVEL(IEIG,JSE)-DISMK*SPHI(IEIG,JSE)

C      Close loop (ISE) through the elements
900       CONTINUE

          PIPHI(IEIG,IIP)=SUMPHI

C     Close loop(IIP) through the interior points
800     CONTINUE


C   Close loop(IEIG) through the eigenfrequencies
810   CONTINUE



      END


C Subordinate routines for AMBEMA
C ==============================

C ----------------------------------------------------------------------
C            Subroutine GL8.FOR by www.numerical-methods.com           |
C ----------------------------------------------------------------------
C
C Subroutine GL8 assigns the weights and points of a 8 point Gaussian
C (Gauss-Legendre) quadrature rule defined on the interval [0,1].
C
C SUBROUTINE GL8(MAXN, N, WTS, PTS)
C integer  maxn: the maximimum number of weights/points
C integer     n: the number of weights/pointsr
C real      wts: the weights
C real      pts: the points
C
C Source of the code: http://www.numerical-methods.com/fortran/GL8.FOR
C Source of the user-guide: http://www.numerical-methods.com/fortran/
C  gl8.htm
C
C Licence: This is 'open source'; the software may be used and applied
C  within other systems as long as its provenance is appropriately
C  acknowledged. See the GNU Licence http://www.gnu.org/licenses/lgpl.txt
C  for more information or contact webmaster@numerical-methods.com

C Original code 1998. Documentation enhanced 2014

       SUBROUTINE GL8(MAXN,N,WTS,PTS)
       INTEGER MAXN
       INTEGER N
       REAL*8 WTS(MAXN)
       REAL*8 PTS(MAXN)
       N=8
       WTS(1)=       5.061426814519E-02
       PTS(1)=         0.980144928249
       WTS(2)=         0.111190517227
       PTS(2)=         0.898333238707
       WTS(3)=         0.156853322939
       PTS(3)=         0.762766204958
       WTS(4)=         0.181341891689
       PTS(4)=         0.591717321248
       WTS(5)=         0.181341891689
       PTS(5)=         0.408282678752
       WTS(6)=         0.156853322939
       PTS(6)=         0.237233795042
       WTS(7)=         0.111190517227
       PTS(7)=         0.101666761293
       WTS(8)=       5.061426814519E-02
       PTS(8)=       1.985507175123E-02
       END


C Subroutines required for H3ALC (not in file H3ALC.FOR) 
C  Subroutine for returning the square root.
       REAL*8 FUNCTION FNSQRT(X)
       REAL*8 X
       FNSQRT=SQRT(X)
       END

C  Subroutine for returning the exponential.
       COMPLEX*16 FUNCTION FNEXP(Z)
       COMPLEX*16 Z
       FNEXP=EXP(Z)
       END
