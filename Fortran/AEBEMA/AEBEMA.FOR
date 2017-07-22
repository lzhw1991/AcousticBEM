C***************************************************************
C           Subroutine AEBEMA by Stephen Kirkup                     
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AEBEMA.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************

C This subroutine computes the solution to the axisymmetric Helmholtz
C equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C in the domain exterior to a closed axisymmetric boundary.
C
C The boundary (S) is defined (approximated) by a set of conical 
C elements. The domain of the equation is exterior to the boundary.
C
C The boundary condition may be Dirichlet, Robin or Neumann. It is also
C axisymmetric and it is assumed to have the following general form
C
C           {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C where {\phi}(q) is the solution at the point q on S, v(q) is the 
C derivative of {\phi} with respect to the outward normal to S at q and
C {\alpha}, {\beta} and f are complex-valued functions defined on S. The
C functions {\alpha} and {\beta} must be specified to define the nature
C of the boundary condition. Important examples are {\alpha}=1, 
C {\beta}=0 which is equivalent to a Dirichlet boundary condition and 
C {\alpha}=0, {\beta}=1 which is equivalent to a Neumann boundary 
C condition. The specification of f completes the definition of the 
C boundary condition.
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
C      |   MAIN PROGRAM     |------->-----|      AEBEMA             |   :
C      |(e.g. aebema_t.for) |       :     |                        |   :
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
C The contents of the main program must be linked to AEBEMA.FOR, 
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
C  coordinates of all the exterior points in PEXT, the subroutine
C  returns the value of {\phi} at these points in PEPHI.
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

C Notes on the exterior points 
C ----------------------------
C (1) The points in PEXT should lie outside the boundary, as defined
C  by the parameters VERTEX and SELV. Any point lying outside the 
C  boundary will return a corresponding value in PEPHI that is near
C  zero.

C Notes on the boundary condition
C -------------------------------
C (1) For each i=1..NSE, it must not be the case that both of SALPHA(i)
C  and SBETA(i) are zero

C External modules in external files
C ==================================
C subroutine H3ALC: Returns the individual discrete Helmholtz integral
C  operators. (in file H3ALC.FOR)
C subroutine CGLS: Solves a general linear system of equations. 
C  (in file CGLS.FOR)
C real function DIST2: Returns the distance between two 2-vectors. (in
C  file GEOM2D.FOR)

C External modules provided in the package (this file)
C ====================================================
C subroutine GL8: Returns the points and weights of the 8-point Gauss-
C  Legendre quadrature rule.
C real function FNSQRT(X): real X : Returns the square root of X.
C complex function FNEXP(Z): complex z : Returns the complex 
C  exponential of X.

C The subroutine

      SUBROUTINE AEBEMA(K,
     *                  MAXNV,NV,VERTEX,MAXNSE,NSE,SELV,
     *                  MAXNPE,NPE,PEXT,
     *                  SALPHA,SBETA,SF,SFFPHI,SFFVEL,PFFPHI,
     *                  LSOL,LVALID,EGEOM,MU,
     *                  SPHI,SVEL,PEPHI,
     *                  WKSPC1,WKSPC2,WKSPC3,WKSPC4,
     *                  WKSPC5,WKSPC6,WKSPC7)
      PARAMETER (MAXNGQ=32)
      PARAMETER (MAXNTQ=10000)

C  Wavenumber
      REAL*8     K

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
      
C  Exterior points at which the solution is to be observed
C   Limit on the number of points exterior to the boundary where 
C    solution is sought
      INTEGER    MAXNPE
C   The number of exterior points
      INTEGER    NPE
C   Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,2)

C  The boundary condition is such that {\alpha} {\phi} + {\beta} v = f
C  where alpha, beta and f are complex valued functions over S.
C  The functions are set values at the collocation points.
C   function alpha
      COMPLEX*16 SALPHA(MAXNSE)
C   function beta
      COMPLEX*16 SBETA(MAXNSE)
C   function f
      COMPLEX*16 SF(MAXNSE)

C  The incident velocity potential on S
      COMPLEX*16 SFFPHI(MAXNSE)
C  The derivative of the incident velocity potential on S
      COMPLEX*16 SFFVEL(MAXNSE)
C  The incident velocity potential at the chosen exterior points
      COMPLEX*16 PFFPHI(MAXNPE)


C  Validation and control parameters
      LOGICAL    LSOL
      LOGICAL    LVALID
      REAL*8     EGEOM
      COMPLEX*16 MU

C  Solution 
C   function phi
      COMPLEX*16 SPHI(MAXNSE)
C   function vel
      COMPLEX*16 SVEL(MAXNSE)
C   domain solution
      COMPLEX*16 PEPHI(MAXNPE)

C  Working space 
      COMPLEX*16 WKSPC1(MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC2(MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC3(MAXNPE,MAXNSE)
      COMPLEX*16 WKSPC4(MAXNPE,MAXNSE)
      COMPLEX*16 WKSPC5(MAXNSE)
      COMPLEX*16 WKSPC6(MAXNSE)
      LOGICAL    WKSPC7(MAXNSE)

c  External function
      REAL*8     DIST2

C  Constants
C   Real scalars: 0, 1, 2, half, pi
      REAL*8 ZERO,ONE,TWO,HALF,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Wavenumber in complex form
      COMPLEX*16 CK

C  Geometrical description of the boundary
C   Elements counter
      INTEGER    ISE,JSE
C   The points exterior to the boundary where the solution is sought 
      INTEGER    IPE
C   Parameters for H3ALC
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
C    Actual number of quadrature points
      INTEGER    NGQ
C    Abscissae of the actual quadrature rule in the generator direction
      REAL*8     AGQ(MAXNGQ)
C    Weights of the actual quadrature rule
      REAL*8     WGQ(MAXNGQ)
C   Counter through the quadrature points
      INTEGER    IGQ
C    Abscissae of the actual quadrature rule in the theta direction
      REAL*8     ATQ(MAXNTQ)
C    Weights of the actual quadrature rule
      REAL*8     WTQ(MAXNTQ)

C  Validation and control parameters for subroutine H3ALC
      LOGICAL    LVAL
      REAL*8     EK
      REAL*8     EQRULE
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C  Parameters for subroutine H3ALC. 
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

      REAL*8 WKSPCE(2*MAXNTQ+MAXNGQ)


C  Other variables
C   Error flag
      LOGICAL    LERROR
C   Failure flag
      LOGICAL    LFAIL
C   Accumulation of solution {\phi}
      COMPLEX*16 SUMPHI
C   Stores a vertex
      REAL*8     QC(2)
C   Maximum,minimum sizes of elements
      REAL*8     SIZMAX,SIZMIN
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


C Validation of parameters of AEBEMA
C ---------------------------------

      IF (LVALID) THEN

C Validation of main paramters
        LERROR=.FALSE.
        IF (K.LT.ZERO) THEN
          WRITE(*,*) 'K = ',K
          WRITE(*,*) 'ERROR(AEBEMA) - K must be positive'
          LERROR=.TRUE.
        END IF
        IF (MAXNV.LT.3) THEN
          WRITE(*,*) 'MAXNV = ',MAXNV
          WRITE(*,*) 'ERROR(AEBEMA) - must have MAXNV>=3'
          LERROR=.TRUE.
        END IF
        IF (NV.LT.3.OR.NV.GT.MAXNV) THEN
          WRITE(*,*) 'NV = ',NV
          WRITE(*,*) 'ERROR(AEBEMA) - must have 3<=NV<=MAXNV'
          LERROR=.TRUE.
        END IF
        IF (MAXNSE.LT.3) THEN
          WRITE(*,*) 'MAXNSE = ',MAXNSE
          WRITE(*,*) 'ERROR(AEBEMA) - must have MAXNSE>=3'
          LERROR=.TRUE.
        END IF
        IF (NSE.LT.3.OR.NSE.GT.MAXNSE) THEN
          WRITE(*,*) 'NSE = ',NSE
          WRITE(*,*) 'ERROR(AEBEMA) - must have 3<=NSE<=MAXNSE'
          LERROR=.TRUE.
        END IF
        IF (MAXNPE.LT.1) THEN
          WRITE(*,*) 'MAXNPE = ',MAXNPE
          WRITE(*,*) 'ERROR(AEBEMA) - must have MAXNPE>=1'
          LERROR=.TRUE.
        END IF
        IF (NPE.LT.0.OR.NPE.GT.MAXNPE) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(AEBEMA) - must have 3<=NPE<=MAXNPE'
          LERROR=.TRUE.
        END IF
        IF (EGEOM.LE.ZERO) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(AEBEMA) - EGEOM must be positive'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error(s) found in the main parameters of AEBEMA'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF

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

      IF (LVALID) THEN
        LERROR=.FALSE.
C Check that the boundary defined by SELV is complete and closed
        DO 120 ISE=1,NSE-1
          IF (SELV(ISE,2).NE.SELV(ISE+1,1)) LERROR=.TRUE.
120     CONTINUE
        IF (VERTEX(SELV(1,1),1).GT.EGEOM) LERROR=.TRUE.
        IF (VERTEX(SELV(NSE,2),1).GT.EGEOM) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(*,*) 'ERROR(AEBEMA) - boundary defined by SELVis not'
          WRITE(*,*) ' complete and closed'
        END IF
C Check that EGEOM is not too large
        IF (EGEOM.GT.DIAM/100.0D0) THEN
          WRITE(*,*) 'EGEOM = ',EGEOM
          WRITE(*,*) 'ERROR(AEBEMA) - EGEOM is set too large'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error in boundary geometry or EGEOM'
          WRITE(*,*) 'Execution terminated'
        END IF
      END IF                  

      IF (LVALID) THEN
C Check that the vertices are distinct with respect to EGEOM
        LERROR=.FALSE.
        DO 130 IV=1,NV-1
          PA(1)=VERTEX(IV,1)
          PA(2)=VERTEX(IV,2)
          DO 140 JV=IV+1,NV
            PB(1)=VERTEX(JV,1)
            PB(2)=VERTEX(JV,2)
            IF (ABS(PA(1)-PB(1)).LT.EGEOM) THEN
              IF (ABS(PA(2)-PB(2)).LT.EGEOM) THEN
                WRITE(*,*) 'Vertices ',IV,JV,' are not distinct'
                LERROR=.TRUE.
              END IF
            END IF
140       CONTINUE
130     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(AEBEMA) - Vertices (see above) coincide'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF          


C Check that the elements are not of disproportionate sizes
      IF (LVALID) THEN
        SIZMAX=ZERO
        SIZMIN=DIAM
        DO 150 ISE=1,NSE
          QA(1)=VERTEX(SELV(ISE,1),1)
          QA(2)=VERTEX(SELV(ISE,1),2)
          QB(1)=VERTEX(SELV(ISE,2),1)
          QB(2)=VERTEX(SELV(ISE,2),2)
          SIZMAX=MAX(SIZMAX,DIST2(QA,QB))
          SIZMIN=MIN(SIZMIN,DIST2(QA,QB))
150     CONTINUE
        IF (SIZMAX.GT.10.0D0*SIZMIN) THEN
          WRITE(*,*) 'WARNING(AEBEMA) - Elements of disproportionate'
          WRITE(*,*) ' sizes'
        END IF
      END IF          
          
C Check that the boundary does not contain sharp angles
      IF (LVALID) THEN
        LERROR=.FALSE.
        DO 160 ISE=2,NSE
          QA(1)=VERTEX(SELV(ISE,1),1)
          QA(2)=VERTEX(SELV(ISE,1),2)
          QB(1)=VERTEX(SELV(ISE,2),1)
          QB(2)=VERTEX(SELV(ISE,2),2)
          QC(1)=VERTEX(SELV(ISE-1,1),1)
          QC(2)=VERTEX(SELV(ISE-1,1),2)
          IF (DIST2(QC,QB).LT.MAX(DIST2(QA,QB),DIST2(QA,QC))) THEN
            WRITE(*,*) 'Sharp angle at node ',SELV(ISE,1)
            LERROR=.TRUE.
          END IF
160     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*)
          WRITE(*,*) 'WARNING(AEBEMA) - Boundary has sharp angles'
        END IF
      END IF          
     

C Validation of the surface functions
      IF (LVALID.AND.LSOL) THEN
        LERROR=.FALSE.
        DO 170 ISE=1,NSE
          IF (MAX(ABS(SALPHA(ISE)),ABS(SBETA(ISE))).LT.1.0D-6) 
     *     LERROR=.TRUE.
170     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(AEBEMA) - at most one of SALPHA(i),SBETA(i)'
          WRITE(*,*) ' may be zero for all i'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF
        

C Set the wavenumber in complex form
      CK=CMPLX(K,ZERO)


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
            WRITE(*,*) 'ERROR(AEBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 146 IDIV=1,NDIV
            DO 156 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQON(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQON(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
156         CONTINUE
146       CONTINUE



C     Only the Mk operators is required. Set LMK true, 
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
          WRITE(*,*) 'ERROR(AEBEMA) - in geometry'
          WRITE(*,*) ' The boundary could be oriented wrongly'
          WRITE(*,*) '  On the outer boundary arrange panels'
     *     // ' in clockwise order'
          WRITE(*,*) '  If there are inner boundaries arrange the'
     *     // ' panels in anticlockwise order'
          STOP
        END IF
        IF (ABS(SUMMK+0.5D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AEBEMA) - in geometry'
          WRITE(*,*) ' The boundary panels may be arranged incorrectly'
        END IF
      END IF  


C Validation that the points in PEXT are exterior points
      IF (LVALID) THEN
        DO IPE=1,NPE
        P(1)=PEXT(IPE,1)
        P(2)=PEXT(IPE,2)
        VECP(1)=0.0D0
        VECP(2)=1.0D0
        SUMMK=0.0D0
        DO 210 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
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
            WRITE(*,*) 'ERROR(AEBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 147 IDIV=1,NDIV
            DO 157 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQON(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQON(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
157         CONTINUE
147       CONTINUE

C     Only the Mk operators is required. Set LMK true, 
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

        IF (ABS(SUMMK).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AEBEMA) - The observation point'
          WRITE(*,*) ' (',P(1),',',P(2),')'
          WRITE(*,*) ' may not be exterior to the boundary'
        END IF
      END DO
      END IF


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
C     Set QA and QB, the coordinates of the edges of the JSEth element
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
            WRITE(*,*) 'ERROR(AEBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 145 IDIV=1,NDIV
            DO 155 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQ(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQ(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
155         CONTINUE
145       CONTINUE


C     All operators are required
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.TRUE.
          LNK=.TRUE.


C    Call of H3ALC routine to compute [Lk], [Mk], [Mkt], [Nk]
          CALL H3ALC(CK,P,VECP,QA,QB,LPONEL,
     *     MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *     WKSPCE)

          WKSPC1(ISE,JSE)=DISLK+MU*DISMKT
          WKSPC2(ISE,JSE)=DISMK+MU*DISNK
         
C    Close loop(JSE) 
520     CONTINUE

        WKSPC1(ISE,ISE)=WKSPC1(ISE,ISE)+MU/TWO
        WKSPC2(ISE,ISE)=WKSPC2(ISE,ISE)-HALF
        IF (LSOL) WKSPC6(ISE)=-(SFFPHI(ISE)+MU*SFFVEL(ISE))
            
C   Close loop(ISE) 
510   CONTINUE

      IF (LSOL) THEN
        CALL CGLS(MAXNSE,NSE,WKSPC2,WKSPC1,WKSPC6,SALPHA,SBETA,SF,
     *   SPHI,SVEL,LFAIL,WKSPC5,WKSPC7)
      END IF
        

C  SOLUTION IN THE DOMAIN

C   Compute sound pressures at the selected exterior points.
C    Loop through the the points in the exterior region
      DO 800 IPE=1,NPE
C    Set P
        P(1)=PEXT(IPE,1)
        P(2)=PEXT(IPE,2)
C    Set VECP, this is arbitrary as the velocity/intensity at P
C     is not sought.
        VECP(1)=ONE
        VECP(2)=ZERO

C    Initialise SUMPHI to zero
        SUMPHI=PFFPHI(IPE)

C    Loop(ISE) through the elements
        DO 850 JSE=1,NSE
C     Compute the discrete Lk and Mk integral operators. 
            
C     Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)

C Quadrature rule in the generator direction
          NGQ=NGQOFF
          DO 1610 IGQ=1,NGQ
            AGQ(IGQ)=AGQOFF(IGQ)
            WGQ(IGQ)=WGQOFF(IGQ)
1610      CONTINUE

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
            WRITE(*,*) 'ERROR(AEBEMA) - MAXNTQ is set too small'
            STOP
          END IF
          DO 1145 IDIV=1,NDIV
            DO 1155 IGQ=1,NGQ
              WTQ((IDIV-1)*NGQ+IGQ)=WGQ(IGQ)/DBLE(NDIV)
              ATQ((IDIV-1)*NGQ+IGQ)=AGQ(IGQ)/DBLE(NDIV)+
     *         TDIV*DBLE(IDIV-1)
1155         CONTINUE
1145       CONTINUE

C     All the points do not lie on the boundary hence LPONEL=.FALSE.
          LPONEL=.FALSE.              

C     Only Lk, Mk operators are required. Set LLK,LMK true, 
C      LMKT,LNK false. 
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.
                
C     Call H3ALC.
             CALL H3ALC(CK,P,VECP,QA,QB,LPONEL,
     *        MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
     *        LVAL,EK,EGEOM,EQRULE,LFAIL,
     *        LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     *        WKSPCE)

          IF (.NOT.LSOL) THEN
            WKSPC3(IPE,JSE)=DISLK
            WKSPC4(IPE,JSE)=DISMK
          END IF

C     Accumulate phi 
          IF (LSOL) SUMPHI=SUMPHI-DISLK*SVEL(JSE)+DISMK*SPHI(JSE)

C      Close loop (JSE) through the elements
850     CONTINUE

        PEPHI(IPE)=SUMPHI

C     Close loop(IPE) through the exterior points
800   CONTINUE

      END


C ----------------------------------------------------------------------

C Subordinate routines for AEBEMA
C ===============================

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
       PTS(1)=       1.985507175123E-02
       WTS(2)=         0.111190517227
       PTS(2)=         0.101666761293
       WTS(3)=         0.156853322939
       PTS(3)=         0.237233795042
       WTS(4)=         0.181341891689
       PTS(4)=         0.408282678752
       WTS(5)=         0.181341891689
       PTS(5)=         0.591717321248
       WTS(6)=         0.156853322939
       PTS(6)=         0.762766204958
       WTS(7)=         0.111190517227
       PTS(7)=         0.898333238707
       WTS(8)=       5.061426814519E-02
       PTS(8)=         0.980144928249
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
