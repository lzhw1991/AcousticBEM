C***************************************************************
C           Subroutine AIBEM2 by Stephen Kirkup                     
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AIBEM2.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This subroutine computes the solution to the two-dimensional Helmholtz
C equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C in the domain interior to a closed boundary.
C
C The boundary (S) is defined (approximated) by a set of straight line 
C elements. The domain of the equation is within the boundary.
C
C The boundary condition may be Dirichlet, Robin or Neumann. It is 
C assumed to have the following general form
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
C program is required along with the definition of FNHANK, a routine
C that must return the spherical hankel function.
C
C                                   .........................
C                                   : --------------------- :
C Routine to be supplied ---------> : |    FNHANK         | :
C                                   : | (e.g. fnhank.for) | :        
C                                   : --------------------- :
C                                   :........|..............:        
C                                            |                
C                                   .........<..........................
C                                   :        |                         :
C                                   :        |                         :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|      AIBEM2             |   :
C      |(e.g. aibem2_t.for) |       :     |                        |   :
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
C             : (geom2d.for) :---<---:   | H2LC |   :   :  | CGLS |    : 
C             :              :       :   --------   :   :  --------    :  
C             :..............:       : -------------:   : -------------:  
C                                    : |subordinate|:   : |subordinate|: 
C                                    : | routines  |:   : | routines  |:  
C                                    : -------------:   : -------------: 
C                                    :              :   :              : 
C                                    : (h2lc.for)   :   : (cgls.for)   :
C                                    :..............:   :..............:
C                                    
C
C The contents of the main program, the file containing FNHANK must be
C  linked to AIBEM2.FOR, H2LC.FOR and CGLS.FOR.
C
C Method of solution
C ------------------
C 
C In the main program, the boundary must be described as a set of
C  elements. The elements are defined by two indices (integers) which
C  label a node or vertex on the boundary. The data structure
C  VERTEX lists and enumerates the coordinates of the vertices, the
C  data structure SELV defines each element by indicating the labels
C  for the two nodes it lies between and hence enumerates the elements.
C The boundary solution points (the points on the boundary at which 
C  {\phi} (SPHI) and d {\phi}/dn (SVEL) are returned) are at the centres
C  of the elements. The boundary functions {\alpha} (SALPHA), {\beta} 
C  (SBETA) and f (SF) are also defined by their values at the centres
C  of the elements.
C Normally a solution in the domain is required. By listing the 
C  coordinates of all the interior points in PINT, the subroutine
C  returns the value of {\phi} at these points in PIPHI.
C
C
C Format and parameters of the subroutine
C ---------------------------------------
C
C The subroutine has the form
C
C      SUBROUTINE AIBEM2(K,
C     *                 MAXNV,NV,VERTEX,MAXNSE,NSE,SELV,
C     *                 MAXNPI,NPI,PINT,
C     *                 SALPHA,SBETA,SF,SFFPHI,SFFVEL,PFFPHI,
C     *                 LSOL,LVALID,EGEOM,MU,
C     *                 SPHI,SVEL,PIPHI,
C     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,
c     *                 WKSPC5,WKSPC6,WKSPC7)

C The parameters to the subroutine
C ================================

C Wavenumber (input)
C real K: Must be positive.

C Geometry of the boundary S (input)
C integer MAXNV: The limit on the number of vertices of the polygon
C  that defines (approximates) S. MAXNV>=3.
C integer NV: The number of vertices on S. 3<=NV<=MAXNV.
C real VERTEX(MAXNV,2): The coordinates of the vertices. VERTEX(i,1),
C  VERTEX(i,2) are the x,y coordinates of the i-th vertex.
C integer MAXNSE: The limit on the number of elements describing S.
C  MAXNSE>=3.
C integer NSE: The number of elements describing S. 3<=NSE<=MAXNSE.
C integer SELV(MAXNSE,2): The indices of the two vertices defining 
C  each element. The i-th element have vertices 
C  (VERTEX(SELV(i,1),1),VERTEX(SELV(i,1),2)) and
C  (VERTEX(SELV(i,2),1),VERTEX(SELV(i,2),2)).

C Interior points at which the solution is to be observed (input)
C integer MAXNPI: Limit on the number of points interior to the
C  boundary. MAXNPI>=1.
C integer NPI: The number of interior points. 0<=NPI<=MAXNPI.
C real PINT(MAXNPI,2). The coordinates of the interior point.
C  PINT(i,1),PINT(i,2) are the x,y coordinates of the i-th point.

C The boundary condition ({\alpha} {\phi} + {\beta} v = f) (input)
C complex SALPHA(MAXNSE): The values of {\alpha} at the centres
C  of the elements.
C complex SBETA(MAXNSE): The values of {\beta} at the centres
C  of the elements.
C complex SF(MAXNSE): The values of f at the centres of the 
C  elements.

C complex SFFPHI(MAXNSE): The incident velocity potential at the
C  centres of the elements
C complex SFFVEL(MAXNSE): The derivative of the incident velocity 
C  centres of the elements
C complex PFFPHI(MAXNPI): The incident velocity potential at the chosen
C  interior points

C Validation and control parameters (input) 
C logical LSOL: A switch to control whether the particular
C  solution is required.
C logical LVALID: A switch to enable the choice of checking of 
C  subroutine parameters.
C real EGEOM: The maximum absolute error in the parameters that
C  describe the geometry.
C complex MU: The weighting parameter in the direct formulations.
C  As a default, set MU=I/(K+1).

C  Solution (output)
C  complex SPHI(MAXNSE): The velocity potential ({\phi}) at the
C   centres of the boundary elements.
C  complex SVEL(MAXNSE): The velocity (v or d{\phi}/dn where n is
C   the outward normal to the boundary) at the centres of the boundary 
C   elements.
C  complex PIPHI(MAXNPI): The velocity potential ({\phi}) at the 
C   interior points.

C  Working space
C  complex WKSPC1(MAXNSE,MAXNSE)
C  complex WKSPC2(MAXNSE,MAXNSE)
C  complex WKSPC3(MAXNPI,MAXNSE)
C  complex WKSPC4(MAXNPI,MAXNSE)
C  complex WKSPC5(MAXNSE)
C  complex WKSPC6(MAXNSE)
C  logical WKSPC7(MAXNSE)

C External modules to be supplied with the main program
C =====================================================
C Subroutine FNHANK(Z,H): complex Z (input), complex H(0:1) (output):
C  Must return the spherical Hankel functions of the first kind of
C  order zero H(0) and the spherical Hankel function of the first
C  kind of order one H(1). This subroutine is important when k is
C  non-zero. (e.g. file FNHANK.FOR)
 
C Notes on the geometric parameters
C ---------------------------------
C (1) Each of the vertices listed in VERTEX must be distinct points
C  with respect to EGEOM.
C (2) The boundary must be complete and closed. Thus 
C  SELV(i,2)=SELV(i+1,1) for i=1..NSE-1 and SELV(1,1)=SELV(NSE,2).
C (3) The indices of the nodes listed in SELV must be such that they
C  are ordered counter-clockwise around the boundary.
C (4) The largest element must be no more than 10x the length of the
C  smallest element.

C Notes on the interior points 
C ----------------------------
C (1) The points in PINT should lie within the boundary, as defined
C  by the parameters VERTEX and SELV. Any point lying outside the 
C  boundary will return a corresponding value in PIPHI that is near
C  zero.

C Notes on the boundary condition
C -------------------------------
C (1) For each i=1..NSE, it must not be the case that both of SALPHA(i)
C  and SBETA(i) are zero.

C External modules in external files
C ==================================
C subroutine H2LC: Returns the individual discrete Helmholtz integral
C  operators (in file H2LC.FOR).
C subroutine CGLS: Solves a general linear system of equations. 
C  (in file CGLS.FOR)
C real function DIST2: Returns the distance between two 2-vectors. (in
C  file GEOM2D.FOR)

C External modules provided in the package (this file)
C ====================================================
C subroutine GL8: Returns the points and weights of the 8-point Gauss-
C  Legendre quadrature rule.
C real function FNSQRT(X): real X : Returns the square root of X.
C real function FNLOG(X): real X : Returns the natural logarithm of X.


C The subroutine

      SUBROUTINE AIBEM2(K,
     *                 MAXNV,NV,VERTEX,MAXNSE,NSE,SELV,
     *                 MAXNPI,NPI,PINT,
     *                 SALPHA,SBETA,SF,SFFPHI,SFFVEL,PFFPHI,
     *                 LSOL,LVALID,EGEOM,MU,
     *                 SPHI,SVEL,PIPHI,
     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,
     *                 WKSPC5,WKSPC6,WKSPC7)
      PARAMETER (MAXNQ=32)

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
      
C  Interior points at which the solution is to be observed
C   Limit on the number of points interior to the boundary where 
C    solution is sought
      INTEGER    MAXNPI
C   The number of interior points
      INTEGER    NPI
C   Coordinates of the interior points
      REAL*8     PINT(MAXNPI,2)

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
C  The incident velocity potential at the chosen interior points
      COMPLEX*16 PFFPHI(MAXNPI)

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
      COMPLEX*16 PIPHI(MAXNPI)

C  Working space 
      COMPLEX*16 WKSPC1(MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC2(MAXNSE,MAXNSE)
      COMPLEX*16 WKSPC3(MAXNPI,MAXNSE)
      COMPLEX*16 WKSPC4(MAXNPI,MAXNSE)
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
C   The points interior to the boundary where the solution is sought 
      INTEGER    IPI
C   Parameters for H2LC
      REAL*8     P(2),PA(2),PB(2),QA(2),QB(2),VECP(2)
      LOGICAL    LPONEL


C  Quadrature rule information
C   [Note that in this program two quadrature rules are used: one for
C    the case when the point P lies on the element (LPONEL=.TRUE.) and
C    one for the case when P does not lie on the element.]
C   Quadrature rule used when LPONEL=.TRUE.
C    Number of quadrature points
      INTEGER    NQON
C    Abscissae of the actual quadrature rule
      REAL*8     AQON(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQON(MAXNQ)
C   Quadrature rule used when LPONEL=.FALSE.
C    Number of quadrature points
      INTEGER    NQOFF
C    Abscissae of the actual quadrature rule
      REAL*8     AQOFF(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQOFF(MAXNQ)
C   Quadrature rule parameters for H2LC
C    Actual number of quadrature points
      INTEGER    NQ
C    Abscissae of the actual quadrature rule
      REAL*8     AQ(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQ(MAXNQ)
C   Counter through the quadrature points
      INTEGER    IQ

C  Validation and control parameters for subroutine H2LC
      LOGICAL    LVAL
      REAL*8     EK
      REAL*8     EQRULE
      LOGICAL    LFAIL1
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C  Parameters for subroutine H2LC. 
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

C  Other variables
C   Error flag
      LOGICAL    LERROR
C   Failure flag
      LOGICAL    LFAIL
C   Accumulation of solution {\phi}
      COMPLEX*16 SUMPHI
C   Evaluation of FNHANK (temporary)
      COMPLEX*16 THANK(0:1)
C   Stores a vertex
      REAL*8     QC(2)
C   Maximum,minimum sizes of elements
      REAL*8     SIZMAX,SIZMIN
C   The `diameter' of the boundary or the maximum distance between any
C    two vertices
      REAL*8     DIAM
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

      EQRULE=1.0D-6
      EK=1.0D-6

C Validation
C ==========

C Check that the subroutine FNHANK is satisfactory
      IF (LVALID.AND.K.GT.EK) THEN
        CALL FNHANK(CONE,THANK)
        WRITE(*, *) 'THANK=', THANK 
        LERROR=.FALSE.
        IF (ABS(DBLE(THANK(0))-0.7652).GT.1.0D-3) LERROR=.TRUE.
        IF (ABS(AIMAG(THANK(0))-0.0882).GT.1.0D-3) LERROR=.TRUE.
        IF (ABS(DBLE(THANK(1))-0.4401).GT.1.0D-3) LERROR=.TRUE.
        IF (ABS(AIMAG(THANK(1))+0.7812).GT.1.0D-3) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(*,*) 'ERROR(AIBEM2) - in Hankel function routine FNHANK'
          STOP
        END IF
      END IF


C Validation of parameters of AIBEM2
C ---------------------------------

      IF (LVALID) THEN

C Validation of main paramters
        LERROR=.FALSE.
        IF (K.LT.ZERO) THEN
          WRITE(*,*) 'K = ',K
          WRITE(*,*) 'ERROR(AIBEM2) - K must be positive'
          LERROR=.TRUE.
        END IF
        IF (MAXNV.LT.3) THEN
          WRITE(*,*) 'MAXNV = ',MAXNV
          WRITE(*,*) 'ERROR(AIBEM2) - must have MAXNV>=3'
          LERROR=.TRUE.
        END IF
        IF (NV.LT.3.OR.NV.GT.MAXNV) THEN
          WRITE(*,*) 'NV = ',NV
          WRITE(*,*) 'ERROR(AIBEM2) - must have 3<=NV<=MAXNV'
          LERROR=.TRUE.
        END IF
        IF (MAXNSE.LT.3) THEN
          WRITE(*,*) 'MAXNSE = ',MAXNSE
          WRITE(*,*) 'ERROR(AIBEM2) - must have MAXNSE>=3'
          LERROR=.TRUE.
        END IF
        IF (NSE.LT.3.OR.NSE.GT.MAXNSE) THEN
          WRITE(*,*) 'NSE = ',NSE
          WRITE(*,*) 'ERROR(AIBEM2) - must have 3<=NSE<=MAXNSE'
          LERROR=.TRUE.
        END IF
        IF (MAXNPI.LT.1) THEN
          WRITE(*,*) 'MAXNPI = ',MAXNPI
          WRITE(*,*) 'ERROR(AIBEM2) - must have MAXNPI>=1'
          LERROR=.TRUE.
        END IF
        IF (NPI.LT.0.OR.NPI.GT.MAXNPI) THEN
          WRITE(*,*) 'NPI = ',NPI
          WRITE(*,*) 'ERROR(AIBEM2) - must have 3<=NPI<=MAXNPI'
          LERROR=.TRUE.
        END IF
        IF (EGEOM.LE.ZERO) THEN
          WRITE(*,*) 'NPI = ',NPI
          WRITE(*,*) 'ERROR(AIBEM2) - EGEOM must be positive'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error(s) found in the main parameters of AIBEM2'
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
        IF (LERROR) THEN
          WRITE(*,*) 'ERROR(AIBEM2) - boundary defined by SELVis not'
          WRITE(*,*) ' complete and closed'
        END IF
C Check that EGEOM is not too large
        IF (EGEOM.GT.DIAM/100.0D0) THEN
          WRITE(*,*) 'EGEOM = ',EGEOM
          WRITE(*,*) 'ERROR(AIBEM2) - EGEOM is set too large'
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
          WRITE(*,*) 'ERROR(AIBEM2) - Vertices (see above) coincide'
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
          WRITE(*,*) 'WARNING(AIBEM2) - Elements of disproportionate'
          WRITE(*,*) ' sizes'
        END IF
      END IF          
          
C Check that the boundary does not contain sharp angles
      IF (LVALID) THEN
        LERROR=.FALSE.
        DO 160 ISE=1,NSE
          QA(1)=VERTEX(SELV(ISE,1),1)
          QA(2)=VERTEX(SELV(ISE,1),2)
          QB(1)=VERTEX(SELV(ISE,2),1)
          QB(2)=VERTEX(SELV(ISE,2),2)
          IF (ISE.GE.2) THEN
            QC(1)=VERTEX(SELV(ISE-1,1),1)
            QC(2)=VERTEX(SELV(ISE-1,1),2)
          ELSE
            QC(1)=VERTEX(SELV(NSE,1),1)
            QC(2)=VERTEX(SELV(NSE,1),2)
          END IF
          IF (DIST2(QC,QB).LT.MAX(DIST2(QA,QB),DIST2(QA,QC))/5) THEN
            WRITE(*,*) 'Sharp angle at node ',SELV(ISE,1)
            LERROR=.TRUE.
          END IF
160     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*)
          WRITE(*,*) 'WARNING(AIBEM2) - Boundary has sharp angles'
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
          WRITE(*,*) 'ERROR(AIBEM2) - at most one of SALPHA(i),SBETA(i)'
          WRITE(*,*) ' may be zero for all i'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF

C Set up the quadrature rule(s).
C  Set up quadrature rule for the case when P is not on the element.
C   Set up 8 point Gauss-Legendre rules
      CALL GL8(MAXNQ,NQOFF,WQOFF,AQOFF)
C  Set up quadrature rule for the case when P is on the element.
C   This is done by splitting the quadrature rule at the centre.
      NQON=2*NQOFF
      DO 330 IQ=1,NQOFF
        AQON(IQ)=AQOFF(IQ)/TWO
        AQON(NQOFF+IQ)=0.5D0+AQOFF(IQ)/TWO
        WQON(IQ)=WQOFF(IQ)/TWO
        WQON(NQOFF+IQ)=WQOFF(IQ)/TWO
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

C     Switch off the validation parameter for H2LC
          LVAL=.FALSE.


C     Only Mk operators is required. Set LMK true, 
C      LLK, LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H2LC.
          CALL H2LC(CZERO,P,VECP,QA,QB,LPONEL,
     *     MAXNQ,NQON,AQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)
          
          SUMMK=SUMMK+DISMK
180     CONTINUE
        IF (ABS(SUMMK+0.5D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(AIBEM2) - in geometry'
          WRITE(*,*) ' The boundary could be oriented wrongly'
          WRITE(*,*) '  On the outer boundary arrange panels'
     *     // ' in clockwise order'
          WRITE(*,*) '  If there are inner boundaries arrange the'
     *     // ' panels in anticlockwise order'
          STOP
        END IF
        IF (ABS(SUMMK+0.5D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AIBEM2) - in geometry'
          WRITE(*,*) ' The boundary panels may be arranged incorrectly'
          STOP
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
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
C     Set LPONEL
          LPONEL=.FALSE.

C     Only the Mk operator is required. Set LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H2LC.
          CALL H2LC(CZERO,P,VECP,QA,QB,LPONEL,
     *     MAXNQ,NQON,AQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)
          
          SUMMK=SUMMK+DISMK
210     CONTINUE

        IF (ABS(SUMMK+1.0D0).GT.0.01) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AIBEM2) - The observation point'
          WRITE(*,*) ' (',P(1),',',P(2),')'
          WRITE(*,*) ' is may not be interior to the boundary'
        END IF
      END DO
      END IF



C Set the wavenumber in complex form
      CK=CMPLX(K,ZERO)


C Set up validation and control parameters
C  Switch on the validation of H2LC
      LVAL=.FALSE.
C  Set EK
      EK=1.0D-6
C  Set EQRULE
      EQRULE=1.0D-6


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
          LPONEL=(ISE.EQ.JSE)

C     Select quadrature rule for H2LC
C   :  Select the quadrature rule AQON-WQON in the case when the
C   :   point p lies on the element, otherwise select AQOFF-WQOFF
C      [Note that the overall method would benefit from selecting from
C       a wider set of quadrature rules, and an appropriate method
C       of selection]
          IF (LPONEL) THEN
          NQ=NQON
          DO 600 IQ=1,NQ
            AQ(IQ)=AQON(IQ)
            WQ(IQ)=WQON(IQ)
600       CONTINUE
          ELSE 
          NQ=NQOFF
          DO 610 IQ=1,NQ
            AQ(IQ)=AQOFF(IQ)
            WQ(IQ)=WQOFF(IQ)
610       CONTINUE
          END IF

C     All operators are required
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.TRUE.
          LNK=.TRUE.

C     Call H2LC.
          CALL H2LC(CK,P,VECP,QA,QB,LPONEL,
     *     MAXNQ,NQ,AQ,WQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          WKSPC1(ISE,JSE)=DISLK+MU*DISMKT
          WKSPC2(ISE,JSE)=DISMK+MU*DISNK
         
C    Close loop(JSE) 
520     CONTINUE

        WKSPC1(ISE,ISE)=WKSPC1(ISE,ISE)-MU/TWO
        WKSPC2(ISE,ISE)=WKSPC2(ISE,ISE)+HALF
        IF (LSOL) WKSPC6(ISE)=SFFPHI(ISE)+MU*SFFVEL(ISE)
            
C   Close loop(ISE) 
510   CONTINUE

      IF (LSOL) THEN
        CALL CGLS(MAXNSE,NSE,WKSPC2,WKSPC1,WKSPC6,SALPHA,SBETA,SF,
     *   SPHI,SVEL,LFAIL,WKSPC5,WKSPC7)
      END IF  

C  SOLUTION IN THE DOMAIN

C   Compute sound pressures at the selected interior points.
C    Loop through the the points in the interior region
      DO 800 IPI=1,NPI
C    Set P
        P(1)=PINT(IPI,1)
        P(2)=PINT(IPI,2)
C    Set VECP, this is arbitrary as the velocity/intensity at P
C     is not sought.
        VECP(1)=ONE
        VECP(2)=ZERO

C    Initialise SUMPHI to zero
        SUMPHI=PFFPHI(IPI)

C    Loop(ISE) through the elements
        DO 850 JSE=1,NSE
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
                
C     Call H2LC.
          CALL H2LC(CK,P,VECP,QA,QB,LPONEL,
     *     MAXNQ,NQOFF,AQOFF,WQOFF,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          IF (.NOT.LSOL) THEN
            WKSPC3(IPI,JSE)=DISLK
            WKSPC4(IPI,JSE)=DISMK
          END IF

C     Accumulate phi 
          IF (LSOL) SUMPHI=SUMPHI+DISLK*SVEL(JSE)-DISMK*SPHI(JSE)

C      Close loop (JSE) through the elements
850     CONTINUE

        PIPHI(IPI)=SUMPHI

C     Close loop(IPI) through the interior points
800   CONTINUE

      END


C ----------------------------------------------------------------------

C Subordinate routines for AIBEM2
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


C Subroutines required for H2LC (not in file H2LC.FOR) 
C  Subroutine for returning the square root.
       REAL*8 FUNCTION FNSQRT(X)
       REAL*8 X
       FNSQRT=SQRT(X)
       END

C  Subroutine for returning the natural logarithm.
       REAL*8 FUNCTION FNLOG(X)
       REAL*8 X
       FNLOG=LOG(X)
       END

C---------------------------------------------------------------------
      
