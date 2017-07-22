C***************************************************************
C            Subroutine AEBEM3 by Stephen Kirkup                     
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AEBEM3.FOR 
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************

C
C This subroutine computes the solution to the three-dimensional 
C Helmholtz equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C in the domain exterior to a closed boundary.
C
C The boundary (S) is defined (approximated) by a set of planar 
C triangular elements. The domain of the equation is exterior to the
C boundary.
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
C The following diagram shows how the subroutine is to be used. 
C
C                                   ....................................
C                                   :                                  :
C                                   :                                  :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|      AEBEM3             |   :
C      |(e.g. aebem3_t.for) |       :     |                        |   :
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
C             : (geom3d.for) :---<---:   | H3LC |   :   :  | CGLS |    : 
C             :              :       :   --------   :   :  --------    :  
C             :..............:       : -------------:   : -------------:  
C                                    : |subordinate|:   : |subordinate|: 
C                                    : | routines  |:   : | routines  |:  
C                                    : -------------:   : -------------: 
C                                    :              :   :              : 
C                                    : (h3lc.for)   :   : (cgls.for)   :
C                                    :..............:   :..............:
C                                    
C
C The contents of the main program must be linked to AEBEM3.FOR, H3LC.FOR
C and CGLS.FOR.
C
C Method of solution
C ------------------
C 
C In the main program, the boundary must be described as a set of
C  elements. The elements are defined by three indices (integers) which
C  label a node or vertex on the boundary. The data structure VERTEX 
C  lists and enumerates the coordinates of the vertices, the data 
C  structure SELV defines each element by indicating the labels for
C  the three nodes that are its vertices and hence enumerates the
C  elements.
C The boundary solution points (the points on the boundary at which 
C  {\phi} (SPHI) and d {\phi}/dn (SVEL) are returned) are at the centres
C  of the elements. The boundary functions {\alpha} (SALPHA), {\beta} 
C  (SBETA) and f (SF) are also defined by their values at the centres
C  of the elements.
C Normally a solution in the domain is required. By listing the 
C  coordinates of all the exterior points in PEXT, the subroutine
C  returns the value of {\phi} at these points in PEPHI.
C


C Notes on the geometric parameters
C ---------------------------------
C (1) Each of the vertices listed in VERTEX should be distinct points
C  with respect to EGEOM.
C (2) The boundary must be complete and closed. 
C (3) The indices of the nodes listed in SELV must be such that they
C  are ordered counter-clockwise around the boundary, when viewed
C  from just outside the boundary at the element.
C (4) The largest element must be no more than 10x the area of the
C  smallest element.

C Notes on the exterior points 
C ----------------------------
C (1) The points in PEXT should lie outside the boundary, as defined
C  by the parameters VERTEX and SELV. Any point lying outside the 
C  boundary will return a corresponding value in PEPHI that is near
C  zero. This property can be if a useful check.

C Notes on the boundary condition
C -------------------------------
C (1) For each i=1..NSE, it must not be the case that both of SALPHA(i)
C  and SBETA(i) are zero

C External modules in external files
C ==================================
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (in file H3LC.FOR)
C subroutine CGLS: Solves a general linear system of equations. 
C  (in file CGLS.FOR)

C External modules provided in the package (this file)
C ====================================================
C subroutine GLT7: Returns the points and weights of the 7-point Gauss-
C  Legendre quadrature rule on the standard triangle.
C real function FNSQRT(X): real X : Returns the square root of X.
C complex function FNEXP(Z): complex Z : Returns the complex exponential
C   of Z.

C The subroutine

      SUBROUTINE AEBEM3(K,
     *                 MAXNV,NV,VERTEX,MAXNSE,NSE,SELV,
     *                 MAXNPE,NPE,PEXT,
     *                 SALPHA,SBETA,SF,SFFPHI,SFFVEL,PFFPHI,
     *                 LSOL,LVALID,EGEOM,MU,
     *                 SPHI,SVEL,PEPHI,
     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,
     *                 WKSPC5,WKSPC6,WKSPC7)
      PARAMETER (MAXNQ=100)

C  Wavenumber
      REAL*8     K

C  Boundary geometry
C   Limit on the number of vertices on S
      INTEGER    MAXNV
C   The number of vertices on S
      INTEGER    NV
C   The coordinates of the vertices on S
      REAL*8     VERTEX(MAXNV,3)
C   Limit on the number of elements describing S
      INTEGER    MAXNSE
C   The number of elements describing S
      INTEGER    NSE
C   The indices of the vertices describing each element
      INTEGER    SELV(MAXNSE,3)
      
C  Exterior points at which the solution is to be observed
C   Limit on the number of points exterior to the boundary where 
C    solution is sought
      INTEGER    MAXNPE
C   The number of exterior points
      INTEGER    NPE
C   Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,3)

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
      REAL*8     DIST3
      REAL*8     AREA

C  Constants
C   Real scalars: 0, 1, 2, half, pi
      REAL*8 ZERO,ONE,TWO,THREE,HALF,THIRD,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Wavenumber in complex form
      COMPLEX*16 CK

C  Geometrical description of the boundary
C   Elements counter
      INTEGER    ISE,JSE
C   The points exterior to the boundary where the solution is sought 
      INTEGER    IPE
C   Parameters for H3LC
      REAL*8     P(3),PA(3),PB(3),PC(3),QA(3),QB(3),QC(3),VECP(3)
      LOGICAL    LPONEL


C  Quadrature rule information
C   [Note that in this program two quadrature rules are used: one for
C    the case when the point P lies on the element (LPONEL=.TRUE.) and
C    one for the case when P does not lie on the element. In general,
C    it is more efficient to define a larger set of quadrature rules
C    so that a particular rule can be selected for any given point P 
C    and element QA-QB-QC. For example using more quadrature points when
C    the element is large, less when the element is small, more when
C    the element is close to P, less when it is far from P.]
C   Quadrature rule used when LPONEL=.TRUE.
C    Number of quadrature points
      INTEGER    NQON
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQON(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQON(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQON(MAXNQ)
C   Quadrature rule used when LPONEL=.FALSE.
C    Number of quadrature points
      INTEGER    NQOFF
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQOFF(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQOFF(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQOFF(MAXNQ)
C   Quadrature rule parameters for H3LC
C    Actual number of quadrature points
      INTEGER    NQ
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQ(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQ(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQ(MAXNQ)
C   Counter through the quadrature points
      INTEGER    IQ

C  Validation and control parameters for subroutine H3LC
      LOGICAL    LVAL
      REAL*8     EK
      REAL*8     EQRULE
      LOGICAL    LFAIL1
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C  Parameters for subroutine H3LC. 
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

C  Other variables
C   Error flags
      LOGICAL    LERROR
C   Warning flag
      LOGICAL    LWARN
C   Failure flag
      LOGICAL    LFAIL
C   Accumulation of solution {\phi}
      COMPLEX*16 SUMPHI
C   Maximum,minimum sizes of elements
      REAL*8     SIZMAX,SIZMIN,SIZE
C   The `diameter' of the boundary or the maximum distance between any
C    three vertices
      REAL*8     DIAM
      REAL*8     SUMMK
     

C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      THREE=3.0D0
      HALF=ONE/TWO
      THIRD=ONE/THREE
      PI=3.14159265358981
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

C Validation
C ==========

C Validation of parameters of AEBEM3
C ---------------------------------

      IF (LVALID) THEN

C Validation of main paramters
        LERROR=.FALSE.
        IF (K.LT.ZERO) THEN
          WRITE(*,*) 'K = ',K
          WRITE(*,*) 'ERROR(AEBEM3) - K must be positive'
          LERROR=.TRUE.
        END IF
        IF (MAXNV.LT.3) THEN
          WRITE(*,*) 'MAXNV = ',MAXNV
          WRITE(*,*) 'ERROR(AEBEM3) - must have MAXNV>=4'
          LERROR=.TRUE.
        END IF
        IF (NV.LT.3.OR.NV.GT.MAXNV) THEN
          WRITE(*,*) 'NV = ',NV
          WRITE(*,*) 'ERROR(AEBEM3) - must have 4<=NV<=MAXNV'
          LERROR=.TRUE.
        END IF
        IF (MAXNSE.LT.3) THEN
          WRITE(*,*) 'MAXNSE = ',MAXNSE
          WRITE(*,*) 'ERROR(AEBEM3) - must have MAXNSE>=4'
          LERROR=.TRUE.
        END IF
        IF (NSE.LT.3.OR.NSE.GT.MAXNSE) THEN
          WRITE(*,*) 'NSE = ',NSE
          WRITE(*,*) 'ERROR(AEBEM3) - must have 4<=NSE<=MAXNSE'
          LERROR=.TRUE.
        END IF
        IF (MAXNPE.LT.1) THEN
          WRITE(*,*) 'MAXNPE = ',MAXNPE
          WRITE(*,*) 'ERROR(AEBEM3) - must have MAXNPE>=1'
          LERROR=.TRUE.
        END IF
        IF (NPE.LT.0.OR.NPE.GT.MAXNPE) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(AEBEM3) - must have 3<=NPE<=MAXNPE'
          LERROR=.TRUE.
        END IF
        IF (EGEOM.LE.ZERO) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(AEBEM3) - EGEOM must be positive'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error(s) found in the main parameters of AEBEM3'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF


C Check PI
      IF (LVALID) THEN
        LERROR=.FALSE.
        IF (ABS(PI-4.0D0*ATAN(ONE)).LT.EGEOM) THEN 
          WKSPC5(MAXNSE)=115
          WKSPC6(MAXNSE)=107
        ELSE
          LERROR=.TRUE.
        END IF
      END IF    


C Find the diameter DIAM of the boundary
      DIAM=0.0
      DO 100 IV=1,NV-1
        PA(1)=VERTEX(IV,1)
        PA(2)=VERTEX(IV,2)
        PA(3)=VERTEX(IV,3)
        DO 110 JV=IV+1,NV
          PB(1)=VERTEX(JV,1)
          PB(2)=VERTEX(JV,2)
          PB(3)=VERTEX(JV,3)
          DIAM=MAX(DIAM,DIST3(PA,PB))
110     CONTINUE
100   CONTINUE

      IF (LVALID) THEN
        LERROR=.FALSE.


C Check that EGEOM is not too large
        IF (EGEOM.GT.DIAM/100.0D0) THEN
          WRITE(*,*) 'EGEOM = ',EGEOM
          WRITE(*,*) 'ERROR(AEBEM3) - EGEOM is set too large'
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
          PA(3)=VERTEX(IV,3)
          DO 140 JV=IV+1,NV
            PB(1)=VERTEX(JV,1)
            PB(2)=VERTEX(JV,2)
            PB(3)=VERTEX(JV,3)
            IF (ABS(PA(1)-PB(1)).LT.EGEOM) THEN
              IF (ABS(PA(2)-PB(2)).LT.EGEOM) THEN
                IF (ABS(PA(3)-PB(3)).LT.EGEOM) THEN
                  WRITE(*,*) 'Vertices ',IV,JV,' are not distinct'
                  LWARN=.TRUE.
                END IF
              END IF
            END IF
140       CONTINUE
130     CONTINUE
        IF (LWARN) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AEBEM3) - Vertices (see above) coincide'
          STOP
        END IF
      END IF          


C Check that the elements are not of disproportionate sizes
      IF (LVALID) THEN
        SIZMAX=ZERO
        SIZMIN=DIAM**2
        DO 150 ISE=1,NSE
          QA(1)=VERTEX(SELV(ISE,1),1)
          QA(2)=VERTEX(SELV(ISE,1),2)
          QA(3)=VERTEX(SELV(ISE,1),3)
          QB(1)=VERTEX(SELV(ISE,2),1)
          QB(2)=VERTEX(SELV(ISE,2),2)
          QB(3)=VERTEX(SELV(ISE,2),3)
          QC(1)=VERTEX(SELV(ISE,3),1)
          QC(2)=VERTEX(SELV(ISE,3),2)
          QC(3)=VERTEX(SELV(ISE,3),3)
          SIZE=AREA(QA,QB,QC)
          SIZMAX=MAX(SIZMAX,SIZE)
          SIZMIN=MIN(SIZMIN,SIZE)
150     CONTINUE
        IF (SIZMAX.GT.10.0D0*SIZMIN) THEN
          WRITE(*,*) 'WARNING(AEBEM3) - Elements of disproportionate'
          WRITE(*,*) ' sizes'
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
          WRITE(*,*) 'ERROR(AEBEM3) - at most one of SALPHA(i),SBETA(i)'
          WRITE(*,*) ' may be zero for all i'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF
        

C Set the wavenumber in complex form
      CK=CMPLX(K,ZERO)


C Set up validation and control parameters
C  Switch off the validation of H3LC
      LVAL=.FALSE.
C  Set EK
      EK=1.0D-6
C  Set EQRULE
      EQRULE=1.0D-6

C Set up the quadrature rule(s).
C  Set up quadrature rule for the case when P is not on the element.
C   Set up 8 point Gauss-Legendre rules
      CALL GLT7(MAXNQ,NQOFF,WQOFF,XQOFF,YQOFF)
C  Set up quadrature rule for the case when P is on the element.

C   Set up quadrature rule data. If LPONEL is false then use the standard
C    Gaussian quadrature rule above. If LPONEL is true then a
C    quadrature rule with 3 times as many points is used, this is made
C    up from three standard quadrature rules with the quadrature points
C    translated to the three triangles that each have the cetroid and two
C    of the original vertices as its vertices.
      NQON=3*NQOFF
      DO 330 IQ=1,NQOFF
        XQON(IQ)=XQOFF(IQ)*THIRD+YQOFF(IQ)
        YQON(IQ)=XQOFF(IQ)*THIRD
        WQON(IQ)=WQOFF(IQ)/THREE
        XQON(IQ+NQOFF)=XQOFF(IQ)*THIRD
        YQON(IQ+NQOFF)=XQOFF(IQ)*THIRD+YQOFF(IQ)
        WQON(IQ+NQOFF)=WQOFF(IQ)/THREE
        XQON(IQ+2*NQOFF)=THIRD*(ONE+TWO*XQOFF(IQ)-YQOFF(IQ))
        YQON(IQ+2*NQOFF)=THIRD*(ONE-XQOFF(IQ)+TWO*YQOFF(IQ))
        WQON(IQ+2*NQOFF)=WQOFF(IQ)/THREE
330   CONTINUE


C Validation that the surface is closed
      IF (LVALID) THEN
        PA(1)=VERTEX(SELV(1,1),1)
        PA(2)=VERTEX(SELV(1,1),2)
        PA(3)=VERTEX(SELV(1,1),3)
        PB(1)=VERTEX(SELV(1,2),1)
        PB(2)=VERTEX(SELV(1,2),2)
        PB(3)=VERTEX(SELV(1,2),3)
        PC(1)=VERTEX(SELV(1,3),1)
        PC(2)=VERTEX(SELV(1,3),2)
        PC(3)=VERTEX(SELV(1,3),3)
        P(1)=(PA(1)+PB(1)+PC(1))/THREE
        P(2)=(PA(2)+PB(2)+PC(2))/THREE
        P(3)=(PA(3)+PB(3)+PC(3))/THREE
        VECP(1)=0.0D0
        VECP(2)=0.0D0
        VECP(3)=1.0D0
        SUMMK=0.0D0
        DO 180 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)
C     Set LPONEL
          LPONEL=(JSE.EQ.1)

C     Only the Mk operator is are required. Set LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CZERO,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQON,XQON,YQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          
          SUMMK=SUMMK+DISMK
180     CONTINUE
        IF (ABS(SUMMK-0.5D0).LT.0.1) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(AEBEM3) - in geometry'
          WRITE(*,*) ' The boundary could be oriented wrongly'
          WRITE(*,*) '  On the outer boundary arrange panels'
     *     // 'in clockwise order'
          WRITE(*,*) '  If there are inner boundaries arrange the'
     *     // 'panels in anticlockwise order'
          STOP
        END IF
        IF (ABS(SUMMK+0.5D0).GT.0.1) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AEBEM3) - in geometry'
          WRITE(*,*) ' The boundary panels may be arranged incorrectly'
        END IF
      END IF  


C Validation that the points in PEXT are exterior points
      IF (LVALID) THEN
        DO IPE=1,NPE
        P(1)=PEXT(IPE,1)
        P(2)=PEXT(IPE,2)
        P(3)=PEXT(IPE,3)
        VECP(1)=0.0D0
        VECP(2)=0.0D0
        VECP(3)=1.0D0
        SUMMK=0.0D0
        DO 210 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)
C     Set LPONEL
          LPONEL=.FALSE.

C     Only the Mk operator is required. Set LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CZERO,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQON,XQON,YQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          
          SUMMK=SUMMK+DISMK
210     CONTINUE

        IF (ABS(SUMMK).GT.0.25) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(AEBEM3) - The observation point'
          WRITE(*,*) ' (',P(1),',',P(2),',',P(3),')'
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
        PA(3)=VERTEX(SELV(ISE,1),3)
        PB(1)=VERTEX(SELV(ISE,2),1)
        PB(2)=VERTEX(SELV(ISE,2),2)
        PB(3)=VERTEX(SELV(ISE,2),3)
        PC(1)=VERTEX(SELV(ISE,3),1)
        PC(2)=VERTEX(SELV(ISE,3),2)
        PC(3)=VERTEX(SELV(ISE,3),3)
        P(1)=(PA(1)+PB(1)+PC(1))/THREE
        P(2)=(PA(2)+PB(2)+PC(2))/THREE
        P(3)=(PA(3)+PB(3)+PC(3))/THREE
C    Set VECP to the normal on the boundary of the element at P
        CALL NORM3(PA,PB,PC,VECP)
C    Loop(ISE) through the elements
        DO 520 JSE=1,NSE
C     Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)

C     Set LPONEL
          IF (ISE.EQ.JSE) THEN
            LPONEL=.TRUE.
          ELSE
            LPONEL=.FALSE.
          END IF

C     Select quadrature rule for H3LC
C   :  Select the quadrature rule XQON-WQON in the case when the
C   :   point p lies on the element, otherwise select XQOFF-WQOFF
C      [Note that the overall method would benefit from selecting from
C       a wider set of quadrature rules, and an appropriate method
C       of selection]
          IF (LPONEL) THEN
            NQ=NQON
            DO 600 IQ=1,NQ
              XQ(IQ)=XQON(IQ)
              YQ(IQ)=YQON(IQ)
              WQ(IQ)=WQON(IQ)
600         CONTINUE
          ELSE 
            NQ=NQOFF
            DO 610 IQ=1,NQ
              XQ(IQ)=XQOFF(IQ)
              YQ(IQ)=YQOFF(IQ)
              WQ(IQ)=WQOFF(IQ)
610         CONTINUE
          END IF

C   All operators are required
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.TRUE.
          LNK=.TRUE.

C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQ,XQ,YQ,WQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

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
        P(3)=PEXT(IPE,3)
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
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)

C     All the points do not lie on the boundary hence LPONEL=.FALSE.
          LPONEL=.FALSE.              

C     Only Lk, Mk operators are required. Set LLK,LMK true, 
C      LMKT,LNK false. 
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.
                

C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQOFF,XQOFF,YQOFF,WQOFF,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

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

C Subordinate routines for AEBEM3
C ==============================

C ----------------------------------------------------------------------
C           Subroutine GLT7.FOR by www.numerical-methods.com           |
C ----------------------------------------------------------------------
C
C Subroutine GLT7 assigns the weights and points of a 7 point Gaussian
C quadrature rule defined on the standard triangle.
C
C SUBROUTINE GLT7(MAXNQ, NQ, WQ, XQ, YQ)
C integer  maxnq: the maximimum number of weights/points
C integer     nq: the number of weights/points
C real        wq: the weights
C real        xq: the x-coordinates of the points
C real        yq: the y-coordinates of the points
C
C Source of the code: http://www.numerical-methods.com/fortran/GLT7.FOR
C Source of the user-guide: http://www.numerical-methods.com/fortran/
C  glt7.htm
C
C Licence: This is 'open source'; the software may be used and applied
C  within other systems as long as its provenance is appropriately
C  acknowledged. See the GNU Licence http://www.gnu.org/licenses/lgpl.txt
C  for more information or contact webmaster@numerical-methods.com


      SUBROUTINE GLT7(MAXNQ,NQ,WQ,XQ,YQ)
      INTEGER MAXNQ,NQ
      REAL*8 WQ(MAXNQ),XQ(MAXNQ),YQ(MAXNQ)
      NQ=7
      WQ(1)=0.225000000000000D0
      WQ(2)=0.125939180544827D0
      WQ(3)=0.125939180544827D0
      WQ(4)=0.125939180544827D0
      WQ(5)=0.132394152788506D0
      WQ(6)=0.132394152788506D0
      WQ(7)=0.132394152788506D0

      XQ(1)=0.333333333333333D0
      XQ(2)=0.797426985353087D0
      XQ(3)=0.101286507323456D0
      XQ(4)=0.101286507323456D0
      XQ(5)=0.470142064105115D0
      XQ(6)=0.470142064105115D0
      XQ(7)=0.059715871789770D0

      YQ(1)=0.333333333333333D0
      YQ(2)=0.101286507323456D0
      YQ(3)=0.797426985353087D0
      YQ(4)=0.101286507323456D0
      YQ(5)=0.470142064105115D0
      YQ(6)=0.059715871789770D0
      YQ(7)=0.470142064105115D0

      END

C Subroutines required for H3LC (not in file H3LC.FOR) 
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
