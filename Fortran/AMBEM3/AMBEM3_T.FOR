C***************************************************************
C     Test program for subroutine AMBEM3 by Stephen Kirkup           
C***************************************************************
C  
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AMBEM3_T.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This program is a test for the subroutine AMBEM3. The program computes
C  the solution to an acoustic/Helmholtz eigenvalue problem interior to
C  a sphere by interpolating the matrices that arise from the boundary 
C  element method.
C
C Background
C ----------
C
C The Helmholtz problem arises when harmonic solutions of the wave 
C  equation
C                                     2
C         __ 2                 1     d   {\Psi}(p,t)
C         \/   {\Psi}(p,t) -  ----   ---------------   =  0
C                               2        2
C                              c      d t
C                
C  are sought, where {\Psi}(p,t) is the scalar time-dependent velocity
C  potential. In the cases where {\Psi} is periodic, it may be 
C  approximated as a set of frequency components that may be analysed
C  independently. For each frequency a component of the form
C
C                      {\phi}(p) exp(i {\omega} t)
C
C  (where {\omega} = 2 * {\pi} * frequency) the wave equation can be
C  reduced to the Helmholtz equation
C
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0  
C
C  where k (the wavenumber) = {\omega}/c (c=speed of sound in the 
C  medium). {\phi} is known as the velocity potential.
C
C For the eigenvalue problem, the domain lies interior to a closed 
C  boundary S. The boundary condition may be Dirichlet, Robin or 
C  Neumann. For the eigenvalue problem the boundary condidition is
C  assumed to be homogeneous and have the following general form
C
C            {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = 0
C    
C  where {\phi}(q) is the velocity potential at the point q on S, v(q) 
C  is the derivative of {\phi} with respect to the outward normal to S 
C  at q and {\alpha}, and {\beta} are complex-valued functions defined
C   on S. 
C
C Subroutine AMBEM3 accepts the range of wavenumbers, the degree of the
C  interpolating polynomial, a description of the boundary of the domain
C  and the position of the interior points where the solution ({\phi})
C  is sought, the boundary condition and returns the solution ({\phi} 
C  and v) on S and the value of {\phi} at the interior points.
C

C----------------------------------------------------------------------

C The PARAMETER statement
C -----------------------
C There are four components in the PARAMETER statement.
C integer MAXNS  : The limit on the number of boundary elements.
C integer MAXNV   : The limit on the number of vertices.
C integer MAXNK   : The limit on the number of interpolation points.
C integer MAXNEIG  : The limit on the number of eigenvalues (resonant
C  frequencies) that can be found in the range [KA,KB].
C integer MAXNPI  : The limit on the number of interior points.


C External modules related in the package
C ---------------------------------------
C subroutine AMBEM3: Subroutine for solving the Helmholtz eigenvalue
C  problem (file AMBEM3.FOR contains AMBEM3)
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (file H3LC.FOR contains H2LC and subordinate routines)
C subroutine INTEIG: Finds the eigenvalues of the interpolated matrix
C  (file INTEIG.FOR contains INTEIG and subordinate routines)


C The program 

      PROGRAM AMBEM3T
      IMPLICIT NONE

C VARIABLE DECLARATION
C --------------------

C  PARAMETERs for storing the limits on the dimension of arrays
C   Limit on the number of elements
      INTEGER    MAXNS
      PARAMETER (MAXNS=40)
C   Limit on the number of vertices (equal to the number of elements)
      INTEGER    MAXNV
      PARAMETER (MAXNV=30)
C   Limit on the number of points interior to the boundary, where 
C    acoustic properties are sought
      INTEGER    MAXNPI
      PARAMETER (MAXNPI=10)
      
      INTEGER    MAXNEIG
      PARAMETER (MAXNEIG=10)

      INTEGER    MAXNK
      PARAMETER (MAXNK=4)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,THREE,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Properties of the acoustic medium
C   The speed of sound [standard unit: metres per second]
      REAL*8     CVAL

      INTEGER    NK

C  Geometrical description of the boundary(ies)
C   Number of elements and counter
      INTEGER    NS,IS
C   Number of collocation points (on S) and counter
      INTEGER    NSP,ISP
C   Number of vetices and counter
      INTEGER    NV,IV
C   Index of nodal coordinate for defining boundaries (standard unit is 
C    metres)
      REAL*8     VERTEX(MAXNV,3)
C   The three nodes that define each element on the boundaries
      INTEGER    SELV(MAXNS,3)
C   The points interior to the boundary(ies) where the acoustic 
C    properties are sought and the directional vectors at those points.
C    [Only necessary if an interior solution is sought.]
C    Number of interior points and counter
      INTEGER    NPI,IPI
C    Coordinates of the interior points
      REAL*8     PINT(MAXNPI,3)

C  Number of test problems and counter
      INTEGER    NTEST

C   Data structures that are used to define each test problem in turn
C    and are input parameters to AMBEM3.
C    SALPHA(j) is assigned the value of {\alpha} at the centre of the 
C     j-th element.
      COMPLEX*16 SALPHA(MAXNS)
C    SBETA(j) is assigned the value of {\beta} at the centre of the 
C     j-th element.
      COMPLEX*16 SBETA(MAXNS)

C  Validation and control parameters for AMBEM3
C   Validation switch
      LOGICAL    LVALID
C   The maximum absolute error in the parameters that describe the
C    geometry of the boundary.
      REAL*8     EGEOM



C Output from subroutine AMBEM3
C  The velocity potential (phi - the solution) at the centres of the 
C   elements
      COMPLEX*16 SPHI(MAXNEIG,MAXNS)
C  The normal derivative of the velocity potential at the centres of the
C    elements
      COMPLEX*16 SVEL(MAXNEIG,MAXNS)
C  The velocity potential (phi - the solution) at interior points
      COMPLEX*16 PIPHI(MAXNEIG,MAXNPI)


C  Counter through the x,y coordinates
      INTEGER    ICOORD


C  The coordinates of the centres of the elements  
      REAL*8     SELCNT(MAXNS,3)






C Workspace for AMBEM3
C  Working space 
      COMPLEX*16 WKSPC1(MAXNK,MAXNS,MAXNS)
      COMPLEX*16 WKSPC2(MAXNK,MAXNS,MAXNS)
      COMPLEX*16 WKSPC3(MAXNK,MAXNS,MAXNS)
      LOGICAL    WKSPC4(MAXNS)
      REAL*8     WKSPC5(MAXNK)
      COMPLEX*16 WKSPC6((MAXNK-1)*MAXNS,MAXNS)
      LOGICAL    WKSPC7(MAXNS)

      COMPLEX*16 WKSP00((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      COMPLEX*16 WKSP01((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      COMPLEX*16 WKSP02((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      REAL*8     WKSP03((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      REAL*8     WKSP04((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      REAL*8     WKSP05((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      REAL*8     WKSP06((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      REAL*8     WKSP07((MAXNK-1)*MAXNS)
      REAL*8     WKSP08((MAXNK-1)*MAXNS)
      REAL*8     WKSP09(MAXNK)
      INTEGER    WKSP10((MAXNK-1)*MAXNS)
      COMPLEX*16 WKSP11((MAXNK-1)*MAXNS,(MAXNK-1)*MAXNS)
      COMPLEX*16 WKSP12((MAXNK-1)*MAXNS)



C  Other variables
      REAL*8     EPS

      COMPLEX*16 EIGVAL(MAXNEIG,MAXNS)

      REAL*8 KA,KB
      INTEGER NEIG

C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      THREE=3.0D0
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

      EPS=1.0E-10


C Describe the nodes and elements that make up the boundary
C  :The unit sphere, centred at the origin is divided 
C  : into NS=36 uniform elements. VERTEX and SELV are defined 
C  : anti-clockwise around the boundary so that the normal to the 
C  : boundary is assumed to be outward
C  :Set up nodes
C  : Set NS, the number of elements
      NS=36
C  : Set NV, the number of vertices (equal to the number of elements)
      NV=NS
C  : Set coordinates of the nodes

C   Set up VERTEX, the coordinates of the vertices of the elements
      NV=20
      DATA ((VERTEX(IV,ICOORD),ICOORD=1,3),IV=1,20)
     * / 0.000D0, 0.000D0, 1.000D0,
     *   0.000D0, 0.745D0, 0.667D0,
     *   0.645D0, 0.372D0, 0.667D0,
     *   0.645D0,-0.372D0, 0.667D0,
     *   0.000D0,-0.745D0, 0.667D0,
     *  -0.645D0,-0.372D0, 0.667D0,
     *  -0.645D0, 0.372D0, 0.667D0,
     *   0.500D0, 0.866D0, 0.000D0,
     *   1.000D0, 0.000D0, 0.000D0,
     *   0.500D0,-0.866D0, 0.000D0,
     *  -0.500D0,-0.866D0, 0.000D0,
     *  -1.000D0, 0.000D0, 0.000D0,
     *  -0.500D0, 0.866D0, 0.000D0,
     *   0.000D0, 0.745D0,-0.667D0,
     *   0.645D0, 0.372D0,-0.667D0,
     *   0.645D0,-0.372D0,-0.667D0,
     *   0.000D0,-0.745D0,-0.667D0,
     *  -0.645D0,-0.372D0,-0.667D0,
     *  -0.645D0, 0.372D0,-0.667D0,
     *   0.000D0, 0.000D0,-1.000D0/


C  : Set nodal indices that describe the elements of the boundarys.
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in SELV dictates that the normal is outward from the 
C  :  boundary into the acoustic domain.
      DATA ((SELV(IS,ICOORD),ICOORD=1,3),IS=1,36)
     * / 1, 3, 2,    1, 4, 3,    1, 5, 4,    1, 6, 5,
     *   1, 7, 6,    1, 2, 7,    2, 3, 8,    3, 9, 8,
     *   3, 4, 9,    4,10, 9,    4, 5,10,    5,11,10,
     *   5, 6,11,    6,12,11,    6, 7,12,    7,13,12,
     *   7, 2,13,    2, 8,13,    8,15,14,    8, 9,15,
     *   9,16,15,    9,10,16,   10,17,16,   10,11,17,
     *  11,18,17,   11,12,18,   12,19,18,   12,13,19,
     *  13,14,19,   13, 8,14,   14,15,20,   15,16,20,
     *  16,17,20,   17,18,20,   18,19,20,   19,14,20/


C Set the centres of the elements, the collocation points
      DO IS=1,NS
        SELCNT(IS,1)=(VERTEX(SELV(IS,1),1)
     *   +VERTEX(SELV(IS,2),1)+VERTEX(SELV(IS,3),1))/THREE
        SELCNT(IS,2)=(VERTEX(SELV(IS,1),2)
     *   +VERTEX(SELV(IS,2),2)+VERTEX(SELV(IS,3),2))/THREE
        SELCNT(IS,3)=(VERTEX(SELV(IS,1),3)
     *   +VERTEX(SELV(IS,2),3)+VERTEX(SELV(IS,3),3))/THREE
      END DO



C Set the points in the acoustic domain where the acoustic properties
C  are sought, PINT. 
C : Let them be the points (0.000,0.000), (0.000,0.500), (0.250,0.250).
      NPI=10
      DATA ((PINT(IPI,ICOORD),ICOORD=1,3),IPI=1,10)
     *  /  0.000D0,     0.000D0,  0.000D0,
     *     0.250D0,     0.000D0,  0.000D0,
     *     0.000D0,     0.250D0,  0.000D0,
     *     0.000D0,     0.000D0,  0.250D0,
     *     0.500D0,     0.000D0,  0.000D0,
     *     0.000D0,     0.500D0,  0.000D0,
     *     0.000D0,     0.000D0,  0.500D0,
     *     0.750D0,     0.000D0,  0.000D0,
     *     0.000D0,     0.750D0,  0.000D0,
     *     0.000D0,     0.000D0,  0.750D0/


C The number of points on the boundary is equal to the number of 
C  elements
      NSP=NS
        
C Set up test problems
C  :Set the number of test problems
      NTEST=2


C Open output file
      OPEN(10,FILE='AMBEM3.OUT',STATUS='UNKNOWN')


C  TEST PROBLEM 1
C  ==============
C  Properties of the acoustic medium. C the propagation velocity
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second]
      CVAL=344.0D0
C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALPHA and SBETA at the collocation points
C   :In this case a Dirichlet (phi-valued) boundary condition
      DO 160 ISP=1,NSP
        SALPHA(ISP)=CONE
        SBETA(ISP)=CZERO
160   CONTINUE

C Search for solutions in the range k in [3,4]
      KA=3.0D0
      KB=4.0D0

C Number of interpolation points (degree+1 of polynomial interpolant)
      NK=4

C Set up validation and control parameters
C  :Switch on the validation of AMBEM3
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6


      CALL AMBEM3(KA,KB,MAXNK,NK,
     *           MAXNEIG,
     *           MAXNV,NV,VERTEX,MAXNS,NS,SELV,
     *           MAXNPI,NPI,PINT,
     *           SALPHA,SBETA,
     *           LVALID,EGEOM,
     *           NEIG,EIGVAL,SPHI,SVEL,PIPHI,
     *           WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6,WKSPC7,
     *           WKSP00,WKSP01,WKSP02,WKSP03,WKSP04,WKSP05,
     *           WKSP06,WKSP07,WKSP08,WKSP09,WKSP10,WKSP11,WKSP12)
     

      CALL OUTP3(1,MAXNEIG,NEIG,MAXNPI,NPI,CVAL,PINT,
     * EIGVAL,PIPHI)


C  TEST PROBLEM 2
C  ==============
C  Properties of the acoustic medium. C the propagation velocity
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second]
      CVAL=344.0D0
C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALPHA and SBETA at the collocation points
C   :In this case a Dirichlet (phi-valued) boundary condition
      DO 180 ISP=1,NSP
        SALPHA(ISP)=CZERO
        SBETA(ISP)=CONE
180   CONTINUE

C Search for solutions in the range k in [2,3]
      KA=2.0D0
      KB=3.0D0

C Number of interpolation points (degree+1 of polynomial interpolant)
      NK=3

C Set up validation and control parameters
C  :Switch on the validation of AMBEM3
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6


      CALL AMBEM3(KA,KB,MAXNK,NK,
     *           MAXNEIG,
     *           MAXNV,NV,VERTEX,MAXNS,NS,SELV,
     *           MAXNPI,NPI,PINT,
     *           SALPHA,SBETA,
     *           LVALID,EGEOM,
     *           NEIG,EIGVAL,SPHI,SVEL,PIPHI,
     *           WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6,WKSPC7,
     *           WKSP00,WKSP01,WKSP02,WKSP03,WKSP04,WKSP05,
     *           WKSP06,WKSP07,WKSP08,WKSP09,WKSP10,WKSP11,WKSP12)

      CALL OUTP3(2,MAXNEIG,NEIG,MAXNPI,NPI,CVAL,PINT,
     * EIGVAL,PIPHI)

      CLOSE(10)

      END

      SUBROUTINE OUTP3(ITEST,MAXNEIG,NEIG,MAXNPI,NPI,CVAL,PINT,
     * EIGVAL,PIPHI)
      INTEGER    ITEST
      INTEGER    MAXNEIG
      INTEGER    NEIG
      INTEGER    MAXNPI
      INTEGER    NPI
      REAL*8     CVAL
      REAL*8     PINT(MAXNPI,3)
      COMPLEX*16 EIGVAL(MAXNEIG)
      COMPLEX*16 PIPHI(MAXNEIG,MAXNPI)

      REAL*8     PI
      COMPLEX*16 PHIMAX

      PI=4.0D0*ATAN(1.0D0)

      DO 100 IEIG=1,NEIG
        WRITE(10,*)
        WRITE(10,*) 'TEST PROBLEM ',ITEST
        WRITE(10,*) 'Resonant wavenumber = ',DBLE(EIGVAL(IEIG))
        WRITE(10,*) 'Resonant frequency  = ',
     *   DBLE(EIGVAL(IEIG))*CVAL/2.0D0/PI,'Hz'
        PHIMAX=PIPHI(IEIG,1)
        DO 130 IPI=2,NPI
          IF (ABS(PHIMAX).LT.ABS(PIPHI(IEIG,IPI)))
     *     PHIMAX=PIPHI(IEIG,IPI)
130     CONTINUE
        DO 140 IPI=1,NPI
          PIPHI(IEIG,IPI)=PIPHI(IEIG,IPI)/PHIMAX
140     CONTINUE
        WRITE(10,*) '     X         Y          Z'
     *   //'                  potential'
        DO 110 IPI=1,NPI
          WRITE(10,999) PINT(IPI,1),PINT(IPI,2),PINT(IPI,3),
     *     DBLE(PIPHI(IEIG,IPI)),AIMAG(PIPHI(IEIG,IPI))
110     CONTINUE
100   CONTINUE


999   FORMAT(3F10.4,2F16.8)

      END
