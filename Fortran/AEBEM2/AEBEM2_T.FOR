C***************************************************************
C    Test program for subroutine AEBEM2 by Stephen Kirkup           
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AEBEM2_T.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************

C
C This program is a test for the subroutine AEBEM2. The program computes
C  the solution to an acoustic/Helmholtz problem exterior to a square 
C  by the boundary element method.
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
C For the exterior problem, the domain lies exterior to a closed 
C  boundary S. The boundary condition may be Dirichlet, Robin or 
C  Neumann. It is assumed to have the following general form
C
C            {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C  where {\phi}(q) is the velocity potential at the point q on S, v(q) 
C  is the derivative of {\phi} with respect to the outward normal to S 
C  at q and {\alpha}, {\beta} and f are complex-valued functions defined
C   on S. 
C
C Subroutine AEBEM2 accepts the wavenumber, a description of the 
C  boundary of the domain and the position of the exterior points
C  where the solution ({\phi}) is sought, the boundary condition and
C  returns the solution ({\phi} and v) on S and the value of {\phi}
C  at the exterior points.
C

C The test problems
C -----------------
C
C In this test the domain is a exterior to a square of side 0.1 (metre).
C  The acoustic medium is air (at 20 celcius and 1 atmosphere, c=344.0 
C  (metres per second), density {\rho}=1.205 (kilograms per cubic metre)
C  and the  solution to the problem with a Dirichlet boundary condition 
C  ({\alpha}=1, {\beta}=0) and with a Neumann boundary condition 
C  ({\alpha}=0, beta=1) are sought. For both problems the frequency is
C  400Hz (hence specifying k).
C Assuming the boundary condition is derived through assuming it is 
C  equivalent to a point source at the centre of the square is clearly 
C  a solution of the Helmholtz equation.
C
C The form of the test problems are illustrated in the following
C  diagram.
C
C         ^
C       y |
C         |                    ^n
C                              |
C        0.1   --------------------------------------
C              |B                                  C|
C              |                                    |
C              |                                    |
C              |                                    |
C          <-  |                                    | ->
C           n  |                                    |  n
C              |                  *                 |
C              |           equivalent source        |
C              |                                    |         
C              |                                    |
C              |                                    |
C              |                                    |
C              |A                                  D|
C        0.0   --------------------------------------
C                              | n
C              0.0            \ /                0.1  ---> x
C
C Note that on side CD  v = d {\phi}/dx, on AB  v = - d {\phi}/dx,
C  on BC  v = d {\phi}/dy and on DA  v = - d{\phi}/dy: the sign
C  is negative if the normal is in the negative x or y direction.
C
C The boundary is described by a set of NS=32 elements of equal size,
C  so that each side comprises eight elements. The boundary solution
C  points are the centres of the elements. 
C The * shows the position of the equivalent source.

C----------------------------------------------------------------------

C The PARAMETER statement
C -----------------------
C There are four components in the PARAMETER statement.
C integer MAXNS   : The limit on the number of boundary elements.
C integer MAXNV   : The limit on the number of vertices.
C integer MAXNPE  : The limit on the number of exterior points.


C External modules related in the package
C ---------------------------------------
C subroutine AEBEM2: Subroutine for solving the exterior Helmholtz
C  equation (file AEBEM2.FOR contains IBEM2 and subordinate routines)
C subroutine H2LC: Returns the individual discrete Helmholtz integral
C  operators. (file H2LC.FOR contains H2LC and subordinate routines)
C subroutine CGLS: Solves a general linear system of equations.
C  (file CGLS.FOR contains CGSL and subordinate routines)
C subroutine FNHANK: This computes Hankel functions of the first kind
C  and of order zero and one. (e.g. file FNHANK.FOR)
C file GEOM2D contains the set of relevant geometric subroutines

C The program 

      PROGRAM AEBEM2T
      IMPLICIT REAL*8 (A-H,O-Z)

C VARIABLE DECLARATION
C --------------------

C  PARAMETERs for storing the limits on the dimension of arrays
C   Limit on the number of elements
      INTEGER    MAXNS
      PARAMETER (MAXNS=32)
C   Limit on the number of vertices (equal to the number of elements)
      INTEGER    MAXNV
      PARAMETER (MAXNV=MAXNS)
C   Limit on the number of test problems
      INTEGER    MAXTEST
      PARAMETER (MAXTEST=5)
C   Limit on the number of points exterior to the boundary, where 
C    acoustic properties are sought
      INTEGER    MAXNPE
      PARAMETER (MAXNPE=6)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  The reference pressure, used to convert units to decibels.
      REAL*8     PREREF


C  Properties of the acoustic medium
C   The speed of sound [standard unit: metres per second]
      REAL*8     CVAL(MAXTEST)
C   The density [standard unit: kilograms per cubic metre]
      REAL*8     RHOVAL(MAXTEST)

C   Wavenumber parameter for AEBEM2
      REAL*8     K
C   Angular frequency 
      COMPLEX*16 OMEGA

C  Geometrical description of the boundary(ies)
C   Number of elements and counter
      INTEGER    NS,IS
C   Number of collocation points and counter
      INTEGER    NSP,ISP
C   Number of vetices and counter
      INTEGER    NV, IV
C   Index of nodal coordinate for defining boundaries (standard unit is 
C    metres)
      REAL*8     VERTEX(MAXNV,2)
C   The two nodes that define each element on the boundaries
      INTEGER    SELV(MAXNS,2)
C   The points exterior to the boundary(ies) where the acoustic 
C    properties are sought and the directional vectors at those points.
C    [Only necessary if an exteriorsolution is sought.]
C    Number of exterior points
      INTEGER    NPE
C    Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,2)

C  Data structures that contain the parameters that define the test
C   problems
C   The acoustic frequency for each test. FRVAL(i) is assigned the
C    acoustic frequency of the i-th test problem.
      REAL*8     FRVAL(MAXTEST)
C   The wavenumber for each test. KVAL(i) is assigned the wavenumber
C    of the i-th test problem.
      REAL*8     KVAL(MAXTEST)
C   The nature of the boundary condition is specified by assigning 
C    values to the data structures SALVAL and SBEVAL. 
C    SALVAL(i,j) is assigned the value of {\alpha} at the center of the
C     j-th element for the i-th test problem.
      COMPLEX*16 SALVAL(MAXTEST,MAXNS)
C    SBEVAL(i,j) is assigned the value of {\beta} at the center of the
C     j-th element for the i-th test problem.
      COMPLEX*16 SBEVAL(MAXTEST,MAXNS)      
C   The actual boundary condition is specified by assigning values to 
C    the data structure SFVAL. 
C    SFVAL(i,j) is assigned the value of f at the center of the j-th 
C    element for the i-th test problem.
      COMPLEX*16 SFVAL(MAXTEST,MAXNS)
C    The incident potential at the centres of the elements in each test
C     case
      COMPLEX*16 SPHIIN(MAXTEST,MAXNS)
C    The derivative of the incident potential at the centres of the 
C     elements in each test
      COMPLEX*16 SVELIN(MAXTEST,MAXNS)
C    The incident potential at the exterior points 
      COMPLEX*16 PPHIIN(MAXTEST,MAXNPE)

C   Data structures that are used to define each test problem in turn
C    and are input parameters to AEBEM2.
C    SALPHA(j) is assigned the value of {\alpha} at the centre of the 
C     j-th element.
      COMPLEX*16 SALPHA(MAXNS)
C    SBETA(j) is assigned the value of {\beta} at the centre of the 
C     j-th element.
      COMPLEX*16 SBETA(MAXNS)
C    SF(j) is assigned the value of f at the centre of the j-th element.
      COMPLEX*16 SF(MAXNS)

C    The incident potential at the centres of the elements in each test
C     case
      COMPLEX*16 SFFPHI(MAXNS)
C    The derivative of the incident potential at the centres of the 
C     elements in each test
      COMPLEX*16 SFFVEL(MAXNS)
C    The incident potential at the interior points 
      COMPLEX*16 PFFPHI(MAXNPE)

C  Validation and control parameters for AEBEM2
C   Switch for particular solution
      LOGICAL    LSOL
C   Validation switch
      LOGICAL    LVALID
C   The maximum absolute error in the parameters that describe the
C    geometry of the boundary.
      REAL*8     EGEOM
C   The weighting parameter.
      COMPLEX*16 MU

C Output from subroutine AEBEM2
C  The velocity potential (phi - the solution) at the centres of the 
C   elements
      COMPLEX*16 SPHI(MAXNS)
C  The normal derivative of the velocity potential at the centres of the
C    elements
      COMPLEX*16 SVEL(MAXNS)
C  The velocity potential (phi - the solution) at exterior points
      COMPLEX*16 PEPHI(MAXNPE)

C Workspace for AEBEM2
      COMPLEX*16 WKSPC1(MAXNS,MAXNS)
      COMPLEX*16 WKSPC2(MAXNS,MAXNS)
      COMPLEX*16 WKSPC3(MAXNPE,MAXNS)
      COMPLEX*16 WKSPC4(MAXNPE,MAXNS)
      COMPLEX*16 WKSPC5(MAXNS)
      COMPLEX*16 WKSPC6(MAXNS)
      LOGICAL    WKSPC7(MAXNS)

C Number of test problems and counter
      INTEGER    NTEST,ITEST


C   Acoustic properties. These data structures are appended after each
C    execution of AEBEM2 and contain the numerical solution to the test
C    problems. 
C    At the centres of the elements
C     Sound pressure [standard unit: newtons per square metre (or
C      pascals) and phase] 
      COMPLEX*16 SPRESS(MAXTEST,MAXNS)
C     Velocity potential phi
      COMPLEX*16 SPHIVAL(MAXTEST,MAXNS)
C  Velocity (v) [standard unit: metres per second (and phase)]
C   At the centres of the elements 
      COMPLEX*16 SV(MAXTEST,MAXNS)
C     Sound intensity [standard unit: watts per square metre]
      REAL*8     SINTY(MAXTEST,MAXNS)
C    At the exterior points
C     Sound pressure [standard unit: newtons per square metre (or
C      pascals) and phase] 
      COMPLEX*16 EPRESS(MAXTEST,MAXNPE)
C     Velocity potential phi
      COMPLEX*16 EPHIVAL(MAXTEST,MAXNPE)

C  Local storage of pressure, pressure/velocity 
      COMPLEX*16 PRESSURE

C  The coordinates of the centres of the elements  
      REAL*8     SELCNT(MAXNS,2)

C  Variables used in setting up test boundary conditions
      REAL*8     CENTX,CENTY,DRBDN,RR(2)
      COMPLEX*16 H(0:1),KR

C  Other variables
      REAL*8     EPS

C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

      EPS=1.0E-10

C  Reference for decibel scales
      PREREF=2.0D-05


C Describe the nodes and elements that make up the boundary
C  :The square with vertices (0,0), (0.1,0), (0.1,0.1), (0,0.1) is divided 
C  : into NS=32 uniform elements. VERTEX and SELV are defined 
C  : anti-clockwise around the boundary so that the normal to the 
C  : boundary is assumed to be outward
C  :Set up nodes
C  : Set NS, the number of elements
      NS=32
C  : Set NV, the number of vertices (equal to the number of elements)
      NV=NS
C  : Set coordinates of the nodes
      DATA ((VERTEX(IV,I),I=1,2),IV=1,32)
     * / 0.00000000000D0, 0.00000000000D0,
     *   0.00000000000D0, 0.01250000000D0,
     *   0.00000000000D0, 0.02500000000D0,
     *   0.00000000000D0, 0.03750000000D0,
     *   0.00000000000D0, 0.05000000000D0,
     *   0.00000000000D0, 0.06250000000D0,
     *   0.00000000000D0, 0.07500000000D0,
     *   0.00000000000D0, 0.08750000000D0,

     *   0.00000000000D0, 0.10000000000D0,
     *   0.01250000000D0, 0.10000000000D0,
     *   0.02500000000D0, 0.10000000000D0,
     *   0.03750000000D0, 0.10000000000D0,
     *   0.05000000000D0, 0.10000000000D0,
     *   0.06250000000D0, 0.10000000000D0,
     *   0.07500000000D0, 0.10000000000D0,
     *   0.08750000000D0, 0.10000000000D0,

     *   0.10000000000D0, 0.10000000000D0,
     *   0.10000000000D0, 0.08750000000D0,
     *   0.10000000000D0, 0.07500000000D0,
     *   0.10000000000D0, 0.06250000000D0,
     *   0.10000000000D0, 0.05000000000D0,
     *   0.10000000000D0, 0.03750000000D0,
     *   0.10000000000D0, 0.02500000000D0,
     *   0.10000000000D0, 0.01250000000D0,

     *   0.10000000000D0, 0.00000000000D0,
     *   0.08750000000D0, 0.00000000000D0,
     *   0.07500000000D0, 0.00000000000D0,
     *   0.06250000000D0, 0.00000000000D0,
     *   0.05000000000D0, 0.00000000000D0,
     *   0.03750000000D0, 0.00000000000D0,
     *   0.02500000000D0, 0.00000000000D0,
     *   0.01250000000D0, 0.00000000000D0 /

C  :Describe the elements that make up the two boundarys
C  : Set NS, the number of elements
      NS=32
C  : Set nodal indices that describe the elements of the boundarys.
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in SELV dictates that the normal is outward from the 
C  :  boundary into the acoustic domain.
      DATA ((SELV(IE,J),J=1,2),IE=1,32)
     * /  1,  2,
     *    2,  3,
     *    3,  4,
     *    4,  5,
     *    5,  6,
     *    6,  7,
     *    7,  8,
     *    8,  9,

     *    9,  10,
     *   10,  11,
     *   11,  12,
     *   12,  13,
     *   13,  14,
     *   14,  15,
     *   15,  16,
     *   16,  17,

     *   17,  18,
     *   18,  19,
     *   19,  20,
     *   20,  21,
     *   21,  22,
     *   22,  23,
     *   23,  24,
     *   24,  25,

     *   25,  26,
     *   26,  27,
     *   27,  28,
     *   28,  29,
     *   29,  30,
     *   30,  31,
     *   31,  32,
     *   32,  1 /
       
C Set the centres of the elements, the collocation points
      DO 100 IS=1,NS
        SELCNT(IS,1)=(VERTEX(SELV(IS,1),1)
     *   +VERTEX(SELV(IS,2),1))/TWO
        SELCNT(IS,2)=(VERTEX(SELV(IS,1),2)
     *   +VERTEX(SELV(IS,2),2))/TWO
100   CONTINUE


C Set the points in the acoustic domain where the acoustic properties
C  are sought, PEXT. 
C : Let them be the points (0.025,0.025), (0.075,0.025), (0.025,0.075),
C :  (0.075,0.075) and (0.5,0.5).
      NPE=4
      DATA ((PEXT(IPE,ICOORD),ICOORD=1,2),IPE=1,4)
     *  /  0.0000D0,     0.1500D0,
     *     0.0500D0,     0.1500D0,
     *     0.1000D0,     0.1500D0,
     *     0.0500D0,    -0.1000D0 /


C The number of points on the boundary is equal to the number of 
C  elements
      NSP=NS
        
C Set up test problems
C  :Set the number of test problems
      NTEST=3


C  TEST PROBLEM 1
C  ==============
C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL(1)=344.0D0
      RHOVAL(1)=1.205D0

C  :Set acoustic frequency value (hertz) in FRVAL
      FRVAL(1)=400.0D0

C  : Set the wavenumber in KVAL
      KVAL(1)=TWO*PI*FRVAL(1)/CVAL(1)

C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALVAL and SBEVAL at the collocation points
C   :In this case a Dirichlet (phi-valued) boundary condition
      DO 160 ISP=1,NSP
        SALVAL(1,ISP)=CONE
        SBEVAL(1,ISP)=CZERO
160   CONTINUE

C  :The test problem is devised so that \phi=i*h0(kr)/4 
C    where r is the distance between the collocation point
C    and the point at the centre of the square
C   :Set K, the wavenumber
      K=KVAL(1)
C   :Set centre of the square
      CENTX=0.05D0
      CENTY=0.05D0
      DO 170 ISP=1,NSP
        X=SELCNT(ISP,1)
        Y=SELCNT(ISP,2)
        R=SQRT((X-CENTX)**2+(Y-CENTY)**2)
        KR=K*R
        CALL FNHANK(KR,H)
        SFVAL(1,ISP)=CIMAG*H(0)/4.0D0
170   CONTINUE     

C  :There is no incident field 
      DO 180 ISP=1,NSP
        SPHIIN(1,ISP)=0.0D0
        SVELIN(1,ISP)=0.0D0
180   CONTINUE
      DO 190 IPI=1,NPE
        PPHIIN(1,IPI)=0.0D0
190   CONTINUE


C  TEST PROBLEM 2
C  ==============
C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL(2)=344.0D0
      RHOVAL(2)=1.205D0
C  :Set acoustic frequency value (hertz) in FRVAL
      FRVAL(2)=400.0D0

C  : Set the wavenumber in KVAL
      KVAL(2)=TWO*PI*FRVAL(2)/CVAL(2)

C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALVAL and SBEVAL at the collocation points
C   :In this case a Neumann (v-valued) boundary condition
      DO 200 ISP=1,NSP
        SALVAL(2,ISP)=CZERO
        SBEVAL(2,ISP)=CONE
200   CONTINUE

C   :Set K, the wavenumber
      K=KVAL(2)
C   :Set centre of the square
      CENTX=0.05D0
      CENTY=0.05D0
      DO 210 ISP=1,NSP
        X=SELCNT(ISP,1)
        Y=SELCNT(ISP,2)
        RR(1)=X-CENTX
        RR(2)=Y-CENTY
        R=SQRT(RR(1)**2+RR(2)**2)
        KR=K*R
        CALL FNHANK(KR,H)
C    Face X=0. Note the normal is in the direction of -x
        IF (X.LT.EPS) THEN
         DRBDN=-RR(1)/R
         SFVAL(2,ISP)=-CIMAG*K*DRBDN*H(1)/4.0D0
C    Face X=1. Note that the normal is in the direction of +x
        ELSE IF (X.GT.0.1D0-EPS) THEN
         DRBDN=RR(1)/R
         SFVAL(2,ISP)=-CIMAG*K*DRBDN*H(1)/4.0D0
C    Face Y=0. Note the normal is in the direction of -y
        ELSE IF (Y.LT.EPS) THEN
         DRBDN=-RR(2)/R
         SFVAL(2,ISP)=-CIMAG*K*DRBDN*H(1)/4.0D0
        ELSE 
C    Face Y=1. Note the normal is in the direction of +y
         DRBDN=RR(2)/R
         SFVAL(2,ISP)=-CIMAG*K*DRBDN*H(1)/4.0D0
        END IF
210   CONTINUE     

C   :There is no incident field
      DO 220 ISP=1,NSP
        SPHIIN(2,ISP)=0.0D0
        SVELIN(2,ISP)=0.0D0
220   CONTINUE
      DO 230 IPI=1,NPE
        PPHIIN(2,IPI)=0.0D0
230   CONTINUE


C  TEST PROBLEM 3
C  ==============
C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL(3)=344.0D0
      RHOVAL(3)=1.205D0

C  :Set acoustic frequency value (hertz) in FRVAL
      FRVAL(3)=400.0D0

C  : Set the wavenumber in KVAL
      KVAL(3)=TWO*PI*FRVAL(1)/CVAL(1)

C  :The test problem computes the field produced by a unit source at
C    the point (0.5,0.25) within the square with a rigid boundary.
C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALVAL and SBEVAL at the collocation points
C   :The rigid boundary implies the bondary condition v=0.
      DO 240 ISP=1,NSP
        SALVAL(3,ISP)=CZERO
        SBEVAL(3,ISP)=CONE
        SFVAL(3,ISP)=CZERO
240   CONTINUE

C   The incident velocity potential is given by {\phi}_inc=i*h0(kr)/4
C    where r is the distance from the point (0.5,0.25)
      PX=0.05D0
      PY=-0.05D0
C   :Set K, the wavenumber
      K=TWO*PI*FRVAL(3)/CVAL(3)
      DO 260 ISP=1,NSP
        X=SELCNT(ISP,1)
        Y=SELCNT(ISP,2)
        RR(1)=X-PX
        RR(2)=Y-PY
        R=SIZE2(RR)
        KR=K*R
        CALL FNHANK(KR,H)
        SPHIIN(3,ISP)=CIMAG*H(0)/4.0D0
C    Face X=0. Note the normal is in the direction of -x
        IF (X.LT.EPS) THEN
          SVELIN(3,ISP)=(-CIMAG*K*H(1)/4.0D0)*(-RR(1)/R)
C    Face X=1. Note that the normal is in the direction of +x
        ELSE IF (X.GT.0.1D0-EPS) THEN
          SVELIN(3,ISP)=(-CIMAG*K*H(1)/4.0D0)*(RR(1)/R)
C    Face Y=0. Note the normal is in the direction of -y
        ELSE IF (Y.LT.EPS) THEN
          SVELIN(3,ISP)=(-CIMAG*K*H(1)/4.0D0)*(-RR(2)/R)
        ELSE 
C    Face Y=1. Note the normal is in the direction of +y
          SVELIN(3,ISP)=(-CIMAG*K*H(1)/4.0D0)*(RR(2)/R)
        END IF
260   CONTINUE     
      DO 270 IPI=1,NPE
        X=PEXT(IPI,1)
        Y=PEXT(IPI,2)
        RR(1)=X-PX
        RR(2)=Y-PY
        R=SIZE2(RR)
        KR=K*R
        CALL FNHANK(KR,H)
        PPHIIN(3,IPI)=CIMAG*H(0)/4.0D0
270   CONTINUE



C Set up validation and control parameters
C  :Switch for particular solution
      LSOL=.TRUE.
C  :Switch on the validation of AEBEM2
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6

C Loop(ITEST) through the test problems
      DO 500 ITEST=1,NTEST
C  Set OMEGA, the angular frequency omega and K, the wavenumber
        K=KVAL(ITEST)
        OMEGA=2.0D0*PI*FRVAL(ITEST)

C   Set up particular alpha and beta functions for this wavenumber
C    and type of boundary condition
          DO 640 ISP=1,NSP
            SALPHA(ISP)=SALVAL(ITEST,ISP)
            SBETA(ISP)=SBEVAL(ITEST,ISP)
            SF(ISP)=SFVAL(ITEST,ISP)
            SFFPHI(ISP)=SPHIIN(ITEST,ISP)
            SFFVEL(ISP)=SVELIN(ITEST,ISP)
640       CONTINUE
          DO 650 IPE=1,NPE
            PEPHI(IPE)=PPHIIN(ITEST,IPE)
650       CONTINUE

C   Set the parameter
          MU=CIMAG/(K+ONE)

       
          CALL AEBEM2(K,
     *                 MAXNV,NV,VERTEX,MAXNS,NS,SELV,
     *                 MAXNPE,NPE,PEXT,
     *                 SALPHA,SBETA,SF,SFFPHI,SFFVEL,PFFPHI,
     *                 LSOL,LVALID,EGEOM,MU,
     *                 SPHI,SVEL,PEPHI,
     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,
     *                 WKSPC5,WKSPC6,WKSPC7)


C Compute the sound pressure at the exterior points. Also compute
C  the velocity and intensity at the points for each type of boundary
C  condition and each related input function f and at each point.
            DO 690 ISP=1,NSP
              SPHIVAL(ITEST,ISP)=SPHI(ISP)
              PRESSURE=CIMAG*RHOVAL(ITEST)*OMEGA*SPHI(ISP)
              SPRESS(ITEST,ISP)=PRESSURE
              SV(ITEST,ISP)=SVEL(ISP)
              SINTY(ITEST,ISP)=
     *         DBLE(CONJG(PRESSURE)*SVEL(ISP))/TWO
690         CONTINUE

            DO 695 ISP=1,NPE
              EPHIVAL(ITEST,ISP)=PEPHI(ISP)
              EPRESS(ITEST,ISP)=CIMAG*RHOVAL(ITEST)*OMEGA*PEPHI(ISP)
695         CONTINUE

C  Close loop(ITEST) through the test problems
500     CONTINUE


C Output the solutions
C  Open file for the output data
      OPEN(UNIT=20,FILE='AEBEM2.OUT')

C  Formats for output
2800  FORMAT(1X,'Acoustic frequency = ',F8.2,'  Hz'/)
2810  FORMAT(1X,'Wavenumber = ',F8.2/)
2830  FORMAT(4X,'Acoustic properties at the boundary points'/)
2845  FORMAT(4X,'Sound pressure at the exterior points',/)
2850  FORMAT(5X,'index',7X,'Potential',19X,'Pressure',24X,
     * 'Velocity',17X,'Intensity'/)
2855  FORMAT(5X,'index',8X,'Potential',20X,'Pressure',13X,
     *'Magnitude',13X,'Phase'/)
2860  FORMAT(4X,I4,2X,E10.4,'+ ',E10.4,' i    ',
     * E10.4, '+ ',E10.4,' i    ',4X,
     * E10.4, '+ ',E10.4,7X,F10.4)
2910  FORMAT(4X,I4,2X,E10.4,'+ ',E10.4,' i',4X,
     * E10.4, '+ ',E10.4,' i    ',E10.4,' dB',7X,F10.4)

      WRITE(20,*) 'AEBEM2: Computed solution to the acoustic properties'
      WRITE(20,*)
C  Loop(ITEST) through the test problems.
      DO 2000 ITEST=1,NTEST
C   Output the acoustic frequency
        WRITE(20,*)
        WRITE(20,*)
        WRITE(20,2800) FRVAL(ITEST)
        WRITE(20,2810) KVAL(ITEST)
        WRITE(20,*)
        WRITE(20,2830)
        WRITE(20,*)
        WRITE(20,2850)
        WRITE(20,*)
C      Loop(ISP) through the points on the boundary
        DO 2030 ISP=1,NSP
C       Output the sound pressure, velocity and intensity at each point
          WRITE(20,2860) ISP,DBLE(SPHIVAL(ITEST,ISP)),
     *     AIMAG(SPHIVAL(ITEST,ISP)),DBLE(SPRESS(ITEST,ISP)),
     *     AIMAG(SPRESS(ITEST,ISP)),DBLE(SV(ITEST,ISP)),
     *     AIMAG(SV(ITEST,ISP)),SINTY(ITEST,ISP)
2030    CONTINUE
        WRITE(20,*)
        WRITE(20,2845)
        WRITE(20,2855)
C      Loop(IPE) through the points in the exterior
        DO 2040 IPE=1,NPE
          PRESSURE=EPRESS(ITEST,IPE)
C       Output the sound pressure, its magnitude(dB) and phase
          WRITE(20,2910) IPE,DBLE(EPHIVAL(ITEST,IPE)),
     *     AIMAG(EPHIVAL(ITEST,IPE)),
     *     DBLE(PRESSURE),AIMAG(PRESSURE),
     *     LOG10(ABS(PRESSURE)/PREREF)*20.0D0,
     *     ATAN2(AIMAG(PRESSURE),DBLE(PRESSURE))
2040    CONTINUE
        WRITE(20,*)
2000  CONTINUE  

      CLOSE(20)

      END
