C*************************************************************
C    Test program for subroutine AEBEMA by Stephen Kirkup           
C*************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/AEBEMA_T.FOR
C
C  Issued under the GNU General Public License 2007, 
C   see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C************************************************************


C This program is a test for the subroutine AEBEMA. The program computes
C  the solution to an acoustic/Helmholtz problem exterior to a sphere 
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
C Subroutine AEBEMA accepts the wavenumber, a description of the 
C  boundary of the domain and the position of the exterior points
C  where the solution ({\phi}) is sought, the boundary condition and
C  returns the solution ({\phi} and v) on S and the value of {\phi}
C  at the exterior points.
C

C The test problems
C -----------------
C
C In this test the domain is a sphere of diameter 1 (metre). The acoustic
C  medium is air (at 20 celcius and 1 atmosphere, c=344.0 (metres per
C  second), density {\rho}=1.205 (kilograms per cubic metre) and the 
C  solution to the problem with a Dirichlet boundary condition 
C  ({\alpha}=1, {\beta}=0) and with a Neumann boundary condition 
C  ({\alpha}=0, beta=1) are sought. For both problems the frequency is
C  400Hz (hence specifying k).
C
C In the R-z plane, the boundary conditions are specified through 
C  taking the solution to be determined by
C
C              {\phi} = sin(k z) ,
C
C  which is clearly a solution of the Helmholtz equation.
C
C
C The boundary is described by a set of NS=32 elements of equal size,
C  so that each side comprises eight elements. The boundary solution
C  points are the centres of the elements. 
C The *s show the exterior points at which the solution is sought;
C  the points (0.025,0.025), (0.075,0.025), (0.025,0.075),
C  (0.075,0.075), and (0.05,0.05).

C----------------------------------------------------------------------

C The PARAMETER statement
C -----------------------
C There are four components in the PARAMETER statement.
C integer MAXNS  : The limit on the number of boundary elements.
C integer MAXNV   : The limit on the number of vertices.
C integer MAXNPE  : The limit on the number of exterior points.


C External modules related to the package
C ---------------------------------------
C subroutine AEBEMA: Subroutine for solving the exterior Helmholtz
C  equation. (file AEBEMA.FOR contains AEBEMA and subordinate routines)
C subroutine H2LC: Returns the individual discrete Helmholtz integral
C  operators. (file H2LC.FOR contains H2LC and subordinate routines)
C subroutine CGLS: Solves a general linear system of equations.
C  (file CGLS.FOR contains CGSL and subordinate routines)
C subroutine FNHANK: This computes Hankel functions of the first kind
C  and of order zero and one. (e.g. file FNHANK.FOR)
C file GEOM2D.FOR contains the set of relevant geometric subroutines 


C The program 

      PROGRAM AEBEMAT
      IMPLICIT NONE

C VARIABLE DECLARATION
C --------------------

C  PARAMETERs for storing the limits on the dimension of arrays
C   Limit on the number of elements
      INTEGER    MAXNS
      PARAMETER (MAXNS=32)
C   Limit on the number of vertices (equal to the number of elements)
      INTEGER    MAXNV
      PARAMETER (MAXNV=MAXNS+1)
C   Limit on the number of test problems
      INTEGER    MAXTEST
      PARAMETER (MAXTEST=200)
C   Limit on the number of points exterior to the boundary, where 
C    acoustic properties are sought
      INTEGER    MAXNPE
      PARAMETER (MAXNPE=1)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,FOUR,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  The reference pressure, used to convert units to decibels.
      REAL*8     PREREF


C  Properties of the acoustic medium
C   The speed of sound [standard unit: metres per second]
      REAL*8     CVAL(MAXTEST)
C   The density [standard unit: kilograms per cubic metre]
      REAL*8     RHOVAL(MAXTEST)

C   Wavenumber parameter for AEBEMA
      REAL*8     K
C   Angular frequency 
      COMPLEX*16 OMEGA

C  Geometrical description of the boundary(ies)
C   Number of elements and counter
      INTEGER    NS,IS
C   Number of collocation points (on S) and counter
      INTEGER    NSP,ISP
C   Number of vetices and counter
      INTEGER    NV,IV
C   Index of nodal coordinate for defining boundaries (standard unit is 
C    metres)
      REAL*8     VERTEX(MAXNV,2)
C   The two nodes that define each element on the boundaries
      INTEGER    SELV(MAXNS,2)
C   The points exterior to the boundary(ies) where the acoustic 
C    properties are sought and the directional vectors at those points.
C    [Only necessary if an exterior solution is sought.]
C    Number of exterior points and counter
      INTEGER    NPE,IPE
C    Coordinates of the exterior points
      REAL*8     PEXT (MAXNPE,2)

C  Number of test problems and counter
      INTEGER    NTEST,ITEST

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
C    and are input parameters to AEBEMA.
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

      
C  Validation and control parameters for AEBEMA
C   Switch for particular solution
      LOGICAL    LSOL
C   Validation switch
      LOGICAL    LVALID
C   The maximum absolute error in the parameters that describe the
C    geometry of the boundary.
      REAL*8     EGEOM
C   The parameter
      COMPLEX*16 MU

C Output from subroutine AEBEMA
C  The velocity potential (phi - the solution) at the centres of the 
C   elements
      COMPLEX*16 SPHI(MAXNS)
C  The normal derivative of the velocity potential at the centres of the
C    elements
      COMPLEX*16 SVEL(MAXNS)
C  The velocity potential (phi - the solution) at exterior points
      COMPLEX*16 PEPHI(MAXNPE)

C Element areas 
      REAL*8     ELAREA(MAXNS),AREACN
C Computing radiation ratio
      REAL*8     POWER,BPOWER,RADRAT

C Workspace for AEBEMA
      COMPLEX*16 WKSPC1(MAXNS,MAXNS)
      COMPLEX*16 WKSPC2(MAXNS,MAXNS)
      COMPLEX*16 WKSPC3(MAXNPE,MAXNS)
      COMPLEX*16 WKSPC4(MAXNPE,MAXNS)
      COMPLEX*16 WKSPC5(MAXNS)
      COMPLEX*16 WKSPC6(MAXNS)
      LOGICAL    WKSPC7(MAXNS)


C   Acoustic properties. These data structures are appended after each
C    execution of AEBEMA and contain the numerical solution to the test
C    problems. 
C    At the centres of the elements
C     Sound pressure [standard unit: newtons per sphere metre (or
C      pascals) and phase] 
      COMPLEX*16 SPRESS(MAXTEST,MAXNS)
C     Velocity potential phi
      COMPLEX*16 SPHIVAL(MAXTEST,MAXNS)
C  Velocity (v) [standard unit: metres per second (and phase)]
C   At the centres of the elements 
      COMPLEX*16 SV(MAXTEST,MAXNS)
C     Sound intensity [standard unit: watts per sphere metre]
      REAL*8     SINTY(MAXTEST,MAXNS)
C    At the exterior points
C     Sound pressure [standard unit: newtons per sphere metre (or
C      pascals) and phase] 
      COMPLEX*16 IPRESS(MAXTEST,MAXNPE)
C     Velocity potential phi
      COMPLEX*16 IPHIVAL(MAXTEST,MAXNPE)

C  Counter through the x,y coordinates
      INTEGER    ICOORD

C  Local storage of pressure, pressure/velocity 
      COMPLEX*16 PRESSURE

C  The coordinates of the centres of the elements  
      REAL*8     SELCNT(MAXNS,2)


      REAL*8     EPS

C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      FOUR=4.0D0
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

      EPS=1.0E-10

C  Reference for decibel scales
      PREREF=2.0D-05


C Describe the nodes and elements that make up the boundary
C  :The circle that generates the sphere is divided into NS=18 uniform
C  : elements. VERTEX and SELV are defined anti-clockwise around the
C  : boundary so that the normal to the boundary is assumed to be 
C  : outward
C  :Set up nodes
C  : Set NS, the number of elements
      NS=18
C  : Set NV, the number of vertices (equal to the number of elements)
      NV=NS+1
C  : Set coordinates of the nodes
      DATA ((VERTEX(IV,ICOORD),ICOORD=1,2),IV=1,19)
     * / 0.000D0, 1.000D0,
     *   0.174D0, 0.985D0,
     *   0.342D0, 0.940D0,
     *   0.500D0, 0.866D0,
     *   0.643D0, 0.766D0,
     *   0.766D0, 0.643D0,
     *   0.866D0, 0.500D0,
     *   0.940D0, 0.342D0,
     *   0.985D0, 0.174D0,
     *   1.000D0, 0.000D0,
     *   0.985D0,-0.174D0,
     *   0.940D0,-0.342D0,
     *   0.866D0,-0.500D0,
     *   0.766D0,-0.643D0,
     *   0.643D0,-0.766D0,
     *   0.500D0,-0.866D0,
     *   0.342D0,-0.940D0,
     *   0.174D0,-0.985D0,
     *   0.000D0,-1.000D0 /

C  :Describe the elements that make up the two boundarys
C  : Set NS, the number of elements
      NS=18
C  : Set nodal indices that describe the elements of the boundarys.
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in SELV dictates that the normal is outward from the 
C  :  boundary into the acoustic domain.
      DATA ((SELV(IS,ICOORD),ICOORD=1,2),IS=1,18)
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
     *   18,  19 /
       
C Set the centres of the elements, the collocation points
      DO IS=1,NS
        SELCNT(IS,1)=(VERTEX(SELV(IS,1),1)
     *   +VERTEX(SELV(IS,2),1))/TWO
        SELCNT(IS,2)=(VERTEX(SELV(IS,1),2)
     *   +VERTEX(SELV(IS,2),2))/TWO
        ELAREA(IS)=AREACN(VERTEX(SELV(IS,1),1),VERTEX(SELV(IS,1),2),
     *   VERTEX(SELV(IS,2),1),VERTEX(SELV(IS,2),2))
      END DO


C Set the points in the acoustic domain where the acoustic properties
C  are sought, PEXT . 
      NPE=0


C The number of points on the boundary is equal to the number of 
C  elements
      NSP=NS
        
C Set up test problems
C  :Set the number of test problems
      NTEST=10


      DO 400 ITEST=1,NTEST
C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL(ITEST)=344.0D0
      RHOVAL(ITEST)=1.205D0

C  :Set acoustic frequency value (hertz) in FRVAL
      FRVAL(ITEST)=ITEST*100.0D0

C  : Set the wavenumber in KVAL
      KVAL(ITEST)=TWO*PI*FRVAL(ITEST)/CVAL(ITEST)

C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALVAL and SBEVAL at the collocation points
C   :In this case a Dirichlet (phi-valued) boundary condition
      K=KVAL(ITEST)
      DO 160 ISP=1,NSP
        SALVAL(ITEST,ISP)=CZERO
        SBEVAL(ITEST,ISP)=CONE
        SFVAL(ITEST,ISP)=1.0D0
160   CONTINUE


      DO 180 ISP=1,NSP
        SPHIIN(ITEST,ISP)=CZERO
        SVELIN(ITEST,ISP)=CZERO
180   CONTINUE
      DO 190 IPE=1,NPE
        PPHIIN(ITEST,IPE)=CZERO
190   CONTINUE


C Set up validation and control parameters
C  :Switch for particular solution
      LSOL=.TRUE.
C  :Switch on the validation of AEBEMA
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6

400   CONTINUE

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
            PFFPHI(IPE)=PPHIIN(ITEST,IPE)
650       CONTINUE

C   Set the parameter
          MU=CIMAG/(K+ONE)
       
          CALL AEBEMA(K,
     *                 MAXNV,NV,VERTEX,MAXNS,NS,SELV,
     *                 MAXNPE,NPE,PEXT ,
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
              IPHIVAL(ITEST,ISP)=PEPHI(ISP)
              IPRESS(ITEST,ISP)=CIMAG*RHOVAL(ITEST)*OMEGA*PEPHI(ISP)
695         CONTINUE

C  Close loop(ITEST) through the test problems
500     CONTINUE


C Output the solutions
C  Open file for the output data
      OPEN(UNIT=20,FILE='AEBEMA.OUT')

C  Formats for output
999   FORMAT(3F16.4)

      WRITE(20,*) 'AEBEMA: Computed solution to the acoustic properties'
      WRITE(20,*)
      WRITE(20,*) 'The radiation ratio of a pulsating sphere'
      WRITE(20,*)
      WRITE(20,*)
      WRITE(20,*) '        Frequency        Wavenumber      '//
     *'Radiation ratio'
C  Loop(ITEST) through the test problems.
      DO 2000 ITEST=1,NTEST
C   Output the acoustic frequency
C      Loop(ISP) through the points on the boundary
        POWER=0.0D0
        BPOWER=0.0D0
        DO 2030 ISP=1,NSP
C       Output the sound pressure, velocity and intensity at each point
          POWER=POWER+SINTY(ITEST,ISP)*ELAREA(ISP)
          BPOWER=BPOWER+RHOVAL(ITEST)*CVAL(ITEST)*
     *      ABS(SV(ITEST,ISP))**2*ELAREA(ISP)/2
2030    CONTINUE
        RADRAT=POWER/BPOWER
        WRITE(20,999) FRVAL(ITEST),KVAL(ITEST),RADRAT
2000  CONTINUE  

      CLOSE(20)

      END


       REAL*8 FUNCTION AREACN(R1,Z1,R2,Z2)
       REAL*8 R1,Z1,R2,Z2
       REAL*8 A,B
       REAL*8 PI
       PI=4.0D0*ATAN(1.0D0)
       IF (ABS(Z2-Z1).GT.0.000001) THEN
         A=R1-(R2-R1)*Z1/(Z2-Z1)
         B=(R2-R1)/(Z2-Z1)
         AREACN=
     *    ABS(2.0D0*PI*SQRT(1.0D0+B*B)*(Z2-Z1)*(A+B*(Z1+Z2)/2.0D0))
       ELSE
         AREA=PI*ABS(R1*R1-R2*R2)
       END IF
       END
