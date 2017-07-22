C***************************************************************
C          Subroutine H3ALC by Stephen Kirkup                        
C***************************************************************
C
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/H3ALC.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C

C
C This subroutine computes the discrete form of the 3-dimensional
C axisymmetric Helmholtz integral operators Lk,Mk,Mkt, and Nk over a
C conical element. The subroutine is useful in integral equation methods
C for the solution of 3-dimensional axisymmetric Helmholtz problems.

C The following diagram shows how the subroutine is to be used. A main
C program is required along with the definition of FNEXP and FNSQRT.
C The package is the contents of this file.
C
C                                   ....................................
C                                   :                                  :
C                                   :         -------    --------      :
C Routines to be supplied ------>   :         |FNEXP|    |FNSQRT|      :
C                                   :         -------    --------      :
C                                   :...........|..........|...........:
C                                               |          |
C                                               <          <
C                                   ............|..........|............
C                                   :           |          |           :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|          H3ALC         |   :
C      |                    |       :     |                        |   :
C      ----------------------       :     --------------------------   :
C                                   :                 |                :
C                                   :                 >                :
C                                   :                 |                :
C                                   :      ------------------------    :
C          Package ---------------->:      | Subordinate routines |    :
C                                   :      ------------------------    :
C                                   :                                  :
C                                   :..................................:
C
C
C The subroutine has the form:

C      SUBROUTINE H3ALC(K,P,VECP,QA,QB,LPONEL,
C     * MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
C     * LVALID,EK,EGEOM,EQRULE,LFAIL,
C     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
C     * WKSPCE)

C The parameters to the subroutine
C ================================
C Wavenumber (input)
C complex K: The complex wavenumber.

C Point (input)
C real P(2): The point p in R,z coordinates. P(1)>=0.
C real VECP(2): A unit normal in R-z components, required in the 
C  computation of DISMKT and DISNK. The squares of the components must 
C  sum to one. 

C Geometry of element (input)
C real QA(2): The R-z coordinates of the first edge of the element.
C  QA(1)>=0.
C real QB(2): The R-z coordinates of the second edge of the element.
C  QB(1)>=0.
C logical LPONEL: If the point P lies on QA-QB then LPONEL must be set
C  .TRUE., otherwise LPONEL must be set .FALSE..

C Generator-direction quadrature rule (input)
C integer MAXNGQ: The limit on the size of the generator-direction
C  quadrature rule. The value should not be changed between calls of
C  H3ALC. MAXNGQ>=1.
C integer NGQ: The actual number of generator-direction quadrature 
C  rule points. 1<=NGQ<=MAXNGQ, NGQ<=LIMNGQ.
C real AGQ(MAXNGQ): The generator-direction quadrature rule abscissae.
C  The values must lie in the domain [0,1] and be in ascending order.
C real WGQ(MAXNGQ): The generator-direction quadrature rule weights 
C  which correspond to the quadrature points in AGQ. The components of
C  WGQ must sum to one.

C Theta-direction quadrature rule (input)
C integer MAXNTQ: The limit on the size of the theta-direction
C  quadrature rule. The value should not be changed between calls of
C  H3ALC. MAXNTQ>=1.
C integer NTQ: The actual number of theta-direction quadrature rule
C  points. 1<=NTQ<=MAXNTQ.
C real ATQ(MAXNTQ): The theta-direction quadrature rule abscissae. The
C  values must lie in the domain [0,1].
C real WTQ(MAXNTQ): The theta-direction quadrature rule weights which
C  correspond to the quadrature points in ATQ. The components of WTQ
C  must sum to one.

C Validation and control parameters (input)
C logical LVALID: A switch to enable choice of checking of subroutine
C  parameters (.TRUE.) or not (.FALSE.).
C real EK: The maximum absolute error expected in K. This is used
C  to classify K as 'zero', 'real', 'imaginary' or 'complex'.
C real EGEOM: The maximum absolute error in the parameters that
C  describe the geometry. Value is of importance only when LVALID=.TRUE..
C real EQRULE: The maximum absolute error in the components of the
C  quadrature rule data. Value is of importance only when LVALID=.TRUE..

C Validation parameter (output)
C logical LFAIL: Value is only important if LVALID=.TRUE.. If 
C  LFAIL=.FALSE. then the input data has been found to be satisfactory. 
C  If LFAIL=.TRUE. then the  input data has been found to be 
C  unsatisfactory. The subroutine would have been aborted. The output
C  parameters DISLK, DISMK, DISMKT and DISNK will all be zero.  A 
C  diagnosis of the errors will be given in the file H3ALC.ERR.

C Choice of discrete forms required (input)
C logical LLK: If discrete form of Lk operator is required then set
C  .TRUE., otherwise set .FALSE..
C logical LMK: If discrete form of Mk operator is required then set
C  .TRUE., otherwise set .FALSE..
C logical LMKT: If discrete form of Mkt operator is required then set
C  .TRUE., otherwise set .FALSE..
C logical LNK: If discrete form of Nk operator is required then set
C  .TRUE., otherwise set .FALSE..

C Discrete Helmholtz integral operators (output)
C complex DISLK: The discrete Lk integral operator.
C complex DISMK: The discrete Mk integral operator.
C complex DISMKT: The discrete Mkt integral operator.
C complex DISNK: The discrete Nk integral operator.

C Work space
C real WKSPCE(2*MAXNTQ+MAXNGQ)

C External modules to be supplied
C ================================
C complex FUNCTION FNEXP(Z): complex Z : Must return the exponential
C  of Z.
C real FUNCTION FNSQRT(X): real X : Must return the square root of X.

C Notes on the validation parameters
C ----------------------------------
C  (1) If LVALID=.TRUE. then a file named H3ALC.ERR should be open 
C   when subroutine H3ALC is entered. Use a command such as 
C         OPEN(UNIT=10,FILE='H3ALC.ERR',STATUS='UNKNOWN')
C  at the beginning of the calling program and the corresponding command
C         CLOSE(10)
C  at the end of the calling program.
C  This file accumulates the error messages and warnings from the 
C   subroutine. If this file is not open then an error message is 
C   output to standard output and LFAIL=.TRUE.
C (2) If LVALID=.TRUE. then EGEOM must be less than 1% of the maximum
C  absolute coordinate of QA and QB.
C (3) If LVALID=.TRUE. then EQRULE must be less than 10%/max(NGQ,NTQ).

C Notes on the geometric parameters
C ---------------------------------
C  (1) P, QA and QB must be distinct points (with respect to EGEOM).
C  (2) If LPONEL=.TRUE. the P must lie on the generator of the element
C   QA-QB. If LPONEL=.FALSE. then P must not lie on QA-QB.
C  (3) The normal on the generator to the element is defined to be
C             [-QB(2)+QA(2), QB(1)-QA(1)] normalised.
C   Hence the normal is to the right on the line QA-QB.
C  (4) If LPONEL=.TRUE. (and either LMKT=.TRUE. or LNK=.TRUE.) then 
C   VECP must also be [-QB(2)+QA(2), QB(1)-QA(1)] normalised.
C  (5) QA and QB must not both lie on the axis of symmetry.

C Notes on the quadrature rule parameters
C ---------------------------------------
C  (1) The quadrature rules are assumed to compute linear functions
C   exactly (with respect to EQRULE).
C  (2) If LPONEL=.TRUE. then the first derivative of the Lk and Nk
C   functions on the element are discontinuous at P on the generator.
C   The input quadrature rule AGQ should take account of this by using a
C   composite rule that divides at P in these cases.

C External modules provided
C ==========================
C complex FUNCTION ICMULT(Z): Returns the result of multiplying 
C  complex Z by i.
C complex FUNCTION IRMULT(X): Returns the result of multiplying 
C  real X by i.
C Subroutine SUBV2(VECA,VECB,VEC) gives the result of subtracting
C  the 2-vector VECB from VECA and stores the result in VEC.

C Subroutine
      SUBROUTINE H3ALC(K,P,VECP,QA,QB,LPONEL,
     * MAXNGQ,NGQ,AGQ,WGQ,MAXNTQ,NTQ,ATQ,WTQ,
     * LVALID,EK,EGEOM,EQRULE,LFAIL,
     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK,
     * WKSPCE)
     
C Wavenumber
      COMPLEX*16 K

C Point
      REAL*8     P(2)
      REAL*8     VECP(2)

C Geometry of element
      REAL*8     QA(2)
      REAL*8     QB(2)
      REAL*8     NORMQ(2)
      LOGICAL    LPONEL

C Generator-direction quadrature rule
      INTEGER    MAXNGQ
      INTEGER    NGQ
      REAL*8     AGQ(MAXNGQ)
      REAL*8     WGQ(MAXNGQ)

C Theta-direction quadrature rule
      INTEGER    MAXNTQ
      INTEGER    NTQ
      REAL*8     ATQ(MAXNTQ)
      REAL*8     WTQ(MAXNTQ)

C Validation and control parameters
      LOGICAL    LVALID
      REAL*8     EK
      REAL*8     EGEOM
      REAL*8     EQRULE
      LOGICAL    LFAIL

C Choice of discrete forms required
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C Discrete Helmholtz integral operators
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

C Work space
      REAL*8     WKSPCE(2*MAXNTQ+MAXNGQ)

C External functions to be supplied
      REAL*8     FNSQRT
      COMPLEX*16 FNEXP

C External functions provided
      COMPLEX*16 ICMULT
      COMPLEX*16 IRMULT
      REAL*8     SIZE2
      REAL*8     SSIZE2
      REAL*8     DOT2
      REAL*8     SSIZE3
      REAL*8     DOT3

C Variables that remain constant throughout subroutine, once initialised
C  Constants
      REAL*8     ZERO,ONE,TWO,FOUR,HALF
      REAL*8     PI,ROOT2
      COMPLEX*16 CZERO,CONE,CIMAG
C  H3ALC open status and unit number
      LOGICAL    LOPEN
      INTEGER    IU
C  Useful geometric values
      REAL*8     QBMA(2),PMA(2),PMB(2)
      REAL*8     PM(3),VERTU(2),VERTL(2),VECPM(3)
      REAL*8     GQLEN,GPALEN
C  Magnitude (one norm) of geometric values
      REAL*8     MAGVP
      REAL*8     MAGBMA,MAGPMA,MAGPMB
C  Control values relating to required solutions     
      LOGICAL    LLORN,LMORN,LMTORN,LMORMT,LMSORN
C  Values that classify K as 'zero','real','imaginary', or 'complex'
      LOGICAL    KZERO,KREAL,KIMAG,KCOMP
C  Useful constants related to K
      REAL*8     REK,IMK
      COMPLEX*16 SK,SKO2

C Variables used in the validation of the input data
C  Error variables
      LOGICAL    LERROR,LOCERR
C  Temporary variables
      REAL*8     TEMP,TEMP1,TEMP2,TEMP3,CPX,SQLEN,TOP,VAL
C  Corresponding error bounds to the temporary variables
      REAL*8     ETEMP,ETEMP1,ETEMP2,ETEMP3,EPX,ESQLEN,ETOP

C Variables used when P lies on the element QA-QB-QC
C  Geometric values
      REAL*8     RCORU,RCORL,ZCORU,ZCORL,RCORM,ZCORM
      REAL*8     RADU,ZU,RADL,ZL,PX,TEMPPX
      REAL*8     QU(3),QL(3),NORMQU(3),NORMQL(3),RRU(3),RRL(3)
      REAL*8     DNPNQU,DNPNQL,SRU,RU,CRU,SRL,RL,CRL
      REAL*8     RUNP,RUNQ,RLNP,RLNQ,RUN,RLN
      REAL*8     CONLNU,CONLNL
C  Variables for the quadrature rule over the cone
      INTEGER    NODIVU,NODIVL
      REAL*8     AGQX,FLNDU,FLNDL,FLID
      LOGICAL    LSWAP
C   Variables for the accumulation of integrals
      REAL*8     TGSUMU,TGSUML,GSUMU,GSUML
C   Number,index of divisions of the cone for the composite integral
      INTEGER*4  IDIV   
C Resultant values of L0,N0
      REAL*8     L0VAL,N0VAL

C Variables used in the operation of the quadrature rule 
C  Variables for the accumulation of integrals
      COMPLEX*16 SUML,SUMM,SUMMT,SUMN
      COMPLEX*16 TSUML,TSUMM,TSUMMT,TSUMN
C  Indices of quadrature point
      INTEGER*4  IGQ,ITQ,JGQ
C  Geometric variables
      REAL*8     Q(3),NORMQQ(3),DNPNQ,RR(3),SR,R,CR,RAD
      REAL*8     RNP,RNQ,RNPRNQ,RNPNQ
C  Green's functions for the Laplace operator and its r-derivatives
      REAL*8     FPG0,FPG0R,FPG0RR
C  Geometric variables related to K
      COMPLEX*16 KR,SKR,IKR
C  Green's functions for the Helmholtz operator and its r-derivatives
      COMPLEX*16 FPG,FPGR,FPGRR
      COMPLEX*16 WFPGR
C  Variable for the storage of the value of the exponential functions
      COMPLEX*16 E
C Local variables.
      COMPLEX*16 EOSR,GWRO2

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      FOUR=4.0D0
      HALF=0.5D0
      PI=3.141592653589732
      ROOT2=SQRT(TWO)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

C Initialise useful geometrical information.
      CALL SUBV2(QB,QA,QBMA)
      CALL SUBV2(P,QA,PMA)
      CALL SUBV2(P,QB,PMB)
      GQLEN=SIZE2(QBMA)
      GPALEN=SIZE2(PMA)

C Set LFAIL to .FALSE.
      LFAIL=.FALSE.
      LERROR=.FALSE.

C If LVALID is false then bypass the validation process
      IF (.NOT.LVALID) GOTO 900  


C VALIDATION OF INPUT

C Check that the file H3ALC.ERR has been opened, if not then exit
      INQUIRE(FILE='H3ALC.ERR',OPENED=LOPEN)
      IF (.NOT.LOPEN) THEN
        WRITE(*,*) 'ERROR(H3ALC) - File for error messages "H3ALC.ERR"'
        WRITE(*,*) ' is not open'
        WRITE(*,*) 'Execution of H3ALC is aborted'
        LERROR=.TRUE.
        GOTO 998
      END IF

C Find out the unit number of the file H3ALC.ERR
      INQUIRE(FILE='H3ALC.ERR',NUMBER=IU)

C Set useful constants
      IF (LMKT.OR.LNK) MAGVP=ABS(VECP(1))+ABS(VECP(2))
      MAGBMA=ABS(QBMA(1))+ABS(QBMA(2))
      MAGPMA=ABS(PMA(1))+ABS(PMA(2))
      MAGPMB=ABS(PMB(1))+ABS(PMB(2))


C Check that the function FNSQRT is satisfactory
      IF (LVALID) THEN
        LERROR=.FALSE.
        IF (ABS(FNSQRT(1.0D0)-1.0D0).GT.0.01) LERROR=.TRUE.
        IF (ABS(FNSQRT(100.0D0)-10.0D0).GT.0.1) LERROR=.TRUE.
        IF (ABS(FNSQRT(0.01D0)-0.1).GT.0.001) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(IU,*) 'ERROR(H2LC) - in square root function routine'
     *     //' FNSQRT'
          STOP
        END IF
      END IF

        
C Check that the function FNEXP is satisfactory
      IF (LVALID) THEN
        LERROR=.FALSE.
        IF (ABS(FNEXP(CONE)-EXP(CONE)).GT.0.1) LERROR=.TRUE.
        IF (ABS(FNEXP(CZERO)-EXP(CZERO)).GT.0.1) LERROR=.TRUE.
        IF (ABS(FNEXP(CIMAG)-EXP(CIMAG)).GT.0.1) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(IU,*) 'ERROR(H2LC) - in exponential function routine'
     *     //' FNSQRT'
          STOP
        END IF
      END IF

        


C Check the tolerances EK,EGEOM,EQRULE are positive
      IF (EK.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - PARAMETER EK must be positive'
        LERROR=.TRUE.
      END IF
      IF (EGEOM.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - PARAMETER EGEOM must be positive'
        LERROR=.TRUE.
      END IF
      IF (EQRULE.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - PARAMETER EQRULE must be positive'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF


C Validation of the geometric information
C  Check that P(1)>=0.
      IF (P(1).LT.-EGEOM) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter P(1) must be'
        WRITE(IU,*) ' non-negative'
        LERROR=.TRUE.
      END IF
C  Check VECP is of unit magnitude
      IF (LMKT.OR.LNK) THEN
        TEMP=SSIZE2(VECP)
        ETEMP=TWO*EGEOM*MAGVP
        IF (ABS(TEMP-ONE).GT.ETEMP) THEN
          WRITE(IU,*) 'ERROR(H3ALC) - Parameter VECP must have unit'
          WRITE(IU,*) ' magnitude'
          LERROR=.TRUE.
        END IF
      END IF
C  Check that QA(1)>=0.
      IF (QA(1).LT.-EGEOM) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter QA(1) must be'
        WRITE(IU,*) ' non-negative'
        LERROR=.TRUE.
      END IF
C  Check that QB(1)>=0.
      IF (QB(1).LT.-EGEOM) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter QB(1) must be'
        WRITE(IU,*) ' non-negative'
        LERROR=.TRUE.
      END IF
C  Check QA and QB are distinct points
      TEMP=MAGBMA
      ETEMP=FOUR*EGEOM
      IF (TEMP.LT.ETEMP) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - QA must not coincide with QB'
        LERROR=.TRUE.
      END IF
C  Check QA and QB do not both lie on the axis of symmetry
      IF (ABS(QA(1)).LT.EGEOM.AND.ABS(QB(1)).LT.EGEOM) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - QA and QB must not both lie'
        WRITE(IU,*) ' on the axis of symmetry'
        LERROR=.TRUE.
      END  IF  
C  Check P is distinct from QA and QB 
      TEMP1=MAGPMA
      TEMP2=MAGPMB
      ETEMP=FOUR*EGEOM
      IF (TEMP1.LT.ETEMP.OR.TEMP2.LT.ETEMP) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - P must not coincide with QA or QB'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF


C Check EGEOM is not set too large
      IF (EGEOM.GT.MAX(ABS(QA(1)),ABS(QA(2)),ABS(QB(1)),
     * ABS(QB(2)))/100.0D0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter EGEOM is too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C Continue validation of the geometric information
C  Check LPONEL
      SQLEN=GQLEN*GQLEN
      ESQLEN=TWO*EGEOM*MAGBMA
      TOP=DOT2(QBMA,PMA)
      ETOP=EGEOM*(MAGPMA+MAGBMA)
      TEMP=TOP/SQLEN
      ETEMP=(ETOP+TEMP*ESQLEN)/SQLEN
      IF (TEMP.LT.-ETEMP) TEMP=-ETEMP
      IF (TEMP.GT.ONE+ETEMP) TEMP=ONE+ETEMP
      TEMP1=ABS(QA(1)+TEMP*QBMA(1)-P(1))
      TEMP2=ABS(QA(2)+TEMP*QBMA(2)-P(2))
      ETEMP1=TWO*EGEOM*(ONE+TEMP)+ETEMP*ABS(QBMA(1))
      ETEMP2=TWO*EGEOM*(ONE+TEMP)+ETEMP*ABS(QBMA(2))
      IF (LPONEL.AND.(TEMP1.GT.ETEMP1.OR.TEMP2.GT.ETEMP2)) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - LPONEL should be .FALSE.'
        LERROR=.TRUE.
      END IF
      IF (.NOT.LPONEL.AND.TEMP1.LT.ETEMP1.AND.TEMP2.LT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - LPONEL should be .TRUE.'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C  Check that VECP is THE normal to QA-QB when P is on QA-QB
C   If not then exit
      IF (LPONEL.AND.(LMKT.OR.LNK)) THEN
        IF (ABS(DOT2(VECP,QBMA)).GT.EGEOM*(TWO*MAGVP+MAGBMA)) THEN
          WRITE(IU,*) 'ERROR(HO2LC) - VECP must be normal to QA-QB'
          LERROR=.TRUE.
        END IF
        IF (.NOT.LERROR) THEN
          IF (-VECP(1)*QBMA(2)+VECP(2)*QBMA(1).LT.ZERO) THEN
            WRITE(IU,*) 'ERROR(HO2LC) - Replace VECP by -VECP'
            LERROR=.TRUE.
          END IF
        END IF
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF


C Validation of the G-quadrature rule data
C  Check G-quadrature rule data
      IF (MAXNGQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have MAXNGQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NGQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have NGQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NGQ.GT.MAXNGQ) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have NGQ <= MAXNGQ'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C Check EQRULE is not set too large
      IF (EQRULE.GT.ONE/FLOAT(NGQ)/10.0D0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter EQRULE is too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C Continue validation of the G-quadrature rule data
C  Check that each AGQ(i) lies in [0,1]
      LOCERR=.FALSE.
      DO 100 JGQ=1,NGQ
        IF (AGQ(JGQ).LT.-EQRULE.OR.AGQ(JGQ).GT.ONE+EQRULE) LOCERR=.TRUE.
100   CONTINUE
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Components of AGQ are outside of'
        WRITE(IU,*) '[0,1]'
        LERROR=.TRUE.
      END IF
C  Check that the AGQ(i) are in ascending order
      LOCERR=.FALSE.
      DO 110 JGQ=2,NGQ
        IF (AGQ(JGQ).LE.AGQ(JGQ-1)) LOCERR=.TRUE.
110   CONTINUE
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Components of AGQ must be in'
        WRITE(IU,*) 'strictly ascending order.'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF
 
C Continue validation of the G-quadrature rule data
C  Check that the integral of 1 and x (and |x-p| when LPONEL) are
C   computed to within the accuracy dictated by EQRULE. That is there
C   is no numerical integration error.
      TEMP1=ZERO
      ETEMP1=ZERO
      TEMP2=ZERO
      ETEMP2=ZERO
      TEMP3=ZERO
      ETEMP3=ZERO
      IF (LPONEL) THEN
        IF (ABS(QBMA(1)).GT.ABS(QBMA(2))) THEN
          PX=ABS(PMA(1)/QBMA(1))
          EPX=2.0D0*EGEOM*(ONE+PX)/ABS(QBMA(1))
        ELSE
          PX=ABS(PMA(2)/QBMA(2))
          EPX=TWO*EGEOM*(ONE+PX)/ABS(QBMA(2))
        END IF
        CPX=ONE-PX
      END IF
      DO 120 JGQ=1,NGQ
        TEMP1=TEMP1+WGQ(JGQ)
        ETEMP1=ETEMP1+EQRULE
        TEMP2=TEMP2+WGQ(JGQ)*AGQ(JGQ)
        ETEMP2=ETEMP2+EQRULE*(ABS(WGQ(JGQ))+ABS(AGQ(JGQ)))
        IF (LPONEL) THEN
          TEMP3=TEMP3+WGQ(JGQ)*ABS(AGQ(JGQ)-PX)
          ETEMP3=ETEMP3+EQRULE*ABS(AGQ(JGQ)-PX)+
     *     ABS(WGQ(JGQ))*(EQRULE+EPX)
        END IF
120   CONTINUE
      IF (ABS(TEMP1-ONE).GT.ETEMP1.OR.
     * ABS(TEMP2-HALF).GT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - In AGQ-WGQ quadrature rule'
        LERROR=.TRUE.
      END IF
      IF (LPONEL) THEN
        VAL=(PX*PX+CPX*CPX)/TWO
        IF (ABS(TEMP3-VAL).GT.ETEMP3) THEN
          WRITE(IU,*) 'WARNING - AGQ-WGQ rule should not assume'
          WRITE(IU,*) ' continuity when LPONEL=.TRUE.'
        LERROR=.TRUE.
        END IF
      END IF
      LOCERR=.FALSE.
      IF (LPONEL) THEN
        DO 125 JGQ=1,NGQ
          IF (ABS(AGQ(JGQ)-PX).LT.EQRULE+EPX) LOCERR=.TRUE.
125     CONTINUE    
      END IF
      IF (LOCERR) THEN
        LERROR=.TRUE.
        WRITE(IU,*) 'ERROR(HO2LC) - One of the quadrature points'
        WRITE(IU,*) ' coincides with the point P'
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C Validation of the T-quadrature rule data
      IF (MAXNTQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have MAXNTQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NTQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have NTQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NTQ.GT.MAXNTQ) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Must have NTQ <= MAXNTQ'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF

C Check EQRULE is not set too large
      IF (EQRULE.GT.ONE/FLOAT(NTQ)/10.0D0) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Parameter EQRULE is too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF


C Continue validation of the T-quadrature rule data
      LOCERR=.FALSE.
      DO 130 ITQ=1,NTQ
        IF (ATQ(ITQ).LT.-EQRULE.OR.ATQ(ITQ).GT.ONE+EQRULE) LOCERR=.TRUE.
130   CONTINUE
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - Components of ATQ are outside'
        WRITE(IU,*) '[0,1]'
        LERROR=.TRUE.
      END IF

C  Check that the integral of 1 and x are computed to within the
C   accuracy dictated by EQRULE. That is there is no numerical 
C   integration error.
      TEMP1=ZERO
      ETEMP1=ZERO
      TEMP2=ZERO
      ETEMP2=ZERO
      DO 135 ITQ=1,NTQ
        TEMP1=TEMP1+WTQ(ITQ)
        ETEMP1=ETEMP1+EQRULE
        TEMP2=TEMP2+WTQ(ITQ)*ATQ(ITQ)
        ETEMP2=ETEMP2+(ABS(WTQ(ITQ))+ABS(ATQ(ITQ)))*EQRULE
135   CONTINUE
      IF (ABS(TEMP1-ONE).GT.ETEMP1.OR.ABS(TEMP2-HALF).GT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H3ALC) - In ATQ-WTQ quadrature rule'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3ALC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3ALC is aborted'
        GOTO 998
      END IF


900   CONTINUE

C FURTHER INITIALISATION

C Set the values of the discrete operators to zero
      DISLK=CZERO
      DISMK=CZERO
      DISMKT=CZERO
      DISNK=CZERO

C Exit if no solutions are required
      IF (.NOT.(LLK.OR.LMK.OR.LMKT.OR.LNK)) GOTO 998

C Initialise useful geometric information
      PM(1)=P(1)
      PM(2)=ZERO
      PM(3)=P(2)


      IF (LPONEL) THEN
        IF (ABS(QBMA(1)).GT.ABS(QBMA(2))) THEN
          PX=ABS(PMA(1)/QBMA(1))
        ELSE
          PX=ABS(PMA(2)/QBMA(2))
        END IF
      END IF


      IF (QA(2).GT.QB(2)) THEN
        VERTU(1)=QA(1)
        VERTU(2)=QA(2)
        VERTL(1)=QB(1)
        VERTL(2)=QB(2)
        IF (LPONEL) TEMPPX=PX
        LSWAP=.FALSE.
      ELSE
        VERTL(1)=QA(1)
        VERTL(2)=QA(2)
        VERTU(1)=QB(1)
        VERTU(2)=QB(2)
        IF (LPONEL) TEMPPX=ONE-PX
        LSWAP=.TRUE.
      END IF

      IF (LMKT.OR.LNK) THEN
        VECPM(1)=VECP(1)
        VECPM(2)=ZERO
        VECPM(3)=VECP(2)
      END IF

      IF (LPONEL) PX=GPALEN/GQLEN

C Initialise the values stored in WKSPCE
      DO 140 IGQ=1,NGQ
        WKSPCE(2*NTQ+IGQ)=AGQ(IGQ)*AGQ(IGQ)
140   CONTINUE

      DO 150 ITQ=1,NTQ
        WKSPCE(ITQ)=COS(PI*ATQ(ITQ))
        WKSPCE(NTQ+ITQ)=SIN(PI*ATQ(ITQ))
150   CONTINUE

C Initialise useful control information
      LLORN=LLK.OR.LNK
      LMORN=LMK.OR.LNK
      LMSORN=LMK.OR.LMKT.OR.LNK
      LMTORN=LMKT.OR.LNK
      LMORMT=LMK.OR.LMKT

C Initialise useful geometric information
      IF (LMORN) THEN
        NORMQ(1)=-QBMA(2)/GQLEN
        NORMQ(2)=QBMA(1)/GQLEN
      END IF

C Useful constants related to K
      IMK=DIMAG(K)
      REK=DBLE(K)
      SK=K*K
      SKO2=SK/TWO

C Classification of K
      KZERO=.FALSE.
      KREAL=.FALSE.
      KIMAG=.FALSE.
      KCOMP=.FALSE.
      IF (ABS(REK).LT.EK.AND.ABS(IMK).LT.EK) THEN
        KZERO=.TRUE.
      ELSE IF (ABS(REK).LT.EK) THEN
        KIMAG=.TRUE.
      ELSE IF (ABS(IMK).LT.EK) THEN
        KREAL=.TRUE.
      ELSE
        KCOMP=.TRUE.
      END IF

C Initialise SUML-SUMN, the variables that accumulate the integrals
C  Lk,Mk,Mkt,Nk.
      SUML=CZERO
      SUMM=CZERO
      SUMMT=CZERO
      SUMN=CZERO


C  Computation of L0 value if the integral is singular. The singularity
C  is treated by dividing the element into two parts at P.
C  The generator quadrature rule is transformed by substituting
C  y squared for x where x is the variable describing the position on the
C  generator.

      IF (LPONEL.AND.(LLK.OR.LNK)) THEN
        RCORU=VERTU(1)
        RCORL=VERTL(1)
        ZCORU=VERTU(2)
        ZCORL=VERTL(2)
        RCORM=P(1)
        ZCORM=P(2)
        PM(1)=RCORM
        PM(2)=ZERO
        PM(3)=ZCORM
        VECPM(1)=VECP(1)
        VECPM(2)=ZERO
        VECPM(3)=VECP(2)
        TGSUMU=ZERO
        TGSUML=ZERO
        DO 160 IGQ=1,NGQ
          RADU=RCORM+WKSPCE(2*NTQ+IGQ)*(RCORU-RCORM)
          ZU=ZCORM+WKSPCE(2*NTQ+IGQ)*(ZCORU-ZCORM)
          RADL=RCORM+WKSPCE(2*NTQ+IGQ)*(RCORL-RCORM)
          ZL=ZCORM+WKSPCE(2*NTQ+IGQ)*(ZCORL-ZCORM)
          GSUMU=ZERO
          GSUML=ZERO
          DO 170 ITQ=1,NTQ
            QU(1)=RADU*WKSPCE(ITQ)
            QU(2)=RADU*WKSPCE(NTQ+ITQ)
            QU(3)=ZU
            QL(1)=RADL*WKSPCE(ITQ)
            QL(2)=RADL*WKSPCE(NTQ+ITQ)
            QL(3)=ZL
            CALL SUBV3(QU,PM,RRU)
            CALL SUBV3(QL,PM,RRL)
            SRU=SSIZE3(RRU)
            RU=SQRT(SRU)
            SRL=SSIZE3(RRL)
            RL=SQRT(SRL)
            GSUMU=GSUMU+WTQ(ITQ)/RU
            GSUML=GSUML+WTQ(ITQ)/RL
170       CONTINUE
          TGSUMU=TGSUMU+GSUMU*WGQ(IGQ)*AGQ(IGQ)*RADU
          TGSUML=TGSUML+GSUML*WGQ(IGQ)*AGQ(IGQ)*RADL
160     CONTINUE
        L0VAL=(TGSUMU*TEMPPX+TGSUML*(ONE-TEMPPX))*GQLEN
      END IF

C Computation of N0 when the function is singular. In this case cones
C are placed on the elements upper and lower rims. The N0 integrals are
C computed over these cones. The values are then subtracted from zero
C to give the N0 value.

C  For the upper cone
C   Compute the length CONLU and the number of divisions NODIVU 
      CONLNU=VERTU(1)*ROOT2
      NODIVU=CONLNU/GQLEN+1
      IF (VERTU(1).LT.EGEOM) NODIVU=0
      FLNDU=DBLE(NODIVU)

C  For the lower cone
C   Compute the length CONLL and the number of divisions NODIVL 
      CONLNL=VERTL(1)*ROOT2
      NODIVL=CONLNL/GQLEN+1
      IF (VERTL(1).LT.EGEOM) NODIVL=0
      FLNDL=DBLE(NODIVL)

      IF (LPONEL.AND.LNK) THEN
        TGSUMU=ZERO
        DO 180 IDIV=1,NODIVU
          FLID=DBLE(IDIV)
          DO 190 IGQ=1,NGQ
            GSUMU=ZERO
            AGQX=(FLID-ONE+AGQ(IGQ))/FLNDU
            RCORU=AGQX*VERTU(1)
            ZCORU=VERTU(2)+(ONE-AGQX)*VERTU(1)
            DO 200 ITQ=1,NTQ
              QU(1)=RCORU*WKSPCE(ITQ)
              QU(2)=RCORU*WKSPCE(NTQ+ITQ)
              QU(3)=ZCORU
              NORMQU(1)=WKSPCE(ITQ)/ROOT2
              NORMQU(2)=WKSPCE(NTQ+ITQ)/ROOT2
              NORMQU(3)=ONE/ROOT2
              DNPNQU=DOT3(VECPM,NORMQU)
              CALL SUBV3(QU,PM,RRU)
              SRU=SSIZE3(RRU)
              RU=SQRT(SRU)
              CRU=RU*SRU
              RUNP=DOT3(RRU,VECPM)/RU
              RUNQ=-DOT3(RRU,NORMQU)/RU
              RUN=RUNP*RUNQ
              GSUMU=GSUMU+WTQ(ITQ)*(DNPNQU+RUN+RUN+RUN)/CRU
200         CONTINUE
            TGSUMU=TGSUMU+WGQ(IGQ)*GSUMU*RCORU
190       CONTINUE
180     CONTINUE
        TGSUML=ZERO
        DO 185 IDIV=1,NODIVL
          FLID=DBLE(IDIV)
          DO 195 IGQ=1,NGQ
            GSUML=ZERO
            AGQX=(FLID-ONE+AGQ(IGQ))/FLNDL
            RCORL=AGQX*VERTL(1)
            ZCORL=VERTL(2)-(ONE-AGQX)*VERTL(1)
            DO 205 ITQ=1,NTQ
              QL(1)=RCORL*WKSPCE(ITQ)
              QL(2)=RCORL*WKSPCE(NTQ+ITQ)
              QL(3)=ZCORL
              NORMQL(1)=WKSPCE(ITQ)/ROOT2
              NORMQL(2)=WKSPCE(NTQ+ITQ)/ROOT2
              NORMQL(3)=-ONE/ROOT2
              DNPNQL=DOT3(VECPM,NORMQL)
              CALL SUBV3(QL,PM,RRL)
              SRL=SSIZE3(RRL)
              RL=SQRT(SRL)
              CRL=RL*SRL
              RLNP=DOT3(RRL,VECPM)/RL
              RLNQ=-DOT3(RRL,NORMQL)/RL
              RLN=RLNP*RLNQ
              GSUML=GSUML+WTQ(ITQ)*(DNPNQL+RLN+RLN+RLN)/CRL
205         CONTINUE
            TGSUML=TGSUML+WGQ(IGQ)*GSUML*RCORL
195       CONTINUE
185     CONTINUE
        IF (NODIVU.GT.0.AND.NODIVL.GT.0)
     *   N0VAL=-(VERTU(1)*TGSUMU/FLNDU+VERTL(1)*TGSUML/FLNDL)/ROOT2
        IF (NODIVU.EQ.0.AND.NODIVL.GT.0)
     *   N0VAL=-VERTL(1)*TGSUML/FLNDL/ROOT2
        IF (NODIVU.GT.0.AND.NODIVL.EQ.0)
     *   N0VAL=-VERTU(1)*TGSUMU/FLNDU/ROOT2
        IF (LSWAP) N0VAL=-N0VAL
      END IF

      IF (LPONEL.AND.LLK) SUML=L0VAL
      IF (LPONEL.AND.LNK) SUMN=(N0VAL-SKO2*L0VAL)


C LOOP THROUGH QUADRATURE POINTS AND ACCUMULATE INTEGRALS

      DO 210 IGQ=1,NGQ
        RAD=QA(1)+AGQ(IGQ)*(QBMA(1))
        Q(3)=QA(2)+AGQ(IGQ)*(QBMA(2))
        TSUML=ZERO
        TSUMM=ZERO
        TSUMMT=ZERO
        TSUMN=ZERO
        DO 220 ITQ=1,NTQ
C A: Set Q(1..3) the point on the element (Q(3) already set).
          Q(1)=RAD*WKSPCE(ITQ)
          Q(2)=RAD*WKSPCE(NTQ+ITQ)
C B: Set NORMQQ(1..3) the unit outward normal to the element at Q(1..3).
          IF (LMK.OR.LNK) THEN
            NORMQQ(1)=NORMQ(1)*WKSPCE(ITQ)
            NORMQQ(2)=NORMQ(1)*WKSPCE(NTQ+ITQ)
            NORMQQ(3)=NORMQ(2)
          END IF
C C: Compute DNPNQ [dot product of VECPM and NORMQQ]
          IF (LNK) DNPNQ=VECPM(1)*NORMQQ(1)+VECPM(3)*NORMQQ(3)
C D: Compute RR(1..3) = PM(1..3) - Q(1..3)
          CALL SUBV3(PM,Q,RR)
C E: Compute R [modulus of RR(1..3)] and SR [R-squared].
          SR=SSIZE3(RR)
          R=FNSQRT(SR)
C F: Compute CR [R-cubed].
          IF (LNK) CR=R*SR
C G: Compute RNQ [derivative of R with respect to NORMQQ].
          IF (LMORN) RNQ=-DOT3(RR,NORMQQ)/R
C H: Compute RNP [derivative of R with respect to VECP].
          IF (LMTORN) RNP=(RR(1)*VECPM(1)+RR(3)*VECPM(3))/R
C I: Compute RNPRNQ [RNP*RNQ].
          IF (LNK) RNPRNQ=RNP*RNQ
C J: Compute RNPNQ [second derivative of R with respect to VECP and NORMQ].
          IF (LNK) RNPNQ=-(DNPNQ+RNPRNQ)/R

C Set values of FPG0,FPG0R,FPG0RR [the Green's funstion and its first
C and second derivatives with respect to R when K=(0,0)]. Only
C necessary when P lies on the element or K<>(0,0).
          IF (LPONEL.AND.(.NOT.KZERO)) THEN
            IF (LLORN) FPG0=ONE/R
            IF (LMSORN) FPG0R=-ONE/SR
            IF (LNK) FPG0RR=TWO/CR
          END IF

C Case K is zero. Note IF LPONEL=.TRUE. then Lk and Nk values have already
C  been computed.
          IF (KZERO) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary).
C L: Compute SKR [K*R*K*R] (unnecessary).
C M: Compute E [exp(IKR)] (unnecessary).
C N: Compute Green's function FPG [E/R] (real)
            IF (LLK) FPG=CMPLX(ONE/R,ZERO)
C O: Multiply Lk kernel by weight and add to sum TSUML (real)
            IF (LLK) THEN
              IF (.NOT.LPONEL) TSUML=TSUML+
     *         CMPLX(WTQ(ITQ)*DBLE(FPG),ZERO)
            END IF
C P: Compute FPGR, derivative of FPG with respect to R (real)
            IF (LMSORN) FPGR=CMPLX(-ONE/SR,ZERO)
C Q: Compute WFPGR, the weight multiplied by FPGR (real)
            IF (LMORMT) WFPGR=CMPLX(WTQ(ITQ)*DBLE(FPGR),ZERO)
C R: Compute Mk kernel multiplied by weight and add to sum TSUMM (real)
            IF (LMK) TSUMM=TSUMM+CMPLX(DBLE(WFPGR)*RNQ,ZERO)
C S: Compute Mkt kernel multiplied by weight and add to sum TSUMMT (real)
            IF (LMKT) TSUMMT=TSUMMT+CMPLX(DBLE(WFPGR)*RNP,ZERO)
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R (real)
              FPGRR=CMPLX(TWO/CR,ZERO)
C U: Multiply Nk kernel by weight and add to sum SUMN.
              IF (.NOT.LPONEL) TSUMN=TSUMN+
     *         CMPLX(WTQ(ITQ)*(DBLE(FPGR)*RNPNQ+
     *         DBLE(FPGRR)*RNPRNQ),ZERO)
            END IF

C Case K is real.
          ELSE IF (KREAL) THEN
C K: Compute KR [K*R] (real) and IKR [i*K*R] (imag)
            KR=CMPLX(DBLE(K)*R,ZERO)
            IKR=ICMULT(KR)
C L: Compute SKR [K*R*K*R].
            SKR=CMPLX(DBLE(KR)*DBLE(KR),ZERO)
C M: Compute E [exp(IKR)] (comp)
            E=FNEXP(IKR)
C N: Compute Green's function FPG [E/R] (comp)
            IF (LLK) FPG=E/R
C O: Multiply Lk kernel by weight and add to sum TSUML (comp)
            IF (LLK) THEN
              IF (.NOT.LPONEL) TSUML=TSUML+WTQ(ITQ)*FPG
              IF (LPONEL) TSUML=TSUML+WTQ(ITQ)*(FPG-FPG0)
            END IF
C P: Compute FPGR, derivative of FPG with respect to R (comp)
            IF (LMSORN) THEN
              EOSR=E/SR
              FPGR=ICMULT(EOSR)*DBLE(KR)-EOSR
            END IF
            IF (LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR (comp)
              WFPGR=WTQ(ITQ)*FPGR
C R: Compute Mk kernel multiplied by weight and add to sum TSUMM (comp)
              IF (LMK) TSUMM=TSUMM+WFPGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum TSUMMT (comp)
              IF (LMKT) TSUMMT=TSUMMT+WFPGR*RNP
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R (comp)
              FPGRR=E*(TWO-IKR-IKR-SKR)/CR
C U: Multiply Nk kernel by weight and add to sum TSUMN (comp)
              IF (.NOT.LPONEL) THEN
                TSUMN=TSUMN+WTQ(ITQ)*(FPGR*RNPNQ+FPGRR*RNPRNQ)
              ELSE
                TSUMN=TSUMN+WTQ(ITQ)*((FPGR-FPG0R)*RNPNQ+
     *           (FPGRR-FPG0RR)*RNPRNQ+DBLE(SKO2)*FPG0)
              END IF
            END IF

C Case K is imaginary
          ELSE IF (KIMAG) THEN
C K: Compute KR [K*R] (imag) and IKR [i*K*R] (real)
            KR=IRMULT(DIMAG(K)*R)
            IKR=CMPLX(-DIMAG(KR),ZERO)
C L: Compute SKR [K*R*K*R] (real)
            SKR=CMPLX(-DIMAG(KR)*DIMAG(KR),ZERO)
C M: Compute E [exp(IKR)] (real)
            E=FNEXP(IKR)
C N: Compute Green's function FPG [E/R] (real)
            IF (LLK) FPG=CMPLX(DBLE(E)/R,ZERO)
C O: Multiply Lk kernel by weight and add to sum TSUML (real)
            IF (LLK) THEN
              IF (.NOT.LPONEL) TSUML=TSUML+
     *         CMPLX(WTQ(ITQ)*DBLE(FPG),ZERO)
              IF (LPONEL) TSUML=TSUML+
     *         CMPLX(WTQ(ITQ)*DBLE(FPG-FPG0),ZERO)
            END IF
C P: Compute FPGR, derivative of FPG with respect to R (real)
            IF (LMSORN) THEN
              EOSR=CMPLX(DBLE(E)/SR,ZERO)
              FPGR=CMPLX(DBLE(EOSR)*DBLE(IKR-ONE),ZERO)
            END IF
            IF (LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR (real)
              WFPGR=CMPLX(WTQ(ITQ)*DBLE(FPGR),ZERO)
C R: Compute Mk kernel multiplied by weight and add to sum TSUMM.
              IF (LMK) TSUMM=TSUMM+CMPLX(DBLE(WFPGR)*RNQ,ZERO)
C S: Compute Mkt kernel multiplied by weight and add to sum TSUMMT.
              IF (LMKT) TSUMMT=TSUMMT+CMPLX(DBLE(WFPGR)*RNP,ZERO)
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R (real)
              FPGRR=CMPLX(DBLE(E)*DBLE(TWO-IKR-IKR-SKR)/CR,ZERO)
C U: Multiply Nk kernel by weight and add to sum TSUMN.
              IF (.NOT.LPONEL) THEN
                TSUMN=TSUMN+CMPLX(WTQ(ITQ)*(DBLE(FPGR)*RNPNQ
     *           +DBLE(FPGRR)*RNPRNQ),ZERO)
              ELSE
                TSUMN=TSUMN+CMPLX(WTQ(ITQ)*(DBLE(FPGR-FPG0R)*RNPNQ+
     *           DBLE(FPGRR-FPG0RR)*RNPRNQ+DBLE(SKO2)*FPG0),ZERO)
              END IF
            END IF

C Case K is complex.
          ELSE IF (KCOMP) THEN
C K: Compute KR [K*R] and IKR [i*K*R].
            KR=K*R
            IKR=ICMULT(KR)
C L: Compute SKR [K*R*K*R].
            SKR=SK*SR
C M: Compute E [exp(IKR)].
            E=FNEXP(IKR)
C N: Compute Green's function FPG [E/R].
            IF (LLK) FPG=E/R
C O: Multiply Lk kernel by weight and add to sum SUML.
            IF (LLK) THEN
              IF (.NOT.LPONEL) TSUML=TSUML+WTQ(ITQ)*FPG
              IF (LPONEL) TSUML=TSUML+WTQ(ITQ)*(FPG-FPG0)
            END IF
C P: Compute FPGR, derivative of FPG with respect to R.
            IF (LMSORN) THEN
              EOSR=E/SR
              FPGR=ICMULT(EOSR)*KR-EOSR
            END IF
            IF (LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR.
              WFPGR=WTQ(ITQ)*FPGR
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
              IF (LMK) TSUMM=TSUMM+WFPGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
              IF (LMKT) TSUMMT=TSUMMT+WFPGR*RNP
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R.
              FPGRR=E*(TWO-IKR-IKR-SKR)/CR
C U: Multiply Nk kernel by weight and add to sum SUMN.
              IF (.NOT.LPONEL) THEN
                TSUMN=TSUMN+WTQ(ITQ)*(FPGR*RNPNQ+FPGRR*RNPRNQ)
              ELSE
                TSUMN=TSUMN+WTQ(ITQ)*((FPGR-FPG0R)*RNPNQ+
     *           (FPGRR-FPG0RR)*RNPRNQ+SKO2*FPG0)
              END IF
            END IF
          END IF
220     CONTINUE
        GWRO2=GQLEN*WGQ(IGQ)*RAD/TWO
        SUML=SUML+GWRO2*TSUML
        SUMM=SUMM+GWRO2*TSUMM
        SUMMT=SUMMT+GWRO2*TSUMMT
        SUMN=SUMN+GWRO2*TSUMN
210   CONTINUE

      DISLK=SUML
      DISMK=SUMM
      DISMKT=SUMMT
      DISNK=SUMN

998   CONTINUE

      LFAIL=LERROR

      END

C ***********************************************************************

C ***********************************************************************

C Subordinate routines.

C complex function ICMULT: Returns the result of multiplying a complex
C  number by i.
      COMPLEX*16 FUNCTION ICMULT(Z)
      COMPLEX*16 Z
      REAL*8 REZ,IMZ
      REZ=DBLE(Z)
      IMZ=DIMAG(Z)
      ICMULT=CMPLX(-IMZ,REZ)
      END

C complex function IRMULT: Returns the result of multiplying a real
C  number by i.
      COMPLEX*16 FUNCTION IRMULT(X)
      REAL*8 X,ZERO
      ZERO=0.0D0
      IRMULT=CMPLX(ZERO,X)
      END
