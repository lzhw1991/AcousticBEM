C***************************************************************
C            Subroutine H2LC by Stephen Kirkup                        
C***************************************************************
C
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/H2LC.FOR
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
C This subroutine computes the discrete form of the 2-dimensional
C Helmholtz integral operators Lk,Mk,Mkt, and Nk over a straight line
C element. The subroutine is useful in integral equation methods for the
C solution of 2-dimensional Helmholtz problems.
C
C The following diagram shows how the subroutine is to be used. A main
C program is required along with the definition of FNHANK, FNLOG and
C FNSQRT. The package is the contents of this file.
C
C                                   ....................................
C                                   :                                  :
C                                   :      --------  ------- --------  :
C Routines to be supplied --------->:      |FNHANK|  |FNLOG| |FNSQRT|  :
C                                   :      --------  ------- --------  :
C                                   :........|.........|.......|.......:        
C                                            |         |       |
C                                   .........<.........<.......<........
C                                   :        |         |       |       :
C                                   :        |         |       |       :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|           H2LC         |   :
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
C
C                                            -------    --------
C Routines to be supplied --------->         |FNEXP|    |FNSQRT|
C                                            -------    --------
C                                               |          |      
C                                   ............<..........<............
C                                   .           |          |           .
C      ----------------------       .     --------------------------   .
C      |                    |       .     |                        |   .
C      |   MAIN PROGRAM     |------->-----|          H3ALC         |   .
C      |                    |       .     |                        |   .
C      ----------------------       .     --------------------------   .
C                                   .                 |                .
C                                   .                 >                .
C                                   .                 |                .
C                                   .      ------------------------    .
C                                   .      | Subordinate routines |    .
C                                   .      ------------------------    .
C                                   .                                  .
C                                   ....................................
C
C
C The subroutine has the form:

C      SUBROUTINE H2LC(K,P,VECP,QA,QB,LPONEL,
C     * MAXNQ,NQ,AQ,WQ,
C     * LVALID,EK,EGEOM,EQRULE,LFAIL,
C     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C The parameters to the subroutine
C ================================
C Wavenumber (input)
C complex K: The complex wavenumber.

C Point (input)
C real P(2): The Cartesian coordinates of the point p.
C real VECP(2): The Cartesian components of the unit normal at p. 
C  Required in the computation of DISMKT and DISNK. The squares of the
C  components must sum to one.

C Geometry of element (input)
C real QA(2): The Cartesian coordinates of the first edge of the element.
C real QB(2): The Cartesian coordinates of the second edge of the element.
C logical LPONEL: If the point P(2) lies on QA-QB then LPONEL must be set
C  .TRUE., otherwise LPONEL must be set .FALSE..

C Quadrature rule (input)
C integer MAXNQ: The limit on the size of the quadrature rule. The value
C  should not be changed between calls of H2LC. MAXNQ>=1.
C integer NQ: The actual number of quadrature rule points. 1=<NQ<=MAXNQ.
C real AQ(MAXNQ): The quadrature rule abscissae. The values must lie in
C  the domain [0,1] and be in ascending order.
C real WQ(MAXNQ): The quadrature rule weights which correspond to the
C  quadrature points in AQ. The components of WQ must sum to one.

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
C  parameters DISLK, DISMK, DISMKT and DISNK will all be zero. A 
C  diagnosis will be given in the file H2LC.ERR.

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

C External modules to be supplied
C ===============================
C Subroutine FNHANK(Z,H): complex Z (input), complex H(0:1) (output):
C  Must return the spherical Hankel functions of the first kind of
C  order zero H(0) and the spherical Hankel function of the first
C  kind of order one H(1). This subroutine is important when k is
C  non-zero.
C real function FNSQRT(X): real X : Must return the square root of X.
C real function FNLOG(X): real X : Must return the natural logarithm
C  of X.
 
C Notes on the validation parameters
C ----------------------------------
C  (1) If LVALID=.TRUE. then a file named H2LC.ERR should be open when
C   subroutine H2LC is entered. Use a command such as 
C         OPEN(UNIT=10,FILE='H2LC.ERR',STATUS='UNKNOWN')
C  at the beginning of the calling program and the corresponding command
C         CLOSE(10)
C  at the end of the calling program.
C  This file accumulates the error messages and warnings from the 
C   subroutine. If this file is not open then an error message is 
C   output to standard output and LFAIL=.TRUE.
C (2) If LVALID=.TRUE. then EGEOM must be less than 1% of the maximum
C  absolute coordinate of QA and QB.
C (3) If LVALID=.TRUE. then EQRULE must be less than 10%/NQ.

C Notes on the geometric parameters
C ---------------------------------
C  (1) P, QA and QB must be distinct points (with respect to EGEOM).
C  (2) If LPONEL=.TRUE. then P must lie on element QA-QB. If 
C   LPONEL=.FALSE. then P must not lie on QA-QB.
C  (3) The normal to the element is defined to be
C             [-QB(2)+QA(2), QB(1)-QA(1)] normalised.
C   Hence the normal is to the right on the line QA-QB.
C  (4) If LPONEL=.TRUE. (and either LMKT=.TRUE. or LNK=.TRUE.) then 
C   VECP must be [-QB(2)+QA(2), QB(1)-QA(1)] normalised.

C Notes on the quadrature rule parameters
C ---------------------------------------
C  (1) The quadrature rule is assumed to compute linear functions
C   exactly (with respect to EQRULE).
C  (2) If LPONEL=.TRUE. then the first derivative of the Lk and Nk
C   functions on the element are discontinuous at P. The input
C   quadrature rule should take account of this by using a
C   composite rule that divides at P in these cases.

C External modules provided
C =========================
C complex function ICMULT(Z): Returns the result of multiplying
C  complex Z by i.
C real function SIZE2: Returns the modulus of a 2-vector.
C real function SSIZE2: Returns the square of the modulus of a 2-vector.
C real function DOT2: Returns the dot product of two 2-vectors.
C Subroutine SUBV2: Gives the output in VEC for the subtraction.

C Subroutine

      SUBROUTINE H2LC(K,P,VECP,QA,QB,LPONEL,
     * MAXNQ,NQ,AQ,WQ,
     * LVALID,EK,EGEOM,EQRULE,LFAIL,
     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C Wavenumber
      COMPLEX*16 K

C Point
      REAL*8     P(2)
      REAL*8     VECP(2)

C Geometry of element
      REAL*8     QA(2)
      REAL*8     QB(2)
      LOGICAL    LPONEL

C Quadrature rule
      INTEGER    MAXNQ
      INTEGER    NQ
      REAL*8     AQ(MAXNQ)
      REAL*8     WQ(MAXNQ)

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

C External functions to be supplied
      REAL*8     FNSQRT
      REAL*8     FNLOG

C External functions provided
      COMPLEX*16 ICMULT
      REAL*8     SIZE2
      REAL*8     SSIZE2
      REAL*8     DOT2

C Variables that remain constant throughout subroutine, once initialised
C  Constants
      REAL*8     ZERO,HALF,ONE,TWO,FOUR,PI,TWOPI,OO2PI
      COMPLEX*16 CZERO,CONE
C  H2LC.ERR open status and unit number
      LOGICAL    LOPEN
      INTEGER    IU
C  Useful geometric values
      REAL*8     QBMA(2),PMA(2),PMB(2),NORMQ(2)
      REAL*8     QLEN,PQALEN,PQBLEN,DNPNQ
C  Magnitude (one norm) of geometric values
      REAL*8     MAGVP,MAGBMA,MAGPMA,MAGPMB
C  Control values relating to required solutions     
      LOGICAL    LLORN,LMORN,LMTORN,LMSORN,LMORMT,LMKSV,LMKTSV
C  Values that classify K as 'zero','real','imaginary', or 'complex'
      LOGICAL    KZERO,KCOMP,KREAL,KIMAG
C  Useful constants related to K
      COMPLEX*16 KO4,SK,SKO4,SKO2
      REAL*8     REK,IMK,REKO4,IMKO4,RESKO2,RESKO4

C Variables used in the validation of the input data
C  Error variables
      LOGICAL    LERROR,LOCERR
C  Temporary variables
      REAL*8     TEMP,TEMP1,TEMP2,TEMP3,PX,CPX,VAL,SQLEN,TOP
C  Corresponding error bounds to the temporary variables
      REAL*8     ETEMP,ETEMP1,ETEMP2,ETEMP3,EPX,ESQLEN,ETOP
C  Index of quadrature point
      INTEGER*4  JQ

C Variables used in the operation of the quadrature rule
C  Variables for the accumulation of integrals
      COMPLEX*16 SUML,SUMM,SUMMT,SUMN
C  Index of quadrature point
      INTEGER*4  IQ
C  Geometric variables
      REAL*8     Q(2),RR(2)
      REAL*8     SR,R,RNQ,RNP,RNPRNQ,RNPNQ
C  Green's functions for the Laplace operator and its r-derivatives
      REAL*8     G0,G0R,G0RR
C  Geometric variables related to K
      REAL*8     REKR,IMKR
      COMPLEX*16 KR
C  Green's functions for the Helmholtz operators and its r-derivatives
      COMPLEX*16 G,GR,GRR
C  Variable for the storage of the value of the Hankel functions
      COMPLEX*16 H(0:1)
C  Other variables
      COMPLEX*16 WGR


C INITIALISATION

C Set constants
      ZERO=0.0D0
      HALF=0.5D0
      ONE=1.0D0
      TWO=2.0D0
      FOUR=4.0D0
      PI=3.1415926535897932
      TWOPI=TWO*PI
      OO2PI=ONE/TWOPI
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)

C Initialise useful geometrical information.
      CALL SUBV2(QB,QA,QBMA)
      CALL SUBV2(P,QA,PMA)
      CALL SUBV2(P,QB,PMB)
      QLEN=SIZE2(QBMA)
      PQALEN=SIZE2(PMA)
      PQBLEN=SIZE2(PMB)


C Set LFAIL,LERROR to FALSE
      LFAIL=.FALSE.
      LERROR=.FALSE.

C If LVALID is false then bypass the validation process
      IF (.NOT.LVALID) GOTO 900


C VALIDATION OF INPUT

C Check that the file H2LC.ERR has been opened, if not then exit
      INQUIRE(FILE='H2LC.ERR',OPENED=LOPEN)
      IF (.NOT.LOPEN) THEN
        WRITE(*,*) 'ERROR(H2LC) - File for error messages "H2LC.ERR"'
        WRITE(*,*) ' is not open'
        WRITE(*,*) 'Execution of H2LC is aborted'
        LERROR=.TRUE.
        GOTO 998
      END IF

C Find out the unit number of the file H2LC.ERR
      INQUIRE(FILE='H2LC.ERR',NUMBER=IU)


C Check that the subroutine FNHANK is satisfactory
      IF (LVALID.AND.DBLE(K).GT.EK) THEN
        CALL FNHANK(CONE,H)
        LERROR=.FALSE.
        IF (ABS(DBLE(H(0))-0.7652).GT.1.0D-1) LERROR=.TRUE.
        IF (ABS(AIMAG(H(0))-0.0882).GT.1.0D-1) LERROR=.TRUE.
        IF (ABS(DBLE(H(1))-0.4401).GT.1.0D-1) LERROR=.TRUE.
        IF (ABS(AIMAG(H(1))+0.7812).GT.1.0D-1) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(IU,*) 'ERROR(H2LC) - in Hankel function routine FNHANK'
          STOP
        END IF
      END IF


C Check that the function FNLOG is satisfactory
      IF (LVALID) THEN
        LERROR=.FALSE.
        IF (ABS(FNLOG(1.0D0)).GT.0.01) LERROR=.TRUE.
        IF (ABS(FNLOG(100.0D0)-4.6051702).GT.0.01) LERROR=.TRUE.
        IF (ABS(FNLOG(0.001D0)+6.9077553).GT.0.01) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(IU,*) 'ERROR(H2LC) - in log function routine FNLOG'
          STOP
        END IF
      END IF

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

        


C Set useful constants
      IF (LMKT.OR.LNK) MAGVP=ABS(VECP(1))+ABS(VECP(2))
      MAGBMA=ABS(QBMA(1))+ABS(QBMA(2))
      MAGPMA=ABS(PMA(1))+ABS(PMA(2))
      MAGPMB=ABS(PMB(1))+ABS(PMB(2))

C Set LERROR to .FALSE.
      LERROR=.FALSE.

C Check the tolerances EK,EGEOM,EQRULE are positive
      IF (EK.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Parameter EK must be positive'
        LERROR=.TRUE.
      END IF
      IF (EGEOM.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Parameter EGEOM must be positive'
        LERROR=.TRUE.
      END IF
      IF (EQRULE.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Parameter EQRULE must be positive'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C Validation of the geometric information
C  Check VECP is of unit magnitude
      IF (LMKT.OR.LNK) THEN
        TEMP=SSIZE2(VECP)
        ETEMP=TWO*EGEOM*MAGVP
        IF (ABS(TEMP-ONE).GT.ETEMP) THEN
          WRITE(IU,*) 'ERROR(H2LC) - Parameter VECP must have unit'
          WRITE(IU,*) 'magnitude'
          LERROR=.TRUE.
        END IF
      END IF
C  Check QA and QB are distinct points
      IF (MAGBMA.LT.FOUR*EGEOM) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Check specification of QA and QB'
        LERROR=.TRUE.
      END IF
C  Check P is distinct from QA and QB
      IF (MAGPMA.LT.FOUR*EGEOM.OR.MAGPMB.LT.FOUR*EGEOM) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Check specification of P,QA,QB'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C Check that EGEOM is not too large
      IF (EGEOM.GT.MAX(ABS(QA(1)),ABS(QA(2)),ABS(QB(1)),
     * ABS(QB(2)))/100.0D0) THEN
        WRITE(IU,*) 'ERROR(H2LC) - EGEOM is set too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C  Check LPONEL
      SQLEN=QLEN*QLEN
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
        WRITE(IU,*) 'ERROR(H2LC) - LPONEL should be .FALSE.'
        LERROR=.TRUE.
      END IF
      IF (.NOT.LPONEL.AND.TEMP1.LT.ETEMP1.AND.TEMP2.LT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H2LC) - LPONEL should be .TRUE.'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C  Check that VECP is THE normal to QA-QB when P is on QA-QB
      IF (LPONEL.AND.(LMKT.OR.LNK)) THEN
        IF (ABS(DOT2(VECP,QBMA)).GT.EGEOM*(TWO*MAGVP+MAGBMA)) THEN
          WRITE(IU,*) 'ERROR(H2LC) - VECP must be normal to QA-QB'
          LERROR=.TRUE.
        END IF
        IF (.NOT.LERROR) THEN
          IF (-VECP(1)*QBMA(2)+VECP(2)*QBMA(1).LT.ZERO) THEN
            WRITE(IU,*) 'ERROR(H2LC) - Replace VECP by -VECP'
            LERROR=.TRUE.
          END IF
        END IF
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF


     
        

C Validation of the quadrature rule data
C  Check MAXNQ,NQ
      IF (MAXNQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Must have MAXNQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NQ.LE.0.OR.NQ.GT.MAXNQ) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Must have 0 < NQ <= MAXNQ'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C  Check that EQRULE is not too large
      IF (EQRULE.GT.ONE/FLOAT(NQ)/10.0D0) THEN
        WRITE(IU,*) 'ERROR(H2LC) - EQRULE is set too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

C  Check that the abscissae AQ all lie in [0,1]
      LOCERR=.FALSE.
      DO 100 JQ=1,NQ
        IF (AQ(JQ).LT.-EQRULE.OR.AQ(JQ).GT.ONE+EQRULE) LOCERR=.TRUE.
100   CONTINUE
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H2LC) - Components of AQ are outside of'
        WRITE(IU,*) '[0,1]'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

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
          EPX=TWO*EGEOM*(ONE+PX)/ABS(QBMA(1))
        ELSE
          PX=ABS(PMA(2)/QBMA(2))
          EPX=TWO*EGEOM*(ONE+PX)/ABS(QBMA(2))
        END IF
      CPX=ONE-PX
      END IF
      DO 110 JQ=1,NQ
        TEMP1=TEMP1+WQ(JQ)
        ETEMP1=ETEMP1+EQRULE
        TEMP2=TEMP2+WQ(JQ)*AQ(JQ)
        ETEMP2=ETEMP2+EQRULE*(ABS(WQ(JQ))+ABS(AQ(JQ)))
        IF (LPONEL) THEN
          TEMP3=TEMP3+WQ(JQ)*ABS(AQ(JQ)-PX)
          ETEMP3=ETEMP3+EQRULE*ABS(AQ(JQ)-PX)+ABS(WQ(JQ))*(EQRULE+EPX)
        END IF
110   CONTINUE
      IF (ABS(TEMP1-ONE).GT.ETEMP1.OR.
     *  ABS(TEMP2-HALF).GT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H2LC) - In AQ-WQ quadrature rule'
        LERROR=.TRUE.
      END IF
      IF (LPONEL) THEN
        VAL=(PX*PX+CPX*CPX)/TWO
        IF (ABS(TEMP3-VAL).GT.ETEMP3) THEN
          WRITE(IU,*) 'WARNING - AQ-WQ rule should not assume'
          WRITE(IU,*) 'continuity when LPONEL=.TRUE.'
          LERROR=.TRUE.
        END IF
      END IF

C Check that a quadrature point does not coincide with P
      IF (LPONEL.AND.(LLK.OR.LNK).AND.
     * (ABS(DBLE(K)).GT.EK.OR.ABS(DIMAG(K)).GT.EK)) THEN
        LOCERR=.FALSE.
        DO 130 JQ=1,NQ
          IF (ABS(AQ(JQ)-PX).LT.EQRULE+EPX) LOCERR=.TRUE.
130     CONTINUE
        IF (LOCERR) THEN
          WRITE(IU,*) 'ERROR(H2LC) - One of the quadrature points'
          WRITE(IU,*) ' coincides with the point P'
          LERROR=.TRUE.
        END IF
      END IF

C  If LERROR then exit H2LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H2LC is aborted'
        GOTO 998
      END IF

900   CONTINUE


C FURTHER INITIALISATION

C Set the values  of the discrete operators to zero.
      DISLK=CZERO
      DISMK=CZERO
      DISMKT=CZERO
      DISNK=CZERO

C Exit if no solutions are required
      IF (.NOT.(LLK.OR.LMK.OR.LMKT.OR.LNK)) GOTO 998

C Initialise useful control information
      LLORN=LLK.OR.LNK
      LMORN=LMK.OR.LNK
      LMSORN=LMK.OR.LMKT.OR.LNK
      LMTORN=LMKT.OR.LNK
      LMORMT=LMK.OR.LMKT
      IF (LPONEL) THEN
        LMKSV=LMK
        LMKTSV=LMKT
        LMK=.FALSE.
        LMKT=.FALSE.
      END IF

C Initialise useful geometric information
      IF (LMORN) THEN
        NORMQ(1)=-QBMA(2)/QLEN
        NORMQ(2)=QBMA(1)/QLEN
      END IF
      IF (LNK) DNPNQ=DOT2(VECP,NORMQ)

C Useful constants related to K
      REK=DBLE(K)
      IMK=DIMAG(K)
      SK=K*K
      KO4=K/FOUR
      SKO4=SK/FOUR
      SKO2=SKO4*TWO
      REKO4=DBLE(KO4)
      IMKO4=DIMAG(KO4)
      RESKO2=DBLE(SKO2)
      RESKO4=DBLE(SKO4)

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


C Initialise SUML-SUMN, the variables that accumulate 
C  the integrals Lk,Mk,Mkt,Nk (divided by QLEN)
      SUML=CZERO
      SUMM=CZERO
      SUMMT=CZERO
      SUMN=CZERO
      IF (LPONEL) THEN
        IF (LLORN) SUML=(ONE-(PQALEN*FNLOG(PQALEN)
     *   +PQBLEN*LOG(PQBLEN))/QLEN)/TWOPI
        IF (LNK) SUMN=-(ONE/PQALEN+ONE/PQBLEN)/QLEN/TWOPI-SKO2*SUML
      END IF


C IF LPONEL AND K=0 THEN SOLUTION IS COMPLETE

      IF (LPONEL.AND.KZERO) GOTO 999


C LOOP THROUGH THE QUADRATURE POINTS AND ACCUMULATE INTEGRALS

        DO 120 IQ=1,NQ
C A: Set Q(1..2) the point on the element.
          Q(1)=QA(1)+AQ(IQ)*QBMA(1)
          Q(2)=QA(2)+AQ(IQ)*QBMA(2)
C B: Set NORMQ(1..2) (already set).
C C: Compute DNPNQ [dot product of VECP and NORMQ] (already set).
C D: Compute RR(1..2) = P(1..2)-Q(1..2)
          CALL SUBV2(P,Q,RR)
C E: Compute R [modulus of RR(1..2)], SR [R-squared].
          SR=SSIZE2(RR)
          R=FNSQRT(SR)
C F: Compute CR [R-cubed] (unnecessary).
C G: Compute RNQ [derivative of R with respect to NORMQ].
          IF (LMORN) RNQ=-DOT2(RR,NORMQ)/R
C H: Compute RNP [derivative of R with respect to VECP].
          IF (LMTORN) RNP=DOT2(RR,VECP)/R
C I: Compute RNPRNQ [RNP*RNQ].
          IF (LNK) RNPRNQ=RNP*RNQ
C J: Compute RNPNQ [second derivative of R with respect to
C    VECP and NORMQ].
          IF (LNK) RNPNQ=-(DNPNQ+RNPRNQ)/R

C Set values of G0,G0R,G0RR [the Green's function, the first derivative
C of the Green's function with respect to R and the second derivative
C of the Green's function with respect to R when K=(0,0)]. Only
C necessary when P lies on the element and K<>(0,0).
          IF (LPONEL.AND.(.NOT.KZERO)) THEN
            IF (LLORN) G0=-OO2PI*FNLOG(R)
            IF (LMSORN) G0R=-OO2PI/R
            IF (LNK) G0RR=OO2PI/SR
          END IF

C Case K is zero.
          IF (KZERO) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary).
C L: Compute SKR [K*R*K*R] (unnecessary).
C M: Compute H(0..1) the Hankel functions (unnecessary).
            IF (LLK) THEN
C N: Compute G, the Green's function.
              G=CMPLX(-OO2PI*FNLOG(R),ZERO)
C O: Multiply Lk kernel by weight and add to sum SUML.
              SUML=SUML+CMPLX(WQ(IQ)*DBLE(G),ZERO)
            END IF
C P: Compute GR, the derivative of the G with respect to R.
            IF (LMSORN) GR=CMPLX(-OO2PI/R,ZERO)
C Q: Compute WGR, the weight multiplied by GR.
            IF (LMORMT) WGR=CMPLX(WQ(IQ)*DBLE(GR),ZERO)
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
            IF (LMK) SUMM=SUMM+CMPLX(DBLE(WGR)*RNQ,ZERO)
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
            IF (LMKT) SUMMT=SUMMT+CMPLX(DBLE(WGR)*RNP,ZERO)
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R.
              GRR=CMPLX(OO2PI/SR,ZERO)
C U: Multiply Nk kernel by weight and add to sum SUMN
              SUMN=SUMN+
     *         CMPLX(WQ(IQ)*(DBLE(GR)*RNPNQ+DBLE(GRR)*RNPRNQ),ZERO)
            END IF

C Case K is real.
          ELSE IF (KREAL) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary).
            REKR=REK*R
            IMKR=ZERO
            KR=CMPLX(REKR,IMKR)
C L: Compute SKR [K*R*K*R] (unnecessary).
C M: Compute H(0..1) the Hankel functions.
            CALL FNHANK(KR,H)
            IF (LLK) THEN
C N: Compute G, the Green's function.
              G=ICMULT(H(0)/FOUR)
C O: Multiply Lk kernel by weight and add to sum SUML.
              IF (.NOT.LPONEL) SUML=SUML+WQ(IQ)*G
              IF (LPONEL) SUML=SUML+WQ(IQ)*(G-G0)
            END IF
            IF (LMSORN) THEN
C P: Compute GR, the derivative of G.
              GR=ICMULT(-REKO4*H(1))
              IF (.NOT.LPONEL.AND.LMORMT) THEN
C Q: Compute WGR, the weight multiplied by GR.
                WGR=WQ(IQ)*GR
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
                IF (LMK) SUMM=SUMM+WGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
                IF (LMKT) SUMMT=SUMMT+WGR*RNP
              END IF
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G.
              GRR=ICMULT(-RESKO4*(-H(1)/REKR+H(0)))
C U: Multiply Nk kernel by weight and add to sum SUMN.
              IF (.NOT.LPONEL) SUMN=SUMN+WQ(IQ)*(GR*RNPNQ+
     *         GRR*RNPRNQ)
              IF (LPONEL) SUMN=SUMN+WQ(IQ)*((GR-G0R)*RNPNQ+
     *         (GRR-G0RR)*RNPRNQ+RESKO2*G0)
            END IF

C Case K is imaginary.
          ELSE IF (KIMAG) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary).
            REKR=ZERO
            IMKR=IMK*R
            KR=CMPLX(ZERO,IMKR)
C L: Compute SKR [K*R*K*R] (unnecessary).
C M: Compute H(0..1) the Hankel functions. 
C     H(0) is imaginary, H(1) is real.
            CALL FNHANK(KR,H)
            IF (LLK) THEN
C N: Compute G, the Green's function. G is real.
              G=-DIMAG(H(0))/FOUR
C O: Multiply Lk kernel by weight and add to sum SUML.
              IF (.NOT.LPONEL) SUML=SUML+WQ(IQ)*DBLE(G)
              IF (LPONEL) SUML=SUML+WQ(IQ)*DBLE(G-G0)
            END IF
            IF (LMSORN) THEN
C P: Compute GR, the derivative of G.
              GR=-IMKO4*DBLE(H(1))
              IF (.NOT.LPONEL.AND.LMORMT) THEN
C Q: Compute WGR, the weight multiplied by GR.
                WGR=WQ(IQ)*DBLE(GR)
C R: Multiply Mk kernel by weight and add to sum SUMM.
                IF (LMK) SUMM=SUMM+WGR*RNQ
C S: Multiply Mkt kernel by weight and add to sum SUMMT.
                IF (LMKT) SUMMT=SUMMT+WGR*RNP
              END IF
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G.
              GRR=RESKO4*(DBLE(H(1))/IMKR+DIMAG(H(0)))
C U: Multiply Nk kernel by weight and add to sum SUMN.
              IF (.NOT.LPONEL) SUMN=SUMN+WQ(IQ)*(DBLE(GR)*RNPNQ+
     *         DBLE(GRR)*RNPRNQ)
              IF (LPONEL) SUMN=SUMN+WQ(IQ)*(DBLE(GR-G0R)*RNPNQ+
     *         DBLE(GRR-G0RR)*RNPRNQ+RESKO2*G0)
            END IF

C Case K is complex
          ELSE IF (KCOMP) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary).
            KR=K*R
C L: Compute SKR [K*R*K*R] (unnecessary).
C M: Compute H(0..1) the Hankel functions.
            CALL FNHANK(KR,H)
            IF (LLK) THEN
C N: Compute G, the Green's function.
              G=ICMULT(H(0)/FOUR)
C O: Multiply Lk kernel by weight and add to sum SUML.
              IF (.NOT.LPONEL) SUML=SUML+WQ(IQ)*G
              IF (LPONEL) SUML=SUML+WQ(IQ)*(G-G0)
            END IF
            IF (LMSORN) THEN
C P: Compute GR, the derivative of G.
              GR=ICMULT(-KO4*H(1))
              IF (.NOT.LPONEL.AND.LMORMT) THEN
C Q: Compute WGR, the weight multiplied by GR.
                WGR=WQ(IQ)*GR
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
                IF (LMK) SUMM=SUMM+WGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
                IF (LMKT) SUMMT=SUMMT+WGR*RNP
              END IF
            END IF
            IF (LNK) THEN
C T: Compute GRR, the second derivative of G.
              GRR=-ICMULT(SKO4*(-H(1)/KR+H(0)))
C U: Multiply Nk kernel by weight and add to sum SUMN.
              IF (.NOT.LPONEL) SUMN=SUMN+WQ(IQ)*(GR*RNPNQ+
     *         GRR*RNPRNQ)
              IF (LPONEL) SUMN=SUMN+WQ(IQ)*((GR-G0R)*RNPNQ+
     *         (GRR-G0RR)*RNPRNQ+SKO2*G0)
            END IF
          END IF

120     CONTINUE

999     CONTINUE
        IF (LLK) DISLK=QLEN*SUML
        IF (LMK) DISMK=QLEN*SUMM
        IF (LMKT) DISMKT=QLEN*SUMMT
        IF (LNK) DISNK=QLEN*SUMN

        IF (LPONEL) THEN
          LMK=LMKSV
          LMKT=LMKTSV
        END IF

998     CONTINUE

        LFAIL=LERROR

        END


C ***********************************************************************



C complex function ICMULT: Returns the result of multiplying a complex
C  number by i.
      COMPLEX*16 FUNCTION ICMULT(Z)
      COMPLEX*16 Z
      REAL*8 REZ,IMZ
      REZ=DBLE(Z)
      IMZ=DIMAG(Z)
      ICMULT=CMPLX(-IMZ,REZ)
      END
