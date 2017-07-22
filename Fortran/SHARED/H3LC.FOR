C***************************************************************
C          Subroutine H3LC by Stephen Kirkup.                       
C***************************************************************
C
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/H3LC.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C

C This subroutine computes the discrete form of the 3-dimensional
C Helmholtz integral operators Lk,Mk,Mkt, and Nk over a triangle. The
C subroutine is useful in integral equation methods for the solution of
C 3-dimensional Helmholtz problems.
C
C The following diagram shows how the subroutine is to be used. A main
C program is required along with the definition of FNEXP and FNSQRT.
C The package is the contents of this file
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
C      |   MAIN PROGRAM     |------->-----|           H3LC         |   :
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

C      SUBROUTINE H3LC(K,P,VECP,QA,QB,QC,LPONEL,
C     * MAXNQ,NQ,XQ,YQ,WQ,
C     * LVALID,EK,EGEOM,EQRULE,LFAIL,
C     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C The parameters to the subroutine
C ================================
C Wavenumber (input)
C complex K: The complex wavenumber.

C Point (input)
C real P(3): The Cartesian coordinates of the point p.
C real VECP(3): The Cartesian components of the unit normal at p, 
C  required in the computation of DISMKT and DISNK. The squares of the 
C  components must sum to one.

C Geometry of element (input)
C real QA(3): The Cartesian coordinates of the first vertex of the
C  element.
C real QB(3): The Cartesian coordinates of the second vertex of the
C  element.
C real QC(3): The Cartesian coordinates of the third vertex of the
C  element.
C logical LPONEL: If the point P(3) lies on element QA-QB-QC then LPONEL
C  must be set .TRUE., otherwise LPONEL must be set .FALSE..

C Quadrature rule (input)
C integer MAXNQ: The limit on the size of the quadrature rule. The value
C  should not be changed between calls of H3LC. MAXNQ>=1.
C integer NQ: The actual number of quadrature rule points. 1<=NQ<=MAXNQ.
C real XQ(MAXNQ): The x-coordinate of the quadrature rule abscissae. The
C  values must lie in the standard triangle.
C real YQ(MAXNQ): The y-coordinate of the quadrature rule abscissae. The
C  values must lie in the standard triangle.
C real WQ(MAXNQ): The quadrature rule weights which correspond to the
C  quadrature points in XQ and YQ. The components of WQ must sum to one.

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
C  diagnosis will be given in the file H3LC.ERR.

C Choice of discrete forms required (input)
C logical LLK: If discrete form of Lk operator is required then set .TRUE.,
C  otherwise set .FALSE..
C logical LMK: If discrete form of Mk operator is required then set .TRUE.,
C  otherwise set .FALSE..
C logical LMKT: If discrete form of Mkt operator is required then set .TRUE.,
C  otherwise set .FALSE..
C logical LNK: If discrete form of Nk operator is required then set .TRUE.,
C  otherwise set .FALSE..

C Discrete Helmholtz integral operators (output)
C complex DISLK: The discrete Lk integral operator.
C complex DISMK: The discrete Mk integral operator.
C complex DISMKT: The discrete Mkt integral operator.
C complex DISNK: The discrete Nk integral operator.

C External modules to be supplied
C ===============================
C complex FUNCTION FNEXP(Z): complex Z : Must return the exponential
C  of Z.
C real FUNCTION FNSQRT(X): real X: Must return the square root of X.

C Notes on the validation parameters
C ----------------------------------
C  (1) If LVALID=.TRUE. then a file named H3LC.ERR should be open when
C   subroutine HO2LC is entered. Use a command such as 
C         OPEN(UNIT=10,FILE='H3LC.ERR',STATUS='UNKNOWN')
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
C  (1) P, QA, QB and QC must be distinct points (with respect to EGEOM).
C   Also P must not lie on the edges of the triangle QA-QB, QB-QC
C   or QC-QA.
C  (2) If LPONEL=.TRUE. the P must lie on element QA-QB-QC. If 
C   LPONEL=.FALSE. then P must not lie on QA-QB-QC.
C  (3) The normal to the element is defined to be 
C       [QB-QA]x[QC-QA] normalised, where x is the vector cross product.
C   Hence the normal is outward when QA-QB-QC are arranged anticlocwise
C   around the triangle, when viewed from a point a positive distance
C   along the normal.
C  (4) If LPONEL=.TRUE. (and either LMKT=.TRUE. or LNK=.TRUE.) then 
C   VECP must be [QB-QA]x[QC-QA] normalised.
C  (5) The triangle QA-QB-QC should be well-proportioned. That is no
C   side should be much smaller or larger than the other two. If 
C   LFAIL=.TRUE. and one side is more than 100x larger than any other
C   side then the subroutine will fail. If one side is more than 10x 
C   larger than any other then a warning message is given.

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
C complex function ICMULT: Returns the result of multiplying a complex
C  number by i.
C real function AREA: Returns the area of the triangle with the vertices
C  of three vectors.
C real function DIST3: Returns the distance between two 3-vectors.
C real function SDIST3: Returns the square of the distance between two
C  3-vectors.
C real function SIZE3: Returns the modulus of a 3-vector.
C real function SSIZE3: Returns the square of the modulus of a 3-vector.
C real function DOT3: Returns the dot product of two 3-vectors.

C Subroutine
       SUBROUTINE H3LC(K,P,VECP,QA,QB,QC,LPONEL,
     * MAXNQ,NQ,XQ,YQ,WQ,
     * LVALID,EK,EGEOM,EQRULE,LFAIL,
     * LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C Wavenumber
      COMPLEX*16 K

C Point
      REAL*8    P(3)
      REAL*8    VECP(3)

C Geometry of element
      REAL*8    QA(3)
      REAL*8    QB(3)
      REAL*8    QC(3)
      LOGICAL   LPONEL

C Quadrature rule
      INTEGER   MAXNQ
      INTEGER   NQ
      REAL*8    XQ(MAXNQ)
      REAL*8    YQ(MAXNQ)
      REAL*8    WQ(MAXNQ)

C Validation and control parameters
      LOGICAL   LVALID
      REAL*8    EK
      REAL*8    EGEOM
      REAL*8    EQRULE
      LOGICAL   LFAIL

C Choice of discrete forms required
      LOGICAL   LLK
      LOGICAL   LMK
      LOGICAL   LMKT
      LOGICAL   LNK

C Discrete Helmholtz integral operators
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

C External functions to be supplied
      COMPLEX*16 FNEXP
      REAL*8     FNSQRT

C External functions provided
      COMPLEX*16 ICMULT
      REAL*8     AREA
      REAL*8     SIZE3
      REAL*8     SSIZE3
      REAL*8     DOT3

C Variables that remain constant throughout subroutine, once initialised
C  Constants
      REAL*8     ZERO,ONE,TWO,THREE,FOUR
      REAL*8     THIRD,PI,FOURPI
      COMPLEX*16 CZERO,CONE,CIMAG
C  H3LC.ERR open status and unit number
      LOGICAL    LOPEN
      INTEGER    IU
C  Useful geometric values
      REAL*8     QBMQA(3),QCMQA(3),QCMQB(3),PMQA(3),PMQB(3),PMQC(3)
      REAL*8     DNPNQ,Q(3),QAREA,NORMQ(3),PX,PY
C  Parameters for triangle proportions - for error,warning
      REAL*8     TPROPE,TPROPW
C  Magnitude (one norm) of geometric values
      REAL*8     MAGVP,MAGQA,MAGQB,MAGQC
      REAL*8     MAGBA,MAGCA,MAGCB,MAGPA,MAGPB,MAGPC
C  Control values relating to required solutions     
      LOGICAL    LLORN,LMORN,LMTORN,LMSORN,LMORMT,LMKSV,LMKTSV
C  Values that classify K as 'zero','real','imaginary', or 'complex'
      LOGICAL    KCOMP,KREAL,KIMAG,KZERO
C  Useful constants related to K
      REAL*8     REK,IMK,RESKO2
      COMPLEX*16 SK,SKO2

C Variables used in the validation of the input data
C  Error variables
      LOGICAL    LERROR,LOCERR
C  Temporary variables
      REAL*8     PA(3),QX,QY
      REAL*8     TEMP,TEMP1,TEMP2,TEMP3,TEMP4,TEMP5
      REAL*8     CROSS(3),ACROSS
      REAL*8     SLBA,SLCA,DPBCA,DPBPA,DPCPA,DET,OODET
      REAL*8     A11,A12,A21,A22,B1,B2
      REAL*8     SAVAL,SBVAL,SCVAL,AVAL,BVAL,CVAL,RVAL
      REAL*8     TOP,TOPA,TOPB,TOPC,BOT,BOTA,BOTB,BOTC,COSA,COSB,COSC
      REAL*8     LAMBDA
      REAL*8     CR1,CR2,CR3,SCR,ACR,DOTVCR
      LOGICAL    LTEMP

C  Corresponding error bounds to the temporary variables
      REAL*8     EPA1,EPA2,EPA3
      REAL*8     ETEMP,ETEMP1,ETEMP2,ETEMP3,ETEMP4,ETEMP5
      REAL*8     EPX,EPY,EQX,EQY
      REAL*8     ESLBA,ESLCA,EDPBCA,EDPBPA,EDPCPA,EDET,EOODET
      REAL*8     EA11,EA12,EA21,EA22,EB1,EB2,EMAG
      REAL*8     ESAVAL,ESBVAL,ESCVAL,ERVAL
      REAL*8     ETOP,EBOT,ECOSA,ECOSB,ECOSC
      REAL*8     ELAMB
      REAL*8     ECR1,ECR2,ECR3,ESCR,ECR,EDVCR

C  Variables used when P lies on the element QA-QB-QC
C   Index of triangle for the evaluation of the polar integral
      INTEGER    II
      REAL*8     AR0(3),ARA(3),AOPP(3)
      REAL*8     RQAQB,RQBQC,RQCQA,RQAP,RQBP,RQCP,A,B
      REAL*8     OPP,R0,RA,SOPP,SR0,SRA

C Variables used in the operation of the quadrature rule
C  Variables for the accumulation of integrals
      COMPLEX*16 CSUML,CSUMM,CSUMMT,CSUMN
C  Index of quadrature point
      INTEGER    IQ,JQ
C  Geometric variables
      REAL*8     RR(3),SR,R,CR,RNQ,RNP,RNPRNQ,RNPNQ
C  Green's functions for the Laplace operator and its r-derivatives
      REAL*8     FPG0,FPG0R,FPG0RR
C  Geometric variables related to K
      REAL*8     REKR,IMKR
      COMPLEX*16 KR,IKR,SKR
C  Variable for the storage of the value of the exponential function
      COMPLEX*16     E
C  Green' functions for the Helmholtz operator and its r-derivatives
      COMPLEX*16 FPG,FPGR,FPGRR
C  Other variables 
      COMPLEX*16 EOSR,WFPGR
      REAL*8     QAO4PI

C INITIALISATION

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      THREE=3.0D0
      FOUR=4.0D0
      THIRD=1.0D0/3.0D0
      PI=3.1415926535897932
      FOURPI=FOUR*PI
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

C Initialise useful geometric information
      CALL SUBV3(QB,QA,QBMQA)
      CALL SUBV3(QC,QA,QCMQA)
      CALL SUBV3(QC,QB,QCMQB)
      CALL SUBV3(P,QA,PMQA)
      CALL SUBV3(P,QB,PMQB)
      CALL SUBV3(P,QC,PMQC)
      QAREA=AREA(QA,QB,QC)


C Set LFAIL,LERROR to FALSE
      LFAIL=.FALSE.
      LERROR=.FALSE.


C VALIDATION OF INPUT

C Check that the file H3LC.ERR has been opened, if not then exit
      IF (LVALID) THEN
        INQUIRE(FILE='H3LC.ERR',OPENED=LOPEN)
        IF (.NOT.LOPEN) THEN
          WRITE(*,*) 'ERROR(H3LC) - File for error messages "H3LC.ERR"'
          WRITE(*,*) ' is not open'
          WRITE(*,*) 'Execution of H3LC is aborted'
          LERROR=.TRUE.
          GOTO 998
        END IF
      END IF

C Find out the unit number of the file H3LC.ERR
      INQUIRE(FILE='H3LC.ERR',NUMBER=IU)

C Set useful constants
      IF (LMKT.OR.LNK) MAGVP=ABS(VECP(1))+ABS(VECP(2))+ABS(VECP(3))
      MAGQA=ABS(QA(1))+ABS(QA(2))+ABS(QA(3))
      MAGQB=ABS(QB(1))+ABS(QB(2))+ABS(QB(3))
      MAGQC=ABS(QC(1))+ABS(QC(2))+ABS(QC(3))
      MAGBA=ABS(QBMQA(1))+ABS(QBMQA(2))+ABS(QBMQA(3))
      MAGCA=ABS(QCMQA(1))+ABS(QCMQA(2))+ABS(QCMQA(3))
      MAGCB=ABS(QCMQB(1))+ABS(QCMQB(2))+ABS(QCMQB(3))
      MAGPA=ABS(PMQA(1))+ABS(PMQA(2))+ABS(PMQA(3))
      MAGPB=ABS(PMQB(1))+ABS(PMQB(2))+ABS(PMQB(3))
      MAGPC=ABS(PMQC(1))+ABS(PMQC(2))+ABS(PMQC(3))
      TPROPE=100.0D0
      TPROPW=10.0D0

C If no validation then bypass the validation process
      IF (.NOT.LVALID) GOTO 950


C Check that the function FNSQRT is satisfactory
      IF (LVALID) THEN
        LERROR=.FALSE.
        IF (ABS(FNSQRT(1.0D0)-1.0D0).GT.0.01) LERROR=.TRUE.
        IF (ABS(FNSQRT(100.0D0)-10.0D0).GT.0.1) LERROR=.TRUE.
        IF (ABS(FNSQRT(0.01D0)-0.1).GT.0.001) LERROR=.TRUE.
        IF (LERROR) THEN
          WRITE(IU,*) 'ERROR(H3LC) - in square root function routine'
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
          WRITE(IU,*) 'ERROR(H3LC) - in exponential function routine'
     *     //' FNSQRT'
          STOP
        END IF
      END IF

        
C Check the tolerances EK,EGEOM,EQRULE is of unit magnitude
      IF (EK.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Parameter EK must be positive'
        LERROR=.TRUE.
      END IF
      IF (EGEOM.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Parameter EGEOM must be positive'
        LERROR=.TRUE.
      END IF
      IF (EQRULE.LE.ZERO) THEN
        WRITE(IU,*) 'ERROR(H3LC) - PARAMETER EQRULE must be positive'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF
      
C Validation of the geometric information
C  Check VECP is of unit magnitude
      IF (LMKT.OR.LNK) THEN
        TEMP=SSIZE3(VECP)
        ETEMP=TWO*EGEOM*MAGVP
        IF (ABS(TEMP-ONE).GT.ETEMP) THEN
          WRITE(IU,*) 'ERROR(H3LC) - Parameter VECP must have'
          WRITE(IU,*) 'unit magnitude'
          LERROR=.TRUE.
        END IF
      END IF
C  Check QA,QB,QC are distinct points
      EMAG=6.0D0*EGEOM
      IF (MIN(MAGBA,MAGCA,MAGCB).LT.EMAG) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Check specification of QA,QB,QC'
        LERROR=.TRUE.
      END IF
C  Check P is distinct from QA,QB and QC
      EMAG=6.0D0*EGEOM
      IF (MIN(MAGPA,MAGPB,MAGPC).LT.EMAG) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Check specification of P,QA,QB,QC'
        LERROR=.TRUE.
      END IF
C Check that QA-QB-QC is well proportioned
      TEMP1=SIZE3(QBMQA)
      TEMP2=SIZE3(QCMQA)
      TEMP3=SIZE3(QCMQB)
      IF (MAX(TEMP1,TEMP2,TEMP3).GT.TPROPE*MIN(TEMP1,TEMP2,TEMP3)) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Element sides are not well'
        WRITE(IU,*) ' proportioned'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF

C Check that the tolerance EGEOM is not too large, if so then exit
      IF (EGEOM.GT.MAX(ABS(QA(1)),ABS(QA(2)),ABS(QA(3)),ABS(QB(1)),
     * ABS(QB(2)),ABS(QB(3)),ABS(QC(1)),ABS(QC(2)),ABS(QC(3)))/100.0D0)
     * THEN
        WRITE(IU,*) 'ERROR(H3LC) - EGEOM is set too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF

C  Check that the angle of the triangle at QA, QB and QC is not zero 
C   and is not 2pi
      SAVAL=SSIZE3(QCMQB)
      SBVAL=SSIZE3(QCMQA)
      SCVAL=SSIZE3(QBMQA)
      ESAVAL=TWO*EGEOM*MAGCB
      ESBVAL=TWO*EGEOM*MAGCA
      ESCVAL=TWO*EGEOM*MAGBA
      AVAL=SQRT(SAVAL)
      BVAL=SQRT(SBVAL)
      CVAL=SQRT(SCVAL)
      TOPA=SBVAL+SCVAL-SAVAL
      TOPB=SAVAL+SCVAL-SBVAL
      TOPC=SAVAL+SBVAL-SCVAL
      ETOP=ESBVAL+ESCVAL+ESAVAL
      BOTA=TWO*BVAL*CVAL
      BOTB=TWO*AVAL*CVAL
      BOTC=TWO*AVAL*BVAL
      COSA=TOPA/BOTA
      ECOSA=ETOP/BOTA+ABS(COSA)*ETOP/BOTA
      COSB=TOPB/BOTB
      ECOSB=ETOP/BOTB+ABS(COSB)*ETOP/BOTB
      COSC=TOPC/BOTC
      ECOSC=ETOP/BOTC+ABS(COSC)*ETOP/BOTC
      IF (ABS(ABS(COSA)-ONE).LT.ECOSA.AND.
     *    ABS(ABS(COSB)-ONE).LT.ECOSB.AND.
     *    ABS(ABS(COSC)-ONE).LT.ECOSC) THEN
        WRITE(IU,*) 'ERROR(HO2LC) - Check specification of element
     * QA-QB-QC'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF


950   CONTINUE

C FURTHER INITIALISATION

C Locate the closest point to P on the element QA-QB-QB.
C  The result PX,PY are the coordinates of the nearest point to P
C  on the standard triangle. EPX,EPY are the error bounds for PX,PY
      SLBA=SSIZE3(QBMQA)
      SLCA=SSIZE3(QCMQA)
      DPBCA=DOT3(QBMQA,QCMQA)
      DPBPA=DOT3(QBMQA,PMQA)
      DPCPA=DOT3(QCMQA,PMQA)
      DET=SLBA*SLCA-DPBCA*DPBCA
      OODET=ONE/DET
      A11=SLCA
      A12=-DPBCA
      A21=-DPBCA
      A22=SLBA
      B1=DPBPA
      B2=DPCPA
      PX=OODET*(A11*B1+A12*B2)
      PY=OODET*(A21*B1+A22*B2)
      IF (LVALID) THEN
        ESLBA=FOUR*EGEOM*MAGBA
        ESLCA=FOUR*EGEOM*MAGCA
        EDPBCA=TWO*EGEOM*(MAGBA+MAGCA)
        EDPBPA=TWO*EGEOM*(MAGBA+MAGPA)
        EDPCPA=TWO*EGEOM*(MAGCA+MAGPA)
        EDET=SLBA*ESLCA+ESLBA*SLCA+TWO*EDPBCA*ABS(DPBCA)
        EOODET=EDET/DET/DET
        EA11=ESLCA
        EA12=EDPBCA
        EA21=EDPBCA
        EA22=ESLBA
        EB1=EDPBPA
        EB2=EDPCPA
        EPX=ABS(OODET)*(ABS(A11)*EB1+ABS(B1)*EA11
     *   +ABS(A12)*EB2+ABS(B2)*EA12)+EOODET*ABS(A11*B1+A12*B2)
        EPY=ABS(OODET)*(ABS(A21)*EB1+ABS(B1)*EA21
     *   +ABS(A22)*EB2+ABS(B2)*EA22)+EOODET*ABS(A21*B1+A22*B2)
        EPA1=EGEOM+EPX*ABS(QBMQA(1))+PX*TWO*EGEOM
     *   +EPY*ABS(QCMQA(1))+PY*TWO*EGEOM
        EPA2=EGEOM+EPX*ABS(QBMQA(2))+PX*TWO*EGEOM
     *   +EPY*ABS(QCMQA(2))+PY*TWO*EGEOM
        EPA3=EGEOM+EPX*ABS(QBMQA(3))+PX*TWO*EGEOM
     *   +EPY*ABS(QCMQA(3))+PY*TWO*EGEOM
        IF (EDET.GT.ABS(DET)) THEN
          WRITE(20,*) 'WARNING(H3LC) - Geometry is difficult to'
          WRITE(20,*) 'validate accurately'
        END IF
      END IF
      IF (LVALID) LTEMP=.FALSE.
      IF (PX.LE.ZERO) THEN
        IF (LVALID.AND.LPONEL.AND.PX.LT.-EPX) LTEMP=.TRUE.
        PX=EGEOM/(MAGQA+MAGQB+MAGQC)
      END IF
      IF (PY.LE.ZERO) THEN
        IF (LVALID.AND.LPONEL.AND.PY.LT.-EPY) LTEMP=.TRUE.
        PY=EGEOM/(MAGQA+MAGQB+MAGQC)
      END IF
      IF (PX+PY.GE.ONE) THEN
        IF (LVALID.AND.LPONEL.AND.PX+PY.GT.ONE+EPX+EPY) LTEMP=.TRUE.
        TEMP=(PX+PY)*(ONE+EGEOM/(MAGQA+MAGQB+MAGQC))
        PX=PX/TEMP
        PY=PY/TEMP
      END IF

C  Set PA: the nearest point to P on QA-QB-QC
      PA(1)=QA(1)+PX*QBMQA(1)+PY*QCMQA(1)
      PA(2)=QA(2)+PX*QBMQA(2)+PY*QCMQA(2)
      PA(3)=QA(3)+PX*QBMQA(3)+PY*QCMQA(3)

      IF (.NOT.LVALID) GOTO 900

C VALIDATION OF INPUT CONTINUED

C Validation of the geometric information continued

C  Check LPONEL

      EPA1=EGEOM+EPX*ABS(QBMQA(1))+PX*EGEOM+
     * EPY*ABS(QCMQA(1))+PY*EGEOM
      EPA2=EGEOM+EPX*ABS(QBMQA(2))+PX*EGEOM+
     * EPY*ABS(QCMQA(2))+PY*EGEOM
      EPA3=EGEOM+EPX*ABS(QBMQA(3))+PX*EGEOM+
     * EPY*ABS(QCMQA(3))+PY*EGEOM
      LTEMP=ABS(PA(1)-P(1)).LT.EPA1+EGEOM.AND.
     *      ABS(PA(2)-P(2)).LT.EPA2+EGEOM.AND.
     *      ABS(PA(3)-P(3)).LT.EPA3+EGEOM
      IF (LPONEL.AND.(.NOT.LTEMP)) THEN
        WRITE(IU,*) 'ERROR(H3LC) - LPONEL should be .FALSE.'
        LERROR=.TRUE.
      END IF
      IF (.NOT.LPONEL.AND.LTEMP.AND.EDET.LT.ABS(DET)) THEN
       WRITE(IU,*) 'ERROR(H3LC) - LPONEL should be .TRUE.'
       LERROR=.TRUE.
      END IF

C Check that P does not lie on the edges QA-QB, QB-QC, QC-QA
      LOCERR=.FALSE.
      TOP=DOT3(QBMQA,PMQA)
      ETOP=EGEOM*(MAGBA+MAGPA)
      BOT=SSIZE3(QBMQA)
      EBOT=2.0D0*EGEOM*MAGBA
      LAMBDA=TOP/BOT
      ELAMB=ETOP/BOT+LAMBDA*EBOT/BOT
      IF (LAMBDA.GT.-ELAMB.AND.LAMBDA.LT.ONE+ELAMB) THEN
        PA(1)=QA(1)+LAMBDA*QBMQA(1)
        PA(2)=QA(2)+LAMBDA*QBMQA(2)
        PA(3)=QA(3)+LAMBDA*QBMQA(3)
        EPA1=EGEOM+ELAMB*ABS(QBMQA(1))+TWO*EGEOM*ABS(LAMBDA)
        EPA2=EGEOM+ELAMB*ABS(QBMQA(2))+TWO*EGEOM*ABS(LAMBDA)
        EPA3=EGEOM+ELAMB*ABS(QBMQA(3))+TWO*EGEOM*ABS(LAMBDA)
        IF (ABS(PA(1)-P(1)).LT.EGEOM+EPA1.AND.
     *   ABS(PA(2)-P(2)).LT.EGEOM+EPA2.AND.
     *   ABS(PA(3)-P(3)).LT.EGEOM+EPA3) LOCERR=.TRUE.
      END IF      
      TOP=DOT3(QCMQA,PMQA)
      ETOP=EGEOM*(MAGCA+MAGPA)
      BOT=SSIZE3(QCMQA)
      EBOT=2.0*EGEOM*MAGCA
      LAMBDA=TOP/BOT
      ELAMB=ETOP/BOT+LAMBDA*EBOT/BOT
      IF (LAMBDA.GT.-ELAMB.AND.LAMBDA.LT.ONE+ELAMB) THEN
        PA(1)=QA(1)+LAMBDA*QCMQA(1)
        PA(2)=QA(2)+LAMBDA*QCMQA(2)
        PA(3)=QA(3)+LAMBDA*QCMQA(3)
        EPA1=EGEOM+ELAMB*ABS(QCMQA(1))+TWO*EGEOM*ABS(LAMBDA)
        EPA2=EGEOM+ELAMB*ABS(QCMQA(2))+TWO*EGEOM*ABS(LAMBDA)
        EPA3=EGEOM+ELAMB*ABS(QCMQA(3))+TWO*EGEOM*ABS(LAMBDA)
        IF (ABS(PA(1)-P(1)).LT.EGEOM+EPA1.AND.
     *   ABS(PA(2)-P(2)).LT.EGEOM+EPA2.AND.
     *   ABS(PA(3)-P(3)).LT.EGEOM+EPA3) LOCERR=.TRUE.
      END IF      
      TOP=DOT3(QCMQB,PMQB)
      ETOP=EGEOM*(MAGCB+MAGPB)
      BOT=SSIZE3(QCMQB)
      EBOT=2.0*EGEOM*MAGCB
      LAMBDA=TOP/BOT
      ELAMB=ETOP/BOT+LAMBDA*EBOT/BOT
      IF (LAMBDA.GT.-ELAMB.AND.LAMBDA.LT.ONE+ELAMB) THEN
        PA(1)=QA(1)+LAMBDA*QCMQB(1)
        PA(2)=QA(2)+LAMBDA*QCMQB(2)
        PA(3)=QA(3)+LAMBDA*QCMQB(3)
        EPA1=EGEOM+ELAMB*ABS(QCMQB(1))+TWO*EGEOM*ABS(LAMBDA)
        EPA2=EGEOM+ELAMB*ABS(QCMQB(2))+TWO*EGEOM*ABS(LAMBDA)
        EPA3=EGEOM+ELAMB*ABS(QCMQB(3))+TWO*EGEOM*ABS(LAMBDA)
        IF (ABS(PA(1)-P(1)).LT.EGEOM+EPA1.AND.
     *   ABS(PA(2)-P(2)).LT.EGEOM+EPA2.AND.
     *   ABS(PA(3)-P(3)).LT.EGEOM+EPA3) LOCERR=.TRUE.
      END IF      
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H3LC) - P must not lie on the edge of the'
        WRITE(IU,*) ' element QA-QB-QC'
        LERROR=.TRUE.
      END IF


C  Check that VECP is THE normal to QA-QB when P is on QA-QB
      IF (LPONEL.AND.(LMKT.OR.LNK)) THEN
        CR1=QBMQA(2)*QCMQA(3)-QBMQA(3)*QCMQA(2)
        CR2=QCMQA(1)*QBMQA(3)-QBMQA(1)*QCMQA(3)
        CR3=QBMQA(1)*QCMQA(2)-QBMQA(2)*QCMQA(1)
        ECR1=TWO*EGEOM*(ABS(QBMQA(2))+ABS(QCMQA(3))+
     *   ABS(QBMQA(3))+ABS(QCMQA(2)))
        ECR2=TWO*EGEOM*(ABS(QCMQA(1))+ABS(QBMQA(3))+
     *   ABS(QBMQA(1))+ABS(QCMQA(3)))
        ECR3=TWO*EGEOM*(ABS(QBMQA(1))+ABS(QCMQA(2))+
     *   ABS(QBMQA(2))+ABS(QCMQA(1)))
        SCR=CR1*CR1+CR2*CR2+CR3*CR3
        ESCR=TWO*(ECR1*ABS(CR1)+ECR2*ABS(CR2)+ECR3*ABS(CR3))
        ACR=SQRT(SCR)
        ECR=ESCR/ACR/TWO
        DOTVCR=VECP(1)*CR1+VECP(2)*CR2+VECP(3)*CR3
        EDVCR=EGEOM*(ABS(CR1)+ABS(CR2)+ABS(CR3))+
     *   EGEOM*(ECR1+ECR2+ECR3)
        IF (ABS(ABS(DOTVCR)-ACR).GT.EDVCR+ECR) THEN
          WRITE(IU,*) 'ERROR(H3LC) - VECP must be normal to QA-QB-QC'
          LERROR=.TRUE.
        END IF
        IF (.NOT.LERROR) THEN
          IF (DOTVCR.LT.ZERO) THEN
            WRITE(IU,*) 'ERROR(H3LC) - Replace VECP by -VECP'
            LERROR=.TRUE.
          END IF
        END IF
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF

C Check that QA-QB-QC is well proportioned
      TEMP1=SIZE3(QBMQA)
      TEMP2=SIZE3(QCMQA)
      TEMP3=SIZE3(QCMQB)
      IF (MAX(TEMP1,TEMP2,TEMP3).GT.TPROPW*MIN(TEMP1,TEMP2,TEMP3)) THEN
        WRITE(IU,*) 'WARNING(H3LC) - Element sides are not'
        WRITE(IU,*) 'well proportioned'
      END IF
        
C Validation of the  quadrature rule data
C  Check MAXNQ,NQ
      IF (MAXNQ.LE.0) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Must have MAXNQ > 0'
        LERROR=.TRUE.
      END IF
      IF (NQ.LE.0.OR.NQ.GT.MAXNQ) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Must have 0< NQ <= MAXNQ'
        LERROR=.TRUE.
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF

C  Check that EQRULE is not too large, if so then exit
      IF (EQRULE.GT.ONE/FLOAT(NQ)/10.0D0) THEN
        WRITE(IU,*) 'EQRULE is set too large'
        LERROR=.TRUE.
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF

C  Check the abscissae XQ,YQ all lie on the standard triangle
      LOCERR=.FALSE.
      DO 100 IQ=1,NQ
        IF (XQ(IQ).LT.-EQRULE.OR.YQ(IQ).LT.-EQRULE.OR.
     *   XQ(IQ)+YQ(IQ).GT.ONE+TWO*EQRULE) LOCERR=.TRUE.
100   CONTINUE
      IF (LOCERR) THEN
        WRITE(IU,*) 'ERROR(H3LC) - Components of XQ-YQ are outside'
        WRITE(IU,*) 'the standard triangle.'
        LERROR=.TRUE.
      END IF

C  Check that a quadrature point does not coincide with P
      IF (LPONEL) THEN
        LOCERR=.FALSE.
        DO 135 JQ=1,NQ
          IF (ABS(XQ(JQ)-PX).LT.EQRULE+EPX
     *   .AND.ABS(YQ(JQ)-PY).LT.EQRULE+EPY) LOCERR=.TRUE.
135     CONTINUE
        IF (LOCERR) THEN
          WRITE(IU,*) 'ERROR(H3LC) - One of the quadrature points'
          WRITE(IU,*) 'coincides with the point P.'
          LERROR=.TRUE.
        END IF
      END IF
C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
        GOTO 998
      END IF


C  Check the integrals of 1 and 1-x-y are computed to within the
C   accuracy dictated by EQRULE. That is that there is no numerical
C   integration error.
      TEMP1=ZERO
      ETEMP1=ZERO
      TEMP2=ZERO
      ETEMP2=ZERO
      TEMP3=ZERO
      ETEMP3=ZERO
      DO 110 IQ=1,NQ
        TEMP1=TEMP1+WQ(IQ)
        ETEMP1=ETEMP1+EQRULE
        TEMP=ONE-XQ(IQ)-YQ(IQ)
        TEMP2=TEMP2+WQ(IQ)*TEMP
        ETEMP2=ETEMP2+ABS(WQ(IQ))*EQRULE*TWO+EQRULE*TEMP
        IF (LPONEL) THEN
          RVAL=SQRT((XQ(IQ)-PX)*(XQ(IQ)-PX)+(YQ(IQ)-PY)*(YQ(IQ)-PY))
          ERVAL=((EPX+EQRULE)*ABS(XQ(IQ)-PX)+
     *     (EPY+EQRULE)*ABS(YQ(IQ)-PY))/RVAL/TWO
          TEMP3=TEMP3+RVAL*WQ(IQ)
          ETEMP3=ETEMP3+ERVAL*ABS(WQ(IQ))+EQRULE*RVAL
        END IF
110   CONTINUE
      IF (ABS(TEMP1-ONE).GT.ETEMP1.OR.
     * ABS(TEMP2-THIRD).GT.ETEMP2) THEN
        WRITE(IU,*) 'ERROR(H3LC) - In XQ-YQ-WQ quadrature rule'
        LERROR=.TRUE.
      END IF
      IF (LPONEL) THEN
        TEMP4=ZERO
        ETEMP4=ZERO
        QX=(PX+ONE)/THREE
        EQX=EPX/THREE
        QY=PY/THREE
        EQY=EPY/THREE
        RVAL=SQRT((QX-PX)*(QX-PX)+(QY-PY)*(QY-PY))
        ERVAL=((EQX+EPX)*ABS(QX-PX)+(EQY+EPY)*ABS(QY-PY))/RVAL/TWO
        TEMP4=TEMP4+RVAL*PX/TWO
        ETEMP4=ETEMP4+(ERVAL*PX+RVAL*EPX)/TWO
        QX=PX/THREE
        EQX=EPX/THREE
        QY=(PY+ONE)/THREE
        EQY=EPY/THREE
        RVAL=SQRT((QX-PX)*(QX-PX)+(QY-PY)*(QY-PY))
        ERVAL=((EQX+EPX)*ABS(QX-PX)+(EQY+EPY)*ABS(QY-PY))/RVAL/TWO
        TEMP4=TEMP4+RVAL*PY/TWO
        ETEMP4=ETEMP4+(ERVAL*PY+RVAL*EPY)/TWO
        QX=(PX+ONE)/THREE
        EQX=EPX/THREE
        QY=(PY+ONE)/THREE
        EQY=EPY/THREE
        RVAL=SQRT((QX-PX)*(QX-PX)+(QY-PY)*(QY-PY))
        ERVAL=((EQX+EPX)*ABS(QX-PX)+(EQY+EPY)*ABS(QY-PY))/RVAL/TWO
        TEMP4=TEMP4+RVAL*(ONE-PX-PY)/TWO
        ETEMP4=ETEMP4+(ERVAL*(ONE-PX+PY)+RVAL*(EPX+EPY))/TWO
        TEMP5=ZERO
        ETEMP5=ZERO
        RVAL=SQRT(PX*PX+PY*PY)
        ERVAL=(EPX*PX+EPY*PY)/RVAL/TWO
        TEMP5=TEMP5+RVAL*(PX+PY)/THREE
        ETEMP5=ETEMP5+(ERVAL*(PX+PY)+RVAL*(EPX+EPY))/THREE
        RVAL=SQRT((ONE-PY)*(ONE-PY)+PX*PX)
        ERVAL=(EPY*(ONE-PY)+EPX*PX)/RVAL/TWO
        TEMP5=TEMP5+RVAL*(ONE-PY)/THREE
        ETEMP5=ETEMP5+(RVAL*EPY+ERVAL*(ONE-PY))/THREE
        RVAL=SQRT((ONE-PX)*(ONE-PX)+PY*PY)
        ERVAL=(EPX*(ONE-PX)+EPY*PY)/RVAL/TWO
        TEMP5=TEMP5+RVAL*(ONE-PX)/THREE
        ETEMP5=ETEMP5+(RVAL*EPX+ERVAL*(ONE-PX))/THREE
        IF (ABS(TEMP3-TEMP5).GT.ABS(TEMP4-TEMP5)*4.0D0+ETEMP3+
     *   ETEMP4+ETEMP5) THEN
          WRITE(IU,*) 'WARNING(H3LC) - XQ-YQ-WQ rule should not'
          WRITE(IU,*) 'assume continuity at P when LPONEL=.TRUE.'
        END IF
      END IF
      IF (LPONEL.AND.(LLK.OR.LNK).AND.
     * ABS(DBLE(K)).GT.EK.AND.ABS(DIMAG(K)).GT.EK) THEN
        DO 120 IQ=1,NQ
          IF (ABS(XQ(IQ)-PX).LT.(EPX+EQRULE).AND.
     *     ABS(YQ(IQ)-PY).LT.(EPY+EQRULE)) THEN
            WRITE(IU,*) 'ERROR(H3LC) - One of the quadrature points'
            WRITE(IU,*) ' coincides with the point P'
            LERROR=.TRUE.
          END IF
120     CONTINUE
      END IF

C  If LERROR then exit H3LC
      IF (LERROR) THEN
        WRITE(IU,*) 'Execution of H3LC is aborted'
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
C  Compute NORMQ, the normalised cross product of QB-QA and QC-QA
C  This is the outward normal when QA,QB,QC are arranged counter-
C   clockwise on the triangle.

      IF (LMORN) THEN
        CROSS(1)=QBMQA(2)*QCMQA(3)-QBMQA(3)*QCMQA(2)
        CROSS(2)=QCMQA(1)*QBMQA(3)-QBMQA(1)*QCMQA(3)
        CROSS(3)=QBMQA(1)*QCMQA(2)-QBMQA(2)*QCMQA(1)
        ACROSS=SIZE3(CROSS)
        NORMQ(1)=CROSS(1)/ACROSS
        NORMQ(2)=CROSS(2)/ACROSS
        NORMQ(3)=CROSS(3)/ACROSS
      END IF

      IF (LNK) DNPNQ=DOT3(VECP,NORMQ)

C Useful constants related to K
      IMK=DIMAG(K)
      REK=DBLE(K)
      SK=K*K
      SKO2=SK/TWO
      RESKO2=DBLE(SKO2)
      KZERO=.FALSE.
      KREAL=.FALSE.
      KIMAG=.FALSE.
      KCOMP=.FALSE.

C Classification of K
      IF (ABS(REK).LT.EK.AND.ABS(IMK).LT.EK) THEN
        KZERO=.TRUE.
      ELSE IF (ABS(IMK).LT.EK) THEN
        KREAL=.TRUE.
      ELSE IF (ABS(REK).LT.EK) THEN
        KIMAG=.TRUE.
      ELSE
        KCOMP=.TRUE.
      END IF

C  Correct PX and PY to ensure that (PX,PY) lies on the standard triangle
      
      IF (PX.LT.ZERO) PX=ZERO
      IF (PY.LT.ZERO) PY=ZERO
      IF (PX+PY.GT.ONE) THEN
       PX=PX/(PX+PY)
       PY=PY/(PX+PY)
      END IF

C Initialise CSUML-CSUMN, the variables that accumulate the integrals
C  Lk,Mk,Mkt,Nk. CSUM* is the Laplace integral* 4 pi/area
      CSUML=ZERO
      CSUMM=ZERO
      CSUMMT=ZERO
      CSUMN=ZERO
      IF (LPONEL.AND.LLORN) THEN
        RQBQC=SIZE3(QCMQB)
        RQCQA=SIZE3(QCMQA)
        RQAQB=SIZE3(QBMQA)
        RQAP=SIZE3(PMQA)
        RQBP=SIZE3(PMQB)
        RQCP=SIZE3(PMQC)
        AR0(1)=RQAP
        AR0(2)=RQBP
        AR0(3)=RQCP
        ARA(1)=RQBP
        ARA(2)=RQCP
        ARA(3)=RQAP
        AOPP(1)=RQAQB
        AOPP(2)=RQBQC
        AOPP(3)=RQCQA
         DO 140 II=1,3
          R0=AR0(II)
          RA=ARA(II)
          OPP=AOPP(II)
          IF (R0.LT.RA) THEN
            TEMP=RA
            RA=R0
            R0=TEMP
          END IF
          SR0=R0*R0
          SRA=RA*RA
          SOPP=OPP*OPP
          A=ACOS((SRA+SR0-SOPP)/TWO/RA/R0)
          B=ATAN(RA*SIN(A)/(R0-RA*COS(A)))
          CSUML=CSUML+CMPLX((R0*SIN(B)*(LOG(TAN((B+A)/TWO))
     *     -LOG(TAN(B/TWO))))/QAREA,ZERO)
          CSUMN=CSUMN+CMPLX((COS(B+A)-COS(B))/R0/SIN(B)/QAREA,ZERO)
140     CONTINUE
      END IF
      CSUMN=CSUMN-SKO2*CSUML

C IF LPONEL AND K=0 THEN SOLUTION IS COMPLETE
      IF (LPONEL.AND.KZERO) GOTO 999

C LOOP THROUGH THE QUADRATURE POINTS AND ACCUMULATE INTEGRALS

      DO 130 IQ=1,NQ
C A: Set Q(1..3), the point on the element.
        Q(1)=QA(1)+XQ(IQ)*QBMQA(1)+YQ(IQ)*QCMQA(1)
        Q(2)=QA(2)+XQ(IQ)*QBMQA(2)+YQ(IQ)*QCMQA(2)
        Q(3)=QA(3)+XQ(IQ)*QBMQA(3)+YQ(IQ)*QCMQA(3)
C B: Set NORMQ(1..3), the unit 'outward' normal to the element at Q(1..3)
C    (already set).
C C: Compute DNPDQ [dot product of VECP and NORMQ] (already set).
C D: Compute RR(1..3) = P(1..3) - Q(1..3)
        CALL SUBV3(P,Q,RR)
C E: Compute R [modulus of RR(1..3)] and SR [R-squared].
        SR=SSIZE3(RR)
        R=FNSQRT(SR)
C F: Compute CR [R-cubed].
        IF (LNK) CR=R*SR
C G: Compute RNQ [derivative of R with respect to NORMQ].
        IF (LMORN) RNQ=-DOT3(RR,NORMQ)/R
C H: Compute RNP [derivative of R with respect to VECP].
        IF (LMTORN) RNP=DOT3(RR,VECP)/R
C I: Compute RNPRNQ [RNP*RNQ].
        IF (LNK) RNPRNQ=RNP*RNQ
C J: Compute RNPNQ [second derivative of R with respect to VECP and NORMQ].
        IF (LNK) RNPNQ=-(DNPNQ+RNPRNQ)/R

C Set values of FPG0,FPG0R,FPG0RR [the Green's funstion and its first
C and second derivatives with respect to R when K=(0,0)]. Only
C necessary when P lies on the element or K=(0,0).
        IF (LPONEL.AND.(.NOT.KZERO)) THEN
          IF (LLORN) FPG0=ONE/R
          IF (LMSORN) FPG0R=-ONE/SR
          IF (LNK) FPG0RR=TWO/CR
        END IF

C Case K is zero.
        IF (KZERO) THEN
C K: Compute KR [K*R] and IKR [i*K*R] (unnecessary)
C L: Compute SKR [K*R*K*R] (unnecessary)
C M: Compute E [exp(IKR)] (unnecessary)
C N: Compute Green's function FPG [1/R] (real)
          IF (LLK) FPG=CMPLX(ONE/R,ZERO)
C O: Multiply Lk kernel by weight and add to sum CSUML (real)
          IF (LLK) CSUML=CSUML+CMPLX(WQ(IQ)*DBLE(FPG),ZERO)
C P: Compute FPGR, derivative of FPG with respect to R
          IF (LMSORN) FPGR=CMPLX(-ONE/SR,ZERO)
C Q: Compute WFPGR, the weight multiplied by FPGR (real)
          IF (LMORMT) WFPGR=CMPLX(WQ(IQ)*DBLE(FPGR),ZERO)
C R: Compute Mk kernel multiplied by weight and add to sum CSUMM (real)
          IF (LMK) CSUMM=CSUMM+CMPLX(DBLE(WFPGR)*RNQ,ZERO)
C S: Compute Mkt kernel multiplied by weight and add to CSUMMT (real).
          IF (LMKT) CSUMMT=CSUMMT+CMPLX(DBLE(WFPGR)*RNP,ZERO)
          IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R (real)
            FPGRR=CMPLX(TWO/CR,ZERO)
C U: Multiply Nk kernel by weight and add to sum CSUMN (real)
            CSUMN=CSUMN+
     *       CMPLX(WQ(IQ)*(DBLE(FPGR)*RNPNQ+DBLE(FPGRR)*RNPRNQ),ZERO)
          END IF

C Case K is real.
        ELSE IF (KREAL) THEN
C K: Compute KR [K*R] (real) and IKR [i*K*R] (imag)
          KR=CMPLX(DBLE(K)*R,ZERO)
          IKR=CMPLX(ZERO,DBLE(KR))
C L: Compute SKR [K*R*K*R] (real)
          SKR=CMPLX(DBLE(KR)*DBLE(KR),ZERO)
C M: Compute E [exp(IKR)].
          E=FNEXP(IKR)
C N: Compute Green's function FPG [E/R].
          IF (LLK) FPG=E/R
C O: Multiply Lk kernel by weight and add to sum SUML.
          IF (LLK) THEN
            IF (.NOT.LPONEL) CSUML=CSUML+WQ(IQ)*FPG
            IF (LPONEL) CSUML=CSUML+WQ(IQ)*(FPG-FPG0)
          END IF
C P: Compute FPGR, derivative of FPG with respect to R.
          IF (LMSORN) THEN
            EOSR=E/SR
            FPGR=ICMULT(EOSR)*DBLE(KR)-EOSR
          END IF
          IF ((.NOT.LPONEL).AND.LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR.
            WFPGR=WQ(IQ)*FPGR
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
            IF (LMK) CSUMM=CSUMM+WFPGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
            IF (LMKT) CSUMMT=CSUMMT+WFPGR*RNP
          END IF
          IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R.
            FPGRR=E*(TWO-IKR-IKR-SKR)/CR
C U: Multiply Nk kernel by weight and add to sum SUMN.
            IF (.NOT.LPONEL) THEN
              CSUMN=CSUMN+WQ(IQ)*(FPGR*RNPNQ+FPGRR*RNPRNQ)
            ELSE
              CSUMN=CSUMN+WQ(IQ)*((FPGR-FPG0R)*RNPNQ+
     *         (FPGRR-FPG0RR)*RNPRNQ+RESKO2*FPG0)
            END IF
          END IF

C Case K is imaginary.
        ELSE IF (KIMAG) THEN
C K: Compute KR [K*R] (imag) and IKR [i*K*R] (real)
          IMKR=IMK*R
          REKR=ZERO
          KR=CMPLX(REKR,IMKR)
          IKR=CMPLX(-IMKR,REKR)
C L: Compute SKR [K*R*K*R] (real)
          SKR=CMPLX(-IMKR*IMKR,ZERO)
C M: Compute E [exp(IKR)] (real)
          E=FNEXP(IKR)
C N: Compute Green's function FPG [E/R] (real)
          IF (LLK) FPG=CMPLX(DBLE(E)/R,ZERO)
C O: Multiply Lk kernel by weight and add to sum CSUML (real)
          IF (LLK) THEN
            IF (.NOT.LPONEL) CSUML=CSUML+CMPLX(WQ(IQ)*DBLE(FPG),ZERO)
            IF (LPONEL) CSUML=CSUML+CMPLX(WQ(IQ)*DBLE(FPG-FPG0),ZERO)
          END IF
C P: Compute FPGR, derivative of FPG with respect to R (real)
          IF (LMSORN) THEN
            EOSR=CMPLX(DBLE(E)/SR,ZERO)
            FPGR=CMPLX(DBLE(EOSR)*DBLE(IKR-ONE),ZERO)
          END IF
          IF ((.NOT.LPONEL).AND.LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR (real)
            WFPGR=CMPLX(WQ(IQ)*DBLE(FPGR),ZERO)
C R: Compute Mk kernel multiplied by weight and add to sum CSUMM (real)
            IF (LMK) CSUMM=CSUMM+CMPLX(DBLE(WFPGR)*RNQ,ZERO)
C S: Compute Mkt kernel multiplied by weight and add to CSUMMT (real)
            IF (LMKT) CSUMMT=CSUMMT+CMPLX(DBLE(WFPGR)*RNP,ZERO)
          END IF
          IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R (real)
            FPGRR=CMPLX(DBLE(E)*DBLE(TWO-IKR-IKR-SKR)/CR,ZERO)
C U: Multiply Nk kernel by weight and add to sum CSUMN (real)
            IF (.NOT.LPONEL) THEN
              CSUMN=CSUMN+CMPLX(WQ(IQ)*(DBLE(FPGR)*RNPNQ
     *         +DBLE(FPGRR)*RNPRNQ),ZERO)
            ELSE
              CSUMN=CSUMN+CMPLX(WQ(IQ)*(DBLE(FPGR-FPG0R)*RNPNQ+
     *         DBLE(FPGRR-FPG0RR)*RNPRNQ+RESKO2*FPG0),ZERO)
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
            IF (.NOT.LPONEL) CSUML=CSUML+WQ(IQ)*FPG
            IF (LPONEL) CSUML=CSUML+WQ(IQ)*(FPG-FPG0)
          END IF
C P: Compute FPGR, derivative of FPG with respect to R.
          IF (LMSORN) THEN
            EOSR=E/SR
            FPGR=ICMULT(EOSR)*KR-EOSR
          END IF
          IF ((.NOT.LPONEL).AND.LMORMT) THEN
C Q: Compute WFPGR, the weight multiplied by FPGR.
            WFPGR=WQ(IQ)*FPGR
C R: Compute Mk kernel multiplied by weight and add to sum SUMM.
            IF (LMK) CSUMM=CSUMM+WFPGR*RNQ
C S: Compute Mkt kernel multiplied by weight and add to sum SUMMT.
            IF (LMKT) CSUMMT=CSUMMT+WFPGR*RNP
          END IF
          IF (LNK) THEN
C T: Compute GRR, the second derivative of G with respect to R.
            FPGRR=E*(TWO-IKR-IKR-SKR)/CR
C U: Multiply Nk kernel by weight and add to sum SUMN.
            IF (.NOT.LPONEL) THEN
              CSUMN=CSUMN+WQ(IQ)*(FPGR*RNPNQ+FPGRR*RNPRNQ)
            ELSE
              CSUMN=CSUMN+WQ(IQ)*((FPGR-FPG0R)*RNPNQ+
     *         (FPGRR-FPG0RR)*RNPRNQ+SKO2*FPG0)
            END IF
          END IF
        END IF

130   CONTINUE

999   CONTINUE

      QAO4PI=QAREA/FOURPI

      DISLK=QAO4PI*CSUML
      DISMK=QAO4PI*CSUMM
      DISMKT=QAO4PI*CSUMMT
      DISNK=QAO4PI*CSUMN

      IF (LPONEL) THEN
        LMK=LMKSV
        LMKT=LMKTSV
      END IF

998   CONTINUE


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
