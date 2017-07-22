C***************************************************************
C              Subroutine INTEIG by Stephen Kirkup                  
C***************************************************************
C
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/INTEIG.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This subroutine computes the eigenvalues and eigenvectors of the
C matrix A(x) which interpolates the matrices A[1], A[2], ... A[m]
C with respect to the points x[1], x[2], ..., x[m] using an m-1 degree
C polynomial. 
C The solution consists of the values of x and the corresponding
C eigenvector u such that
C
C                            A(x) u = 0 .
C
C The polynomial eigenvalue problem is solved through writing it as a
C generalised linear eigenvalue problem. The subroutine calls the
C routine CGEIG to solve the generalised eigenvalue problem. 


      SUBROUTINE INTEIG(MAXN,N,MAXNPT,NPT,X,MATX,NEIG,EIGVAL,EIGVCT,
     * WKSP00,WKSP01,WKSP02,WKSP03,WKSP04,WKSP05,WKSP06,WKSP07,
     * WKSP08,WKSP09,WKSP10,WKSP11,WKSP12)

C Input parameters
C ----------------
C The limit on the dimension of the matrices
      INTEGER    MAXN
C The dimension of the matrices
      INTEGER    N
C The limit on the number of matrix polynomial interpolation points
      INTEGER    MAXNPT
C The number of matrix polynomial interpolation points
      INTEGER    NPT
C The x-coordiate of the interpolation points x[j]
      REAL*8 X(MAXNPT)
C The value of the matrix components at each interpolation point
C  MATX(j,i,l)= the il component of A[j]. MATX is altered by the module
      COMPLEX*16 MATX(MAXNPT,MAXN,MAXN)


C Output parameters
C -----------------
C The list of eigenvalues
      COMPLEX*16 EIGVAL((MAXNPT-1)*MAXN)
C The list of eigenvectors. EIGVCT(i,j) gives the value of the 
C  j-th component of the i-th eigenvector. The eigenvectors are
C  normalised
      COMPLEX*16 EIGVCT((MAXNPT-1)*MAXN,MAXN)

C Work space
C ----------
      COMPLEX*16 WKSP00((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      COMPLEX*16 WKSP01((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      COMPLEX*16 WKSP02((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      REAL*8     WKSP03((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      REAL*8     WKSP04((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      REAL*8     WKSP05((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      REAL*8     WKSP06((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      REAL*8     WKSP07((MAXNPT-1)*MAXN)
      REAL*8     WKSP08((MAXNPT-1)*MAXN)
      REAL*8     WKSP09(MAXNPT)
      INTEGER    WKSP10((MAXNPT-1)*MAXN)
      COMPLEX*16 WKSP11((MAXNPT-1)*MAXN,(MAXNPT-1)*MAXN)
      COMPLEX*16 WKSP12((MAXNPT-1)*MAXN)
      

C Other variables
C ---------------
      REAL*8     SUMX,AVX,WIDX
      COMPLEX*16 EIG
      INTEGER   NEIG,IEIG,MAXNN,NDG,NN,I,J,IP,JP


C Set constants

      MAXNN=(MAXNPT-1)*MAXN
      NDG=NPT-1
      NN=NDG*N
      IF (NN.GT.MAXNN) STOP


C Transform the x-coordinates onto a sensible range -- [-1/2,1/2]

      SUMX=0.0D0
      DO 10 I=1,NPT
        SUMX=SUMX+X(I)
10    CONTINUE
      AVX=SUMX/DBLE(NPT)
      WIDX=X(NPT)-X(1)
      DO 20 I=1,NPT
        WKSP09(I)=(X(I)-AVX)/WIDX
20    CONTINUE


C Use Newton's divided difference method to find interpolant 
C  coefficients: Store in MATX.

      DO 30 I=1,N
        DO 40 J=1,N
          DO 50 IP=2,NPT
            DO 60 JP=NPT,IP,-1
              MATX(JP,I,J)=(MATX(JP,I,J)-MATX(JP-1,I,J))/
     *         (WKSP09(JP)-WKSP09(JP-IP+1))
60          CONTINUE
50        CONTINUE
40      CONTINUE
30    CONTINUE


C Obtain the coefficient matrices of 1,x,x^2 etc and store in MATX

      DO 100 I=1,N
        DO 110 J=1,N
          DO 120 IP=NDG,1,-1
            DO 130 JP=IP,NDG
              MATX(JP,I,J)=MATX(JP,I,J)-WKSP09(IP)*MATX(JP+1,I,J)
130         CONTINUE
120       CONTINUE
110     CONTINUE
100   CONTINUE


C Set up the matrices "A x = {\lambda} B x". "A" is stored in WKSP00
C "B" is stored in WKSP01

      DO 200 I=1,NDG*N
        DO 210 J=1,NDG*N
          WKSP00(I,J)=0.0D0
          WKSP01(I,J)=0.0D0
210     CONTINUE
200   CONTINUE
      DO 220 I=1,N
        DO 230 J=1,N
          DO 240 IP=1,NDG
            WKSP00(I,N*(IP-1)+J)=-MATX(IP,I,J)
240       CONTINUE
          WKSP01(I,(NDG-1)*N+J)=MATX(NDG+1,I,J)
230     CONTINUE
        DO 250 IP=1,NDG-1
          WKSP01(IP*N+I,(IP-1)*N+I)=1.0D0
          WKSP00(IP*N+I,IP*N+I)=1.0D0
250     CONTINUE
220   CONTINUE


C Call the routine for solving the generalised eigenvalue problem
C The eigenvalues are returned in WKSP12, the corresponding
C eigenvectors in WKSPC11.

      CALL CGEIG((MAXNPT-1)*MAXN,(NPT-1)*N,WKSP00,WKSP01,WKSP12,
     * WKSP11,WKSP02,WKSP03,WKSP04,WKSP05,WKSP06,WKSP07,WKSP08,WKSP10)

      IEIG=0
      DO 300 I=1,NDG*N
        IF (ABS(WKSP12(I)).GT.0.0D0) THEN
            EIG=WKSP12(I)
            IF (DBLE(EIG).GT.-0.6D0.AND.DBLE(EIG).LT.0.6D0) THEN
              EIG=WIDX*EIG+AVX
              IF (ABS(AIMAG(EIG)).LT.ABS(DBLE(EIG))/10.0D0) THEN
              IEIG=IEIG+1
              EIGVAL(IEIG)=EIG
              DO 310 J=1,N
                EIGVCT(IEIG,J)=WKSP11(IEIG,J)
310           CONTINUE
            END IF
          END IF
        END IF
300   CONTINUE
      NEIG=IEIG

      END
