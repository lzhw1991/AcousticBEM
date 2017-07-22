C***************************************************************
C              Subroutine CGLS by Stephen Kirkup                  
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/CGLS.FOR
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
C Subroutine CGLS returns the solution x,y to a problem of the form
C     
C                           A x = B y + c
C
C where A and B are n by n complex matrices and c is an n-vector under 
C the condition(s)
C                         
C             {\alpha}_i x_i + {\beta}_i y_i = f_i   for i=1..n.
C
C Clearly only one of {\alpha}_i or {\beta}_i can be zero for each i.
C
C The method employed involves forming a linear system of the form Cz=d 
C where the n by n matrix C and the vector d can be determined from A,B
C and the {\alpha}_i and {\beta}_i. A standard LU factorisation 
C solution method is then employed to return z. From z the actual
C solutions x,y can be determined.


      SUBROUTINE CGLS(MAXN,N,A,B,C,ALPHA,BETA,F,X,Y,LFAIL,
     * WKSPC1,WKSPC2)

C Input parameters
C ----------------
C The limit on the dimension of the matrices A and B 
      INTEGER    MAXN
C The dimension of the matrices
      INTEGER    N
C The matrix A
      COMPLEX*16 A(MAXN,MAXN)
C The matrix B
      COMPLEX*16 B(MAXN,MAXN)
C The vector c
      COMPLEX*16 C(MAXN)
C The {\alpha}_i
      COMPLEX*16 ALPHA(MAXN)
C The {\beta}_i
      COMPLEX*16 BETA(MAXN)
C The f_i
      COMPLEX*16 F(MAXN)

C Output parameters
C -----------------
C The solution vector x
      COMPLEX*16 X(MAXN)
C The solution vector y
      COMPLEX*16 Y(MAXN)

C Work space
C ----------
      COMPLEX*16 WKSPC1(MAXN)
      LOGICAL    WKSPC2(MAXN)


C Local variables
C ---------------
      COMPLEX*16 CTEMP
      LOGICAL LFAIL,LERROR,SPEC
      REAL*8 ANORM,BNORM,ASUM,BSUM
      REAL*8 EPS

C Initialisation
      EPS=1.0D-20
      LFAIL=.FALSE.

C Validation
      IF (MAXN.LT.1.OR.N.LT.1.OR.N.GT.MAXN) THEN
        WRITE(*,*) 'ERROR(CGLS) - Check N,MAXN parameters'
        LFAIL=.TRUE.
      END IF

      LERROR=.FALSE.
      DO 10 I=1,N
        IF (ABS(ALPHA(I)).LT.EPS.AND.ABS(BETA(I)).LT.EPS) LERROR=.TRUE.
10    CONTINUE
      IF (LERROR) THEN
        WRITE(*,*) 'ERROR(CGLS) - ALPHA(i) and BETA(i) must not both'
        WRITE(*,*) ' be zero for any value of i in 1..N'
        LFAIL=.TRUE.
      END IF
  
      IF (LFAIL) GOTO 998

C Special Case: B is a diagonal matrix and BETA(i)=0 for i=1..N
      SPEC=.TRUE.
      DO 720 I=1,N
        IF (SPEC) THEN
          IF (ABS(BETA(I)).GT.EPS) SPEC=.FALSE.
          DO 730 J=1,N
            IF (I.NE.J) THEN
              IF (ABS(B(I,J)).GT.EPS) SPEC=.FALSE.
            END IF
730       CONTINUE
        END IF
720   CONTINUE
      
      IF (.NOT.SPEC) GOTO 990

C Since B is now diagonal, check that all the diagonals are non-zero
      LERROR=.FALSE.
      DO 740 I=1,N
        IF (ABS(B(I,I)).LT.EPS) LERROR=.TRUE.
740   CONTINUE
      IF (LERROR) THEN 
        WRITE(*,*) 'ERROR(CGLS) - No unique solution'
        LFAIL=.TRUE.
        GOTO 998
      END IF
      DO 760 I=1,N
        Y(I)=-C(I)
        DO 770 J=1,N
          Y(I)=Y(I)+A(I,J)*F(J)/ALPHA(J)
770     CONTINUE
        Y(I)=Y(I)/B(I,I)
        X(I)=F(I)/ALPHA(I)
760   CONTINUE
      GOTO 998

990   CONTINUE

C Special Case: A is a diagonal matrix and ALPHA(i)=0 for i=1..N
      SPEC=.TRUE.
      DO 820 I=1,N
        IF (SPEC) THEN
          IF (ABS(ALPHA(I)).GT.EPS) SPEC=.FALSE.
          DO 830 J=1,N
            IF (I.NE.J) THEN
              IF (ABS(A(I,J)).GT.EPS) SPEC=.FALSE.
            END IF
830       CONTINUE
        END IF
820    CONTINUE
      
      IF (.NOT.SPEC) GOTO 980

C Since A is now diagonal, check that all the diagonals are non-zero
      LERROR=.FALSE.
      DO 840 I=1,N
        IF (ABS(A(I,I)).LT.EPS) LERROR=.TRUE.
840   CONTINUE
      IF (LERROR) THEN 
        WRITE(*,*) 'ERROR(CGLS) - No unique solution'
        LFAIL=.TRUE.
        GOTO 998
      END IF
      DO 860 I=1,N
        X(I)=C(I)
        DO 870 J=1,N
          X(I)=X(I)+B(I,J)*F(J)/BETA(J)
870     CONTINUE
        X(I)=X(I)/A(I,I)
        Y(I)=F(I)/BETA(I)
860   CONTINUE
      GOTO 998

980   CONTINUE

C Compute the 1-norms of the matrices A and B
      ANORM=0.0D0
      BNORM=0.0D0
      DO 100 I=1,N
        ASUM=0.0D0
        BSUM=0.0D0
        DO 110 J=1,N
          ASUM=ASUM+ABS(A(I,J))
          BSUM=BSUM+ABS(B(I,J))
110     CONTINUE
        IF (ASUM.GT.ANORM) ANORM=ASUM
        IF (BSUM.GT.BNORM) BNORM=BSUM
100   CONTINUE

C Validation
      IF (ANORM.LT.EPS.OR.BNORM.LT.EPS) THEN
        WRITE(*,*) 'ERROR(CGLS) - One of the matrices A,B is zero'
        LFAIL=.TRUE.
      END IF
      IF (LFAIL) GOTO 998

      GAMMA=BNORM/ANORM
      DO 200 J=1,N
        IF (ABS(BETA(J)).LT.GAMMA*ABS(ALPHA(J))) THEN
          WKSPC2(J)=.FALSE.
        ELSE
          WKSPC2(J)=.TRUE.
        END IF
200   CONTINUE
      

      DO 230 I=1,N
        IF (WKSPC2(I)) THEN
          DO 240 J=1,N
            C(J)=C(J)+F(I)*B(J,I)/BETA(I)
            B(J,I)=-ALPHA(I)*B(J,I)/BETA(I)
240       CONTINUE
        ELSE
          DO 250 J=1,N
            C(J)=C(J)-F(I)*A(J,I)/ALPHA(I)
            A(J,I)=-BETA(I)*A(J,I)/ALPHA(I)
250       CONTINUE
        ENDIF
230   CONTINUE


      DO 310 I=1,N
        DO 320 J=1,N
          A(I,J)=A(I,J)-B(I,J)
320     CONTINUE
310   CONTINUE


      CALL CLINSL(MAXN,N,A,C,Y)
      IF (LFAIL) GOTO 998

      DO 510 I=1,N
        IF (WKSPC2(I)) THEN
          X(I)=(F(I)-ALPHA(I)*Y(I))/BETA(I)
        ELSE
          X(I)=(F(I)-BETA(I)*Y(I))/ALPHA(I)
        END IF
510   CONTINUE


      DO 600 I=1,N
        IF (WKSPC2(I)) THEN
          CTEMP=X(I)
          X(I)=Y(I)
          Y(I)=CTEMP
        END IF
600   CONTINUE

998   CONTINUE

      END


C **********************************************************************
C                 Subroutine CLINSL by Stephen Kirkup                  *
C **********************************************************************
C
C Subroutine CLINSL returns the solution x to a problem of the form
C     
C                             A x = b
C
C where A is an n by n complex matrix and b is a given vector 
C                         
C A standard LU factorisation solution method is then employed to return
C the solution. 
C Warning - the matrix A is altered by the procedure.


        SUBROUTINE CLINSL(MAXN,N,A,B,X)

C Input parameters
C ----------------
C The limit on the dimension of the matrices A and B 
      INTEGER    MAXN
C The dimension of the matrices
      INTEGER    N
C The matrix A
      COMPLEX*16 A(MAXN,MAXN)
C The vector B
      COMPLEX*16 B(MAXN)

C Output parameters
C -----------------
C The solution vector x
      COMPLEX*16 X(MAXN)

C Local variables
      COMPLEX*16 TEMP,CSUM,RATIO
      REAL*8     COLMAX,AA
      INTEGER    I,J,II,IIMAX
      IF (MAXN.LT.N) THEN
        WRITE(6,*) 'ERROR(CLINSL) - Must have MAXN>=N'
        STOP        
      ENDIF
      DO 10 I=1,N-1
        COLMAX=0
        DO 20 II=I,N
          AA=ABS(A(II,I))
          IF (AA.GT.COLMAX) THEN
            COLMAX=AA
            IIMAX=II
          ENDIF
20      CONTINUE
        DO 30 J=I,N
          TEMP=A(I,J)
          A(I,J)=A(IIMAX,J)
          A(IIMAX,J)=TEMP
30      CONTINUE
        TEMP=B(I)
        B(I)=B(IIMAX)
        B(IIMAX)=TEMP
        DO 40 II=I+1,N
          RATIO=A(II,I)/A(I,I)
          DO 50 J=I+1,N
            A(II,J)=A(II,J)-RATIO*A(I,J)
50        CONTINUE
          B(II)=B(II)-RATIO*B(I)
40      CONTINUE
10    CONTINUE
      X(N)=B(N)/A(N,N)
      DO 60 I=N-1,1,-1
        CSUM=B(I)
        DO 70 J=I+1,N
          CSUM=CSUM-A(I,J)*X(J)
70      CONTINUE
        X(I)=CSUM/A(I,I)
60    CONTINUE
      END
