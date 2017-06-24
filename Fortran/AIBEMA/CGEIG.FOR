C***************************************************************
C           Subroutine CGEIG by Stephen Kirkup                   
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/CGEIG.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C

C This subroutine returns the eigenvalues and eigenvectors of the
C  generalised eigenvalue problem Ax=lBx where the ls represent
C  the eigenvalues, the xs are the eigenvectors of the given
C  matrices A and B. The matrices A and B are of size NNxNN.

C This is an example routine. This particular routine uses NAG routine 
C  F02AKF. Users may need to replace this with an alternative routine.


      SUBROUTINE CGEIG(MAXNN,NN,A,B,EIGVAL,EIGVEC,
     * WKSP02,WKSP03,WKSP04,WKSP05,WKSP06,WKSP07,WKSP08,WKSP10)
      INTEGER MAXNN,NN
      COMPLEX*16 A(MAXNN,MAXNN)
      COMPLEX*16 B(MAXNN,MAXNN)
      COMPLEX*16 EIGVAL(MAXNN)
      COMPLEX*16 EIGVEC(MAXNN,MAXNN)
      COMPLEX*16 WKSP02(MAXNN,MAXNN)
      REAL*8     WKSP03(MAXNN,MAXNN)
      REAL*8     WKSP04(MAXNN,MAXNN)
      REAL*8     WKSP05(MAXNN,MAXNN)
      REAL*8     WKSP06(MAXNN,MAXNN)
      REAL*8     WKSP07(MAXNN)
      REAL*8     WKSP08(MAXNN)
      INTEGER    WKSP10(MAXNN)

      REAL*8     SSIZE
      INTEGER    I,J

C Invert the matrix A, store in A

      CALL MATINV(MAXNN,NN,A)


C Compute the matrix A^(-1) B, store in WKSP02

      DO 10 I=1,NN
        DO 20 J=1,NN
          WKSP02(I,J)=0.0D0
          DO 30 L=1,NN
            WKSP02(I,J)=WKSP02(I,J)+A(I,L)*B(L,J)
30        CONTINUE
20      CONTINUE
10    CONTINUE

 
C Split the matrix into real and imaginary parts - in WKSP03 and
C  WKSP04.
         
      DO 50 I=1,NN
        DO 60 J=1,NN
          WKSP03(I,J)=REAL(WKSP02(I,J))
          WKSP04(I,J)=AIMAG(WKSP02(I,J))
60      CONTINUE
50    CONTINUE

      IFAIL=0

C Call the NAG routine. The real part of the eigenvalues are stored
C in WKSP07, the imaginary parts in WKSP08. The real part of the
C eigenvectors is stored in WKSP05 and the imaginary parts in WKSP06.

      CALL F02AKF(WKSP03,MAXNN,WKSP04,MAXNN,NN,
     * WKSP07,WKSP08,WKSP05,MAXNN,WKSP06,MAXNN,WKSP10,IFAIL)

C Since we have the eigenvalues of A^(-1) B then the results are
C the reciprocals of the eigenvalues of A x = {\lambda} B x.
C The eigenvalues and eigenvectors are formed into type complex
C from their real and imaginary parts returned by F02AKF. 

      DO 100 I=1,NN
        SSIZE=WKSP07(I)*WKSP07(I)+WKSP08(I)*WKSP08(I)
        IF (SSIZE.GT.0.0) THEN
          EIGVAL(I)=CMPLX(WKSP07(I)/SSIZE,-WKSP08(I)/SSIZE)
        ELSE
          EIGVAL(I)=1.0D10
        END IF
        DO 110 J=1,NN
          EIGVEC(I,J)=CMPLX(WKSP05(J,I),WKSP06(J,I))
110     CONTINUE
100   CONTINUE

      END


C Module MATINV returns the inverse of the matrix A in A
C MAXNN is the size of the array, NN the size of the matrix.

      SUBROUTINE MATINV(MAXNN,NN,A)
      INTEGER MAXNN,NN
      COMPLEX*16 A(MAXNN,MAXNN)
      COMPLEX*16 SUM
      INTEGER    LOLIM,I,J,K
      CALL MATLU(MAXNN,NN,A)
      DO 30 K=NN,2,-1
        DO 20 J=K-1,1,-1
          SUM=-A(J,K)
          DO 10 I=J+1,K-1
            SUM=SUM-A(J,I)*A(I,K)
10        CONTINUE
          A(J,K)=SUM
20      CONTINUE
30    CONTINUE
      DO 50 K=1,NN-1
        A(K,K)=1.0/A(K,K)
        DO 50 J=K+1,NN
          SUM=0.0D0
          DO 40 I=K,J-1
            SUM=SUM-A(J,I)*A(I,K)
40        CONTINUE
          A(J,K)=SUM/A(J,J)
50      CONTINUE
        A(NN,NN)=1.0/A(NN,NN)
        DO 80 K=1,NN
          DO 70 J=1,NN-1
          IF (J.GE.K) THEN
            LOLIM=J+1
            SUM=A(J,K)
          ELSE
            LOLIM=K
            SUM=0.0D0
          END IF
          DO 60 I=LOLIM,NN
            SUM=SUM+A(J,I)*A(I,K)
60        CONTINUE
          A(J,K)=SUM
70      CONTINUE
80    CONTINUE
      END


C Module MATLU returns the LU factorisation of the matrix A in A
C MAXNN is the size of the array, NN the size of the matrix.

      SUBROUTINE MATLU(MAXNN,NN,A)
      INTEGER  MAXNN,NN
      COMPLEX*16 A(MAXNN,MAXNN),PIV,SUM
      INTEGER    I,J,K
      PIV=A(1,1)
      DO 10 J=2,NN
        A(1,J)=A(1,J)/PIV
10    CONTINUE
      DO 60 K=2,NN
        DO 30 J=K,NN
          SUM=A(J,K)
          DO 20 I=1,K-1
            SUM=SUM-A(J,I)*A(I,K)
20        CONTINUE
30        A(J,K)=SUM
        PIV=A(K,K)
        DO 50 J=K+1,NN
          SUM=A(K,J)
          DO 40 I=1,K-1
            SUM=SUM-A(K,I)*A(I,J)
40        CONTINUE
          A(K,J)=SUM/PIV
50      CONTINUE
60    CONTINUE
      END

