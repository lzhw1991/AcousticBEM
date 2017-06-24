C***************************************************************
C                       Hankel Function 1                              
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/FNHANK.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C

C Subroutine FNHANK: This computes Hankel functions of the first kind
C  and of order zero and one.

C Input parameter
C complex Z    : The argument of the Hankel function (Im(Z)=0).

C Output parameter
C H(0:1)       : The Hankel functions of order zero and one respectively.

C External routines
C S17AEF: From the NAG library.
C S17AFF: From the NAG library.
C S17ACF: From the NAG library.
C S17ADF: From the NAG library.

      SUBROUTINE FNHANK(Z,H)
      use fgsl
      COMPLEX*16 Z,J0,J1,Y0,Y1,H(0:1)
      REAL*8 X,REH0,REH1,IMH0,IMH1
      IF (ABS(DIMAG(Z)).GT.1.0D-6) THEN
         WRITE(*,*) 'ERROR(FNHANK) - Im(Z) must be zero'
         STOP
      END IF
      IFAIL=0
      X=DREAL(Z)
      J0 = fgsl_sf_bessel_jc0(X)
      J1 = fgsl_sf_bessel_jc1(X)
      Y0 = fgsl_sf_bessel_yc0(X)
      Y1 = fgsl_sf_bessel_yc1(X)
C     J0=S17AEF(X,IFAIL)
C     J1=S17AFF(X,IFAIL)
C     Y0=S17ACF(X,IFAIL)
C     Y1=S17ADF(X,IFAIL)
      REH0=DREAL(J0)-DIMAG(Y0)
      IMH0=DIMAG(J0)+DREAL(Y0)
      REH1=DREAL(J1)-DIMAG(Y1)
      IMH1=DIMAG(J1)+DREAL(Y1)
      H(0)=CMPLX(REH0,IMH0)
      H(1)=CMPLX(REH1,IMH1)
      END
      
