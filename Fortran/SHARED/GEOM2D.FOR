C***************************************************************
C         2D geometry routines by Stephen Kirkup             
C***************************************************************
C
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  smkirkup@uclan.ac.uk
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/GEOM2D.FOR
C 
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Manual can be downloaded from 
C   www.boundary-element-method.com/manuals/geom2d.pdf
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com


C Subroutine SUBV2: Gives the result in VEC of the subtraction
C  of 2-vectors VECB from VECA.
      SUBROUTINE SUBV2(VECA,VECB,VEC)
      REAL*8 VECA(2),VECB(2),VEC(2)
      VEC(1)=VECA(1)-VECB(1)
      VEC(2)=VECA(2)-VECB(2)
      END

C real function DOT2: Returns the dot product of two 2-vectors.
      REAL*8 FUNCTION DOT2(VECA,VECB)
      REAL*8 VECA(2),VECB(2)
      DOT2=VECA(1)*VECB(1)+VECA(2)*VECB(2)
      END

C real function SIZE2: Returns the modulus of a 2-vector.
C Requires the external module SSIZE2 (supplied).
      REAL*8 FUNCTION SIZE2(VEC)
      REAL*8 VEC(2),SSIZE2
      SIZE2=SQRT(SSIZE2(VEC))
      END

C real function SSIZE2: Returns the square of the modulus of a 
C 2-vector. Requires the external module DOT2 (supplied).
      REAL*8 FUNCTION SSIZE2(VEC)
      REAL*8 VEC(2),DOT2
      SSIZE2=DOT2(VEC,VEC)
      END

C real function DIST2: Returns the distance between two points
C  VECA and VECB.
      REAL*8 FUNCTION DIST2(VECA,VECB)
      REAL*8 VECA(2),VECB(2)
      REAL*8 SIZE2
      REAL*8 VEC(2)
      CALL SUBV2(VECA,VECB,VEC)
      DIST2=SIZE2(VEC)
      END

C real function NORM2: Finds the unit normal to the line between
C  VECA and VECB.
      SUBROUTINE NORM2(VECA,VECB,VECNOR)
      REAL*8 VECA(2),VECB(2),VECNOR(2)
      REAL*8 SIZE2
      REAL*8 VAMB(2),S
      CALL SUBV2(VECA,VECB,VAMB)
      S=SIZE2(VAMB)
      VECNOR(1)=VAMB(2)/S
      VECNOR(2)=-VAMB(1)/S
      END
