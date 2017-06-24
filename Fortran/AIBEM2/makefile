AIBEM2: AIBEM2.o AIBEM2_T.o GEOM2D.o H2LC.o CGLS.o FNHANK.o
	gfortran -L../fgsl -o AIBEM2 AIBEM2.o AIBEM2_T.o GEOM2D.o H2LC.o CGLS.o FNHANK.o -lfgsl -lgsl -lgslcblas

AIBEM2.o: AIBEM2.FOR
	gfortran -c AIBEM2.FOR

AIBEM2_T.o: AIBEM2_T.FOR
	gfortran -c AIBEM2_T.FOR

GEOM2D.o: GEOM2D.FOR
	gfortran -c GEOM2D.FOR

H2LC.o: H2LC.FOR
	gfortran -c H2LC.FOR

CGLS.o: CGLS.FOR
	gfortran -c CGLS.FOR

FNHANK.o: FNHANK.FOR
	gfortran -I../fgsl -c FNHANK.FOR
clean:
	rm AIBEM2.o AIBEM2_T.o GEOM2D.o H2LC.o CGLS.o FNHANK.o AIBEM2
