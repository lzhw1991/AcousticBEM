libhelmholtz.so: Operators.o
	gcc Operators.o -shared -o libhelmholtz.so

Operators.o: Operators.c
	gcc -c -fPIC Operators.c -o Operators.o

UnitTests: UnitTests.o Operators.o CuTest.o
	gcc Operators.o UnitTests.o CuTest.o -lm -o UnitTests

CuTest.o: cutest/CuTest.c
	gcc -c cutest/CuTest.c -o CuTest.o

UnitTests.o: UnitTests.c
	gcc -c UnitTests.c -o UnitTests.o

clean:
	rm Operators.o UnitTests.o CuTest.o UnitTests libhelmholtz.so *~
