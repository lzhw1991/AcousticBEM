libhelmholtz.so: Operators.o
	gcc Operators.o -shared -o libhelmholtz.so

Operators.o: Operators.c
	gcc -c -fPIC Operators.c -o Operators.o

clean:
	rm Operators.o libhelmholtz.so
