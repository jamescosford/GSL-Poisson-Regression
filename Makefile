all:
	gcc -c -Wall -Werror -fpic -std=c99 gsl_poisson_reg.c
	ar rcs libgsl_poisson_reg.a gsl_poisson_reg.o
	gcc -o example libtest.c -I/usr/include/gsl -lgsl_poisson_reg -L`pwd` -std=c99 -lgsl -lgslcblas -lm

clean:
	rm gsl_poisson_reg.o libgsl_poisson_reg.a example
