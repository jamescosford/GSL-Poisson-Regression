#!/bin/sh
gcc -c -Wall -Werror -fpic -std=c99 gsl_poisson_reg.c
gcc -shared -o libgsl_poisson_reg.so gsl_poisson_reg.o
gcc -o mm libtest.c -I/usr/include/gsl -lgsl_poisson_reg -L`pwd` -std=c99 -lgsl -lgslcblas -lm
cp libgsl_poisson_reg.so /usr/local/lib
ldconfig
