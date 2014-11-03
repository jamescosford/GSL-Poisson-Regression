#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gsl_poisson_reg.h"
int main(int argc, char** argv) {
	const size_t l = 31; // Rows of data
	const size_t n = 2; // Number of params

	// https://onlinecourses.science.psu.edu/stat504/node/170
	// [Income, NCases, CreditCards]
	double data_src[31][3] = {
		{24, 1, 0},
		{27, 1, 0},
		{28, 5, 2},
		{29, 3, 0},
		{30, 9, 1},
		{31, 5, 1},
		{32, 8, 0},
		{33, 1, 0},
		{34, 7, 1},
		{35, 1, 1},
		{38, 3, 1},
		{39, 2, 0},
		{40, 5, 0},
		{41, 2, 0},
		{42, 2, 0},
		{45, 1, 1},
		{48, 1, 0},
		{49, 1, 0},
		{50, 10, 2},
		{52, 1, 0},
		{59, 1, 0},
		{60, 5, 2},
		{65, 6, 6},
		{68, 3, 3},
		{70, 5, 3},
		{79, 1, 0},
		{80, 1, 0},
		{84, 1, 0},
		{94, 1, 0},
		{120, 6, 6},
		{130, 1, 1}
	};
	double* xis = malloc(l*n*sizeof(double *));
	double yis[31];
	double ois[31];
	
	for (int i = 0; i < l; i++) {
		xis[i*n + 0] = 1.0; // Constant
		xis[i*n + 1] = data_src[i][0]; // incomei
		ois[i] = log(data_src[i][1]); // exposure (cases)
		yis[i] = data_src[i][2]; // creditcards
	}
	
	double beta[2];
	double ses[2];
	double ci_l[2];
	double ci_u[2];
	double waldChi2[2];
	double chi2P[2];
	double ll;
	double deviance;
	double pearson;
	int dof;

	gsl_poisson_reg(
		xis,
		yis,
		ois,
		n,
		l,
		beta, 
		ses,
		ci_l,
		ci_u,
		waldChi2,
		chi2P,
		&ll,
		&deviance,
		&pearson,
		&dof
	);
	
	printf("\nGoodness of fit:\n");
	printf("\tLog-likelihood: %.4f\n", ll);	
	printf("\tDeviance: %.4f\n", deviance);
	printf("\tPearson Chi^2: %.4f\n", pearson);
	printf("\tDoF: %d\n", dof);

	printf("\nAnalysis of parameter estimates:\n");
	printf("value\ts.e.\tw95ci_l\tw95ci_u\tchi2\tp>chi2\n");
	for (int i = 0; i < n; i++) {
		printf("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
			beta[i],
			ses[i],
			ci_l[i],
			ci_u[i],
			waldChi2[i],
			chi2P[i]
		);
	}
	free(xis);
}	
/*
void gsl_poisson_reg(
	double ** _xis, // lxn
	double * _yis,
	double * _ois,
	int n,
	int l,
	double * _beta,
	double * _ses,
	double * _ci_l,
	double * _ci_u,
	double * _waldChi2,
	double * _chi2P,
	double * _ll,
	double * _deviance, 
	double * _pearson_chi2,
	int * _dof	
*/	
