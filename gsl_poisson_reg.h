#ifndef __GSL_POISSON_REG_H
#define __GSL_POISSON_REG_H
void gsl_poisson_reg(
	double * _xis, // l*n
	double * _yis,
	double * _ois,
	size_t n,
	size_t l,
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
);
#endif
