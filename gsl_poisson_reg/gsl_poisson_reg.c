#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include "gsl_poisson_reg.h"

#undef __SAS_OUTPUT
#define __SAS_OUTPUT
#undef __DEBUG_OUTPUT
#define __DEBUG_OUTPUT

struct ps {
	size_t n; // problem dimension
	size_t l; // number of observations
	
	gsl_matrix* xis; // explanatory variables + constant term
	gsl_vector* yis; // observed values 
	gsl_vector* ois; // offset / exposure values (logged) (zeros if n.a.)
};

double pois_ll_f(const gsl_vector* beta, void* params) {
	struct ps* p = (struct ps*)params;
	double ll = 0;
	double res_bx;
	double yi = 0;
	double oi = 0;
	
	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int i = 0; i < p->l; i++) {	
		yi = gsl_vector_get(p->yis, i);
		oi = gsl_vector_get(p->ois, i);
		gsl_matrix_get_row(xi, p->xis, i);
		gsl_blas_ddot(beta, xi, &res_bx);
		double lldelta = yi*(oi + res_bx) - exp(oi + res_bx);
		ll += lldelta;
	}
	gsl_vector_free(xi);
	return -ll;
}

void pois_ll_df(const gsl_vector* beta, void* params, gsl_vector* g) {
	struct ps* p = (struct ps*)params;
	double res_beta;

	gsl_vector_set_all(g, 0);
	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int j = 0; j < p->n; j++) {
		double dbi = 0;
		for (int i = 0; i < p->l; i++) {
			gsl_matrix_get_row(xi, p->xis, i);
			gsl_blas_ddot(beta, xi, &res_beta);
			
			double yi = gsl_vector_get(p->yis, i);
			double oi = gsl_vector_get(p->ois, i);
			double xji = gsl_vector_get(xi, j);
			
			dbi += xji*(yi - exp(oi + res_beta));
		}
		gsl_vector_set(g, j, -dbi);
	}
	gsl_vector_free(xi);
}

void pois_ll_fdf(const gsl_vector* beta, void* params, double* f, gsl_vector* g) {
	*f = pois_ll_f(beta, params);
	pois_ll_df(beta, params, g);
}

void calc_ses(const gsl_vector* beta, void* params, gsl_vector* ses) {
	struct ps* p = (struct ps*)params;

	double res_bx;
	gsl_matrix* h = gsl_matrix_calloc(p->n, p->n);
	gsl_vector* xi = gsl_vector_alloc(p->n);
	
	// Generate negative Hessian matrix (Fisher Information matrix)
	for (int k = 0; k < p->n; k++) {
		for (int j = 0; j < p->n; j++) {
			double d2f = 0;
			for (int i = 0; i < p->l; i++) {
				gsl_matrix_get_row(xi, p->xis, i);
				gsl_blas_ddot(beta, xi, &res_bx);
				double xik = gsl_vector_get(xi, k);
				double xij = gsl_vector_get(xi, j);
				double oi = gsl_vector_get(p->ois, i);
				double d2fdelta = xik*xij*exp(oi + res_bx);
				d2f += d2fdelta;
			}
			gsl_matrix_set(h,k,j,d2f);
		}
	}
	
	// Invert it
	gsl_matrix* inverse = gsl_matrix_calloc(p->n, p->n);
	gsl_permutation* perm = gsl_permutation_alloc(p->n);
	int s; // Signum
	gsl_linalg_LU_decomp(h, perm, &s);
	gsl_linalg_LU_invert(h, perm, inverse);

	// Standard errors are square roots of diagonal elements
	for (int k = 0; k < p->n; k++) {
		gsl_vector_set(ses, k, sqrt(gsl_matrix_get(inverse, k,k)));
	}
	
	gsl_permutation_free(perm);
	gsl_matrix_free(inverse);
	gsl_vector_free(xi);
	gsl_matrix_free(h);	
}

double calc_deviance(const gsl_vector* beta, void* params) {	
	struct ps* p = (struct ps*)params;
	double d = 0;
	double beta_xi;
	double yi = 0;
	
	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int i = 0; i < p->l; i++) {	
		yi = gsl_vector_get(p->yis, i);
		gsl_matrix_get_row(xi, p->xis, i);
		gsl_blas_ddot(beta, xi, &beta_xi);
		double oi = gsl_vector_get(p->ois, i);
		double m_i = exp(oi + beta_xi);
		double ddelta = 0;
		if (yi > 0) { // when yi == 0 this blew up
			ddelta = yi*log(yi/m_i) - (yi - m_i);
		} else {
			ddelta = m_i;
		}
		
		d += ddelta;
	}	
	gsl_vector_free(xi);
	return 2*d; 
}

double calc_pearson_chi2(const gsl_vector* beta, void* params) {
	struct ps* p = (struct ps*)params;
	double pearson = 0;
	double beta_xi;
	double yi = 0;
	
	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int i = 0; i < p->l; i++) {	
		yi = gsl_vector_get(p->yis, i);
		gsl_matrix_get_row(xi, p->xis, i);
		gsl_blas_ddot(beta, xi, &beta_xi);
		double oi = gsl_vector_get(p->ois, i);
		double m_i = exp(oi + beta_xi);
		double pdelta = (yi - m_i)*(yi - m_i)/m_i;
		pearson += pdelta;
	}
	gsl_vector_free(xi);
	return pearson; 
}

void calc_param_wald_chi2(
				gsl_vector * beta, 
				gsl_vector * ses, 
				gsl_vector * waldchi2, 
				gsl_vector * pchi2
			) {
	for (int i = 0; i < beta->size; i++) {
		double beta_i = gsl_vector_get(beta, i);
		double se_i = gsl_vector_get(ses, i);
		double chi_sq = beta_i*beta_i/(se_i*se_i); 
		double chi_p = gsl_cdf_chisq_Q(chi_sq, 1); // 1 dof per parameter

		gsl_vector_set(waldchi2, i, chi_sq);
		gsl_vector_set(pchi2, i, chi_p);
	}
}
void calc_param_cis(
				gsl_vector * beta, 
				gsl_vector * ses, 
				gsl_vector * ci_l, 
				gsl_vector * ci_u
			) {
 
	for (int i = 0; i < beta->size; i++) {
		double beta_i = gsl_vector_get(beta, i);
		double se_i = gsl_vector_get(ses, i);
		double p_n = gsl_cdf_ugaussian_Pinv(0.975);
		gsl_vector_set(ci_l, i, beta_i-p_n*se_i);
		gsl_vector_set(ci_u, i, beta_i+p_n*se_i);
	}
}

void gsl_poisson_reg(
	double * _xis,
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
) {
	// Allocate GSL structures to hold data
	gsl_matrix* xis = gsl_matrix_calloc(l,n);
	gsl_vector* yis = gsl_vector_alloc(l);
	gsl_vector* ois = gsl_vector_alloc(l);

	// Copy data from native C types to GSL
	for (int i = 0; i < l; i++) {
		for (int j = 0; j < n; j++) {
			gsl_matrix_set(xis, i, j, _xis[i*n + j]);
#ifdef __SAS_OUTPUT
			printf("%f ",  _xis[i*n + j]);
#endif
		}
#ifdef __SAS_OUTPUT
		printf("%f %f\n", _ois[i], _yis[i]);
#endif
		gsl_vector_set(ois, i, _ois[i]);
		gsl_vector_set(yis, i, _yis[i]);
	}
	
	struct ps p;
	p.xis = xis;
	p.yis = yis;
	p.ois = ois;
	p.n = n;
	p.l = l;

	// GSL fdf minimiser struct
	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = pois_ll_f; // My f function
	my_func.df = pois_ll_df; // My df function
	my_func.fdf = pois_ll_fdf; // My fdf function
	my_func.params = &p; // My params 

	// Parameter vector beta - GSL will modify this to produce a minimum
	gsl_vector* beta = gsl_vector_alloc(n);
	for (int i = 0; i < n; i++) {
		gsl_vector_set(beta, i, _beta[i]);
	}

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer* s;
	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc(T, n);
	
	gsl_multimin_fdfminimizer_set(s, &my_func, beta, 0.01, 0.1);

	int status;
	size_t iter = 0;
	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);
		status = gsl_multimin_test_gradient(s->gradient, 1e-2);
	} while (status == GSL_CONTINUE && iter < 1000);

	// Calculate standard errors
	gsl_vector* ses = gsl_vector_alloc(n);	
	calc_ses(s->x, &p, ses); 

	double deviance = calc_deviance(s->x, &p);
	double pearson = calc_pearson_chi2(s->x, &p);

	gsl_vector* ci_l = gsl_vector_alloc(n);
	gsl_vector* ci_u = gsl_vector_alloc(n);
	gsl_vector* waldChi2 = gsl_vector_alloc(n);
	gsl_vector* chi2P = gsl_vector_alloc(n);

	calc_param_cis(s->x, ses, ci_l, ci_u);
	calc_param_wald_chi2(s->x, ses, waldChi2, chi2P);

	// Copy results to outvars
	for (int i = 0; i < n; i++) {
		_beta[i] = gsl_vector_get(s->x, i);	
		_ses[i] = gsl_vector_get(ses, i);	
		_ci_l[i] = gsl_vector_get(ci_l, i);	
		_ci_u[i] = gsl_vector_get(ci_u, i);	
		_waldChi2[i] = gsl_vector_get(waldChi2, i);	
		_chi2P[i] = gsl_vector_get(chi2P, i);	
	}
	*_ll = -s->f;
	*_deviance = deviance;
	*_pearson_chi2 = pearson;
	*_dof = l - n;

#ifdef __DEBUG_OUTPUT
	printf("Iter: %d\n", (int)iter);
	printf("beta\n");
	for (int i = 0; i < n; i++) printf("%f ", _beta[i]);
	printf("\n");
	for (int i = 0; i < n; i++) printf("%f ", _ses[i]);
	printf("\n");
#endif

	gsl_vector_free(chi2P);
	gsl_vector_free(waldChi2);
	gsl_vector_free(ci_u);
	gsl_vector_free(ci_l);
	gsl_vector_free(ses);
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(beta);
	gsl_vector_free(ois);
	gsl_vector_free(yis);
	gsl_matrix_free(xis);
}
