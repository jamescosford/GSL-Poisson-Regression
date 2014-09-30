# Poisson Regression using GSL

After quite a bit of reading on the subject, I had assembled a couple of key facts:

1. Poisson regression is best performed using the *maximum-likelihood* method
2. GSL can be employed to perform multiparameter minimsation using the *gsl_multimin\** suite of functions

## Maximum likelihood estimation

"Maximum likelihood, also called the maximum likelihood method, is the procedure of finding the value of one or more parameters for a given statistic which makes the *known* likelihood distribution a *maximum*." - **Wolfram**

### Synthesising from various sources:

A discrete random variable $X$ is said to have a Poisson distribution with parameter $\mu \gt 0$, if for $y_i=0,1,2,...,$ the probability mass function of $X$ is given by:

$$f(y; \mu) = Pr(X = y) = \frac{\mu^{y} e^{-\mu}}{y!} \\ \mu = E(X) = Var(X)$$ which says "The probability that the observed value is equal to the mean value is given by... that function" 

Poisson regression employs the following model:

$$\mu_i = \mu_i(\beta) = e^{\mathbf{x}_i\prime \beta}\text{, or} \\
log(\mu_i) = \mathbf{x}_i\prime\beta\ $$  

This is the log-linear Poisson regression model, where $\beta$ is a vector of unknown coefficients to the elements of the $\mathbf{x}_i$ vector, which contains the explanatory variables of observation $i$.

These can be further decomposed to

$$\mu_i = e^{\beta_0 x_{i,0}}\cdot e^{\beta_1 x_{i,1}}\cdot ... \cdot e^{\beta_n x_{i,n}} \text{, or} \\ log(\mu_i) = \beta_0 x_{i,0} + \beta_1 x_{i,1} + ... + \beta_n x_{i,n}$$

The maximum likelihood model is used to determine the coefficient vector $\beta$ which maximises the likelihood of the observed data $\mathbf{y}$ occurring, given the explanatory variables $\mathbf{x}$.

"... a likelihood function (often simply the likelihood) is a function of the parameters of a statistical model . The *likelihood* of a set of parameter values, $\theta$, given outcomes $y$, is equal to the *probability* of those observed outcomes given those parameter values, that is $\mathcal{L}(\theta | y) = P(y | \theta)$. " - **Wikipedia**

 "... likelihood functions functions play a key role in statistical inference, especially methods of estimating a parameter from a set of statistics. In informal contexts, 'likelihood' is often used as a synonym for 'probability'. But in statistical usage, a distinction is made depending on the roles of the outcome or parameter. *Probability* is used when describing a function of the outcome given a fixed parameter value. For example, if a coin is flipped 10 times and it is a fair coin, what is the *probability* of it landing heads-up every time? *Likelihood* is used when describing a function of a parameter given an *outcome*. For example, if a coin is flipped 10 times and it has landed heads-up 10 times, what is the likelihood that the coin is fair?" - **Wikipedia**

The likelihood function is the product of all individual likelihood contributions across $i = 1,2,...,l$, where $l$ is the number of observations. This relies on the assumption that all measurements are independent.

$$
\mathcal{L}_i = \frac{\mu_i^{y_i} \cdot e^{-\mu_i}}{y_i!} \\ 
\mathcal{L} = \prod_{i=1}^l{\mathcal{L}_i}
$$ 

Where $\prod_{i=1}^l{\mathcal{L}_i}$ is the multiplicative product $\mathcal{L}_1 \cdot \mathcal{L}_2 \cdot ... \cdot \mathcal{L}_l$

While it is possible to work with the raw likelihood function as above, it has been determined that choosing to work instead with the log of the likelihood function is preferable, as it is easier to work with both mathematically and computationally. 

The log-likelihood $\ell = log(\mathcal{L})$ for the Poisson regression problem can be expressed as 

$$\ell = \sum_{i=1}^{l} \left[ y_i\ log (\mu_i) - \mu_i - log(y_i!)\right].$$

Remembering that we are attempting to determine the set of coefficients $\beta$ which maximises the (log)likelihood function given our observed data and explanatory variables, we can eliminate the factorial expression on the end as this does not depend on $\beta$. 

Substituting our modeled value for $\mu$ we arrive at

$$\ell = \sum_{i=1}^{l} y_i log(e^{\beta\prime \mathbf{x}_i}) - e^{\beta\prime \mathbf{x}_i} \text{,}$$

which resolves to 

$$\ell = \sum_{i=1}^{l} y_i {\beta\prime \mathbf{x}_i} - e^{\beta\prime \mathbf{x}_i} \text{.}$$

This is the function which we are attempting to maximise, to determine the most likely value of $\beta$. 

To assist in the maximisation, it is necessary to construct a *score* vector, which is a vector of length $n$ (the same dimension as $\beta$) containing the evaluated partial derivative for each parameter if $\beta$. This requires partial differentiation of the $ll$ function with respect to each member of $\beta$.

$$
\begin{align}
\frac{\delta \ell(\beta)}{\delta\beta} & = \sum_{i=1}^l x_{i,j} y_i - e^{\beta\prime \mathbf{x}_i} \cdot x_{i,j} \\
 & = \sum_{i=1}^l x_{i,j}(y_i - e^{\beta\prime \mathbf{x}_i}) \\
 \end{align}
$$ 
for the $j$th member of $\beta$.

### Including exposure
Modeling rare occurrences with Poisson regression may involve inclusions of observation data with varying levels of "exposure". Consider for example a rare infectious disease being modeled among a number of population sub-groups. In a sub-group with more members, there are more people at risk of infection, even though the probability of infection may be similar between groups. A naive model might conclude that the risk was not constant, but higher among larger sub-groups, which may be incorrect. To account for exposure ($t$), we express the model slightly differently;

$$\mu = t_i e^{\beta_i\prime x_i} \text{,}$$

which gives us;

$$Pr(X = y_i|\mu_i,t_i) = \frac{e^{-\mu_i t_i}(\mu_i t_i)^{y_i}}{y_i!}, \text{ for }y = 1, 2, ...$$

The log-likelihood $\ell = log(\mathcal{L})$ becomes

$$
\begin{align}
\ell &= \sum_{i=1}^{l} y_i log (t_i \mu_i) - t_i \mu_i - log(y_i!) \\
&=\sum_{i=1}^{l} y_i \left(log(t_i) + log(e^{\beta\prime x_i})\right) + t_i e^{\beta\prime x_i} \\
&=\sum_{i=1}^{l} y_i \left(log(t_i) + \beta\prime x_i\right) + t_i e^{\beta\prime x_i}  \text{ or } \sum_{i=1}^{l} y_i \left(\log(t_i) + \beta\prime x_i\right) + e^{log t_i + \beta\prime x_i} \text{,}
\end{align}
$$

and the gradient becomes

$$\frac{\delta \ell(\beta)}{\delta\beta} = \sum_{i=1}^l x_{i,j}(y_i - e^{log t_i + \beta\prime \mathbf{x}_i})$$ for the $ith$ element of $\beta$.

## Poisson maximum likelihood estimation using GSL

"The GNU Scientific Library (GSL) is a numerical library for C and C++ programmers. It is free software under the GNU General Public License.

The library provides a wide range of mathematical routines such as random number generators, special functions and least squares fitting. There are over 1000 functions in total with an extensive test suite." - **gnu.org**

"The problem of multidimensional minimization requires finding a point $\mathbf{x}$ such that the scalar function $f(x_1, x_2, ..., x_n)$ takes a value which is lower than at any neighbouring point. For smooth functions the gradient $g = \nabla f$ vanishes at the minimum." - **gnu.org**

"Algorithms making use of the gradient of the function perform a one-dimensional line minimisation along this direction until the lowest point is found to a suitable tolerance. The search direction is then updated with local information from the function and its derivatives, and the whole process is repeated until the true $n$-dimensional minimum is found" - **gnu.org**

The GSL provides multidimensional minimisation via the `gsl_multimin*` family of functions.

*From here on in, I will be providing an overview of implementation, with particular focus on parts of the problem which I found to be poorly explained elsewhere.*

Since we have a relatively simple function which we have been able to differentiate, we can exploit a minimiser which uses gradient information. This will end up being faster than one which does not. As such, we need to create and initialise a `gsl_multimin_fdfminimizer`. Where fdf might be shorthand for **function** value and **differential function** value.

### Minimisation summary

The minimisation routines contain algorithms which modify a parameter vector (in our case $\beta$) repeatedly, and have us provide implementations of our function (and gradient functions) to evaluate after each change. The minimisation routines respond to the changes in our function (and gradient) values by strategically changing $\beta$ until a minimum is reached.

**Note!** because we are using the *maximum likelihood* method, we are attempting to *maximise* the likelihood. But we only have access to routines which perform *minimisation*! So, we intelligently elect to minimise the *negative log-likelihood*, which will the the same effect as maximising the positive log-likelihood.

###Providing a function to minimise

To provide our function to GSL for minimisation, we have to present it in the expected format. GSL requires a value function ($f$), a gradient function ($df$) and a combined value/gradient function ($fdf$), supplied to it within a GSL data type called `gsl_multimin_function_fdf`. This initialisation will be examined in more detail later, but for now I will introduce and explain each of the required functions.

###$f$
```
double (* f) (const gsl_vector * beta, void * params)
```

Receives the current estimate of the parameters in `beta` ($\beta$). Remember, the purpose of the minimisation is to optimise $\beta$ by having the minimisation routines intelligently change the values to find a minimum. The params value can contain anything you need to compute the log-likelihood. In my implementation, it was a custom `struct` containing the observed values $\mathbf{y}$, the explanatory values $\mathbf{x}$ and the values of $l$ and $n$. Using the values in `params` and `beta` we compute our log-likelihood value, negate it, and return it.
```
double pois_ll_f(const gsl_vector* beta, void* params) {
	struct ps* p = (struct ps*)params;
	double ll = 0;
	double res_bx;
	double yi = 0;
	double oi = 0;
	
	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int i = 0; i < p->l; i++) {
		// Observed	
		yi = gsl_vector_get(p->yis, i);
		
		// Exposure/Offset 
		oi = gsl_vector_get(p->ois, i); 
		
		// Get explanatory vector x_i
		gsl_matrix_get_row(xi, p->xis, i); 			
		
		// Calculate beta (dot) x_i
		gsl_blas_ddot(beta, xi, &res_bx);
		
		// Calc ll_i 	
		double lldelta = yi*(oi + res_bx) - exp(oi + res_bx); 
		
		ll += lldelta;
	}
	gsl_vector_free(xi);

	// Return negative for minimisation
	return -ll; 
}
```
###$df$
```
void (* df) (const gsl_vector * beta, void * params, gsl_vector * g)
```
Receives the current estimate of the parameters in `beta`, and the data we placed in params. The vector `g` is the same size as `beta`, and upon completion of this function, each element of `g` should contain the value of the partial differential of the log-likelihood with respect to the appropriate member of $\beta$.

```
void pois_ll_df(const gsl_vector* beta, void* params, gsl_vector* g) {
	struct ps* p = (struct ps*)params;
	double res_bx;

	gsl_vector* xi = gsl_vector_alloc(p->n);
	for (int j = 0; j < p->n; j++) {
		double dbi = 0;
		for (int i = 0; i < p->l; i++) {
			
			// x_i
			gsl_matrix_get_row(xi, p->xis, i);	
			
			// beta (dot) x_i
			gsl_blas_ddot(beta, xi, &res_bx);
			
			double yi = gsl_vector_get(p->yis, i);	// y_i
			double oi = gsl_vector_get(p->ois, i);	// offset_i
			double xij = gsl_vector_get(xi, j);		// x[j][i]
			
			// Calc dll/dtheta_j _i
			dbi += xij*(yi - exp(oi + res_bx));	
		}
		// Set negative gradient value (minimisation)
		gsl_vector_set(g, j, -dbi);		
	}
	gsl_vector_free(xi);
}
```

###$f df$
```
void (* fdf) (const gsl_vector * beta, void * params, double * f, gsl_vector * g)
```

Receives the same data as the others, and computes *both* the value of the log-likelihood, returned via `f` and the score/gradient vector `g`. This function should ideally do both at the same time, and this is an opportunity to exploit efficiency gains should the computation of one duplicate steps in the other. For now, I simply had this employ each of the first two functions in turn.

```
void pois_ll_fdf(const gsl_vector * beta, void * params, double * f, gsl_vector * g) {
    *f = pois_ll_f(beta, params);
    pois_ll_df(beta, params, g);
}
```
### Setup and invocation
Parameter struct - this can contain whatever you need it to contain for your given problem. For this problem, I constructed the following:
```
struct ps {
	size_t n; // problem dimension
	size_t l; // number of observations
	
	// explanatory variables + constant term
	gsl_matrix* xis; 
	
	// observed values 
	gsl_vector* yis; 
	
	// offset / exposure values (logged) (zeros if n.a.)
	gsl_vector* ois; 
};
```
Example data from [https://onlinecourses.science.psu.edu/stat504/node/170](https://onlinecourses.science.psu.edu/stat504/node/170).
```
int main(int argc, char** argv) {
	// Parameter struct
	struct ps p;
	// Rows of data
	const size_t l = 31;    
	// Number of params to fit/dimension of minimsation
	const size_t n = 2;     
	
	gsl_matrix* xis = gsl_matrix_alloc(l,n);
	gsl_vector* yis = gsl_vector_alloc(l);
	gsl_vector* ois = gsl_vector_alloc(l);

	// https://onlinecourses.science.psu.edu/stat504/node/170
	// [Income, NCases, CreditCards]
	double data_src[31][3] = {
		{24, 1, 0}, {27, 1, 0},	{28, 5, 2},	
		{29, 3, 0},	{30, 9, 1},	{31, 5, 1},	
		{32, 8, 0},	{33, 1, 0},	{34, 7, 1},	
		{35, 1, 1},	{38, 3, 1},	{39, 2, 0},	
		{40, 5, 0},	{41, 2, 0},	{42, 2, 0},	
		{45, 1, 1}, {48, 1, 0},	{49, 1, 0},	
		{50, 10, 2},{52, 1, 0},	{59, 1, 0},	
		{60, 5, 2},	{65, 6, 6},	{68, 3, 3},
		{70, 5, 3},	{79, 1, 0},	{80, 1, 0},	
		{84, 1, 0},	{94, 1, 0},	{120, 6, 6},
		{130, 1, 1}
	};
	
	for (int i = 0; i < l; i++) {
		// Constant
		gsl_matrix_set(xis, i, 0, 1.0); 
		// income
		gsl_matrix_set(xis, i, 1, data_src[i][0]);
		// exposure (cases)
		gsl_vector_set(ois, i, log(data_src[i][1])); 
		// creditcards
		gsl_vector_set(yis, i, data_src[i][2]); 
	}

	p.xis = xis;
	p.yis = yis;
	p.ois = ois;
	p.n = n;
	p.l = l;

	poisson_reg(&p);
	
	gsl_vector_free(ois);
	gsl_vector_free(yis);
	gsl_matrix_free(xis);
}
```
### The poisson regression function
```
void poisson_reg(struct ps * p) {
	// GSL fdf minimiser struct
	gsl_multimin_function_fdf my_func;

	my_func.n = p->n;
	my_func.f = pois_ll_f; // poisson f function
	my_func.df = pois_ll_df; // poisson df function
	my_func.fdf = pois_ll_fdf; // poisson fdf function
	my_func.params = p; // params 

	// Parameter vector beta - 
	// GSL will modify this to produce a minimum
	gsl_vector* beta = gsl_vector_alloc(p->n);

	// all zeros starting estimate
	gsl_vector_set_all(beta, 0.); 

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer* s;
	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc(T, p->n);
	
	gsl_multimin_fdfminimizer_set(
		s, 
		&my_func, 
		beta, 
		0.01, 
		0.1
	);

	int status;
	size_t iter = 0;
	do {
		iter++;
		
		status = gsl_multimin_fdfminimizer_iterate(s);
        
        // A GSL example checks the status here for error 
	    // conditions, but this lead to early termination 
		// of minimisations which otherwise complete successfully
		// due to the gradientcondition below, so I removed them.
		
		status = gsl_multimin_test_gradient(s->gradient, 1e-3);
	} while (status == GSL_CONTINUE && iter < 1000);

	// Calculate standard errors
	gsl_vector* ses = gsl_vector_alloc(p->n);	
	calc_ses(s->x, p, ses); 

	printf("\nGoodness of fit:\n");
	printf("Log-likelihood: %.4f\n", -s->f);	
	double deviance = calc_deviance(s->x, p);
	printf("Deviance: %.4f\n", deviance);
	double pearson = calc_pearson_chi2(s->x, p);
	printf("Pearson Chi^2: %.4f\n", pearson);
	printf("DoF: %d\n", (int)(p->l-p->n));

	gsl_vector* ci_l = gsl_vector_alloc(p->n);
	gsl_vector* ci_u = gsl_vector_alloc(p->n);
	gsl_vector* waldChi2 = gsl_vector_alloc(p->n);
	gsl_vector* chi2P = gsl_vector_alloc(p->n);

	calc_param_cis(s->x, ses, ci_l, ci_u);
	calc_param_wald_chi2(s->x, ses, waldChi2, chi2P);

	printf("\nAnalysis of parameter estimates:\n");
	printf("value\ts.e.\tw95ci_l\tw95ci_u\tchi2\tp>chi2\n");
	for (int i = 0; i < p->n; i++) {
		printf("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
			gsl_vector_get(s->x, i),
			gsl_vector_get(ses, i),
			gsl_vector_get(ci_l, i),
			gsl_vector_get(ci_u, i),
			gsl_vector_get(waldChi2, i),
			gsl_vector_get(chi2P, i)
		);
	}

	gsl_vector_free(chi2P);
	gsl_vector_free(waldChi2);
	gsl_vector_free(ci_u);
	gsl_vector_free(ci_l);
	gsl_vector_free(ses);
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(beta);
}
```
### Calculating standard error of estimates
While there appear to be as many ways to do this as to skin a proverbial cat, I will present one implementation which gets appropriate results according to the example linked above. The standard errors associated with each estimate can be approximated as the square root of the diagonal elements of the inverted fisher information matrix of the problem. 

The fisher information matrix is the negative of the hessian matrix of the problem, which is the matrix of second-order partial differentials of our log-likelihood function with respect to each parameter of our parameter vector $\beta$. 

|$\frac{\delta^2\beta}{\delta\beta_i\beta_j}$|$\beta_0$|$\beta_1$|...|$\beta_n$|
|:-|:-|:-|:-|:-|
|$\beta_0$|$-x_{i0}^2e^{\beta\prime x_i}$|$-x_{i0}x_{i1}e^{\beta\prime x_i}$|...|$-x_{i0}x_{in}e^{\beta\prime x_i}$|
|$\beta_1$|$-x_{i1}x_{i0}e^{\beta\prime x_i}$|$-x_{i1}^2e^{\beta\prime x_i}$|...|$-x_{i1}x_{in}e^{\beta\prime x_i}$|
|...|...|...|...|...|
|$\beta_n$|$-x_{in}x_{i0}e^{\beta\prime x_i}$|$-x_{in}x_{i1}e^{\beta\prime x_i}$|...|$-x_{in}^2e^{\beta\prime x_i}$|

If including a term for exposure as mentioned earlier, the exponential portion of the second order differential should include the exposure, such as $-x_{i0}^2e^{\mathbf{log t_i} + \beta\prime x_i}$ for all elements of the matrix.

### Implementation
```
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

```
### Running the complete example
```
developer@developer-VirtualBox:~/Development/temp/multimin$ ./mm

Goodness of fit:
    Log-likelihood: -12.9807
    Deviance: 28.4648
    Pearson Chi^2: 27.2497
    DoF: 29

Analysis of parameter estimates:
value	s.e.	w95ci_l	w95ci_u	chi2	p>chi2
-2.3866	0.3997	-3.1699	-1.6033	35.6599	0.0000
0.0208	0.0052	0.0106	0.0309	16.1542	0.0001

```

Which compares very well with the presented results from SAS in the example linked above.

![SAS output comparison](http://onlinecourses.science.psu.edu/stat504/sites/onlinecourses.science.psu.edu.stat504/files/lesson07/sas_output_13.gif "SAS Output for Comparison")


## Conclusion
I have run multiple tests against output from R and SAS, and the results are solid. It's not necessarily tidy, or optimised and is by no means perfect, but this document would have really helped me when I was trying to put all of this together, so hopefully it will help you too. 

### Other stats
Included in the output are a number of other statistics which are calculated to characterise goodness of fit and the value of the parameters which have been determined. I'm no expert on any of this, and while some of these make a certain kind of sense I don't have the time to go into much detail. There are plenty of references around so if you need to you can find them. 
```
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
```
