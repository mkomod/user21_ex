#include "userex.h"

using namespace CppAD;
using namespace Eigen;

typedef AD<double> a_double;
typedef Matrix<a_double, Eigen::Dynamic, 1> a_vector;


a_double log_p(a_vector beta, VectorXd mu_beta, VectorXd s_beta) {
    a_double res = 0.0;
    for (int i = 0; i < beta.rows(); ++i) {
	res += log(1/(sqrt(2.0*PI)*s_beta(i))) - 
	    pow((beta(i)-mu_beta(i))/s_beta(i), 2)*0.5;
    }
    return res;
}


double log_p(VectorXd beta, VectorXd mu_beta, VectorXd s_beta) {
    double res = 0.0;
    for (int i = 0; i < beta.rows(); ++i) {
	res += log(1/(sqrt(2.0*PI)*s_beta(i))) - 
	    pow((beta(i)-mu_beta(i))/s_beta(i), 2)*0.5;
    }
    return res;
}


a_double L(a_vector beta, VectorXd y, MatrixXd x, double s, 
    VectorXd mu_beta, VectorXd s_beta) 
{
    a_double l = 0.0;
    for (int i = 0; i < y.rows(); ++i) {
	a_double m = 0.0;
	for (int j = 0; j < beta.rows(); ++j) {
	    m += beta(j) * x(i, j);
	}
	l += log(1/(sqrt(2.0*PI)*s)) - pow((y(i)-m)/s, 2)*0.5;
    }
    l += log_p(beta, mu_beta, s_beta);
    return l;
}


// [[Rcpp::export]]
double L(VectorXd beta, VectorXd y, MatrixXd x, double s, 
    VectorXd mu_beta, VectorXd s_beta) 
{
    double l = 0.0;
    for (int i = 0; i < y.rows(); ++i) {
	double m = 0.0;
	for (int j = 0; j < beta.rows(); ++j) {
	    m += beta(j) * x(i, j);
	}
	l += log(1/(sqrt(2.0*PI)*s)) - pow((y(i)-m)/s, 2)*0.5;
    }
    l += log_p(beta, mu_beta, s_beta);
    return l;
}


// [[Rcpp::export]]
VectorXd grad_L(VectorXd beta, VectorXd y, MatrixXd x, double s, 
    VectorXd mu_beta, VectorXd s_beta) 
{
    a_vector a_beta(beta.rows());
    a_vector L_y(1);
    ADFun<double> gr;
    
    for (int i = 0; i < beta.rows(); ++i)
	a_beta(i) = beta(i);

    Independent(a_beta);
    L_y(0) = L(a_beta, y, x, s, mu_beta, s_beta);
    gr.Dependent(a_beta, L_y);

    return gr.Jacobian(beta);
}

