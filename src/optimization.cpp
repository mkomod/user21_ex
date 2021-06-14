#include "userex.h"

using namespace CppAD;
using namespace Eigen;


typedef AD<double> a_double;
typedef Matrix<a_double, Eigen::Dynamic, 1> a_vector;


// [[Rcpp::export]]
double fx(VectorXd x) {
    double r = 0.0;
    for (int i = 0; i < x.rows(); ++i) {
	r += x(i) * x(i);
    }
    r = sqrt(r);
    return sin(r)/r;
}


a_double fx(a_vector x) {
    a_double r = 0.0;
    for (int i = 0; i < x.rows(); ++i) {
	r += x(i) * x(i);
    }
    r = sqrt(r);
    return sin(r)/r;
}

// [[Rcpp::export]]
VectorXd fp(VectorXd x) 
{
    a_vector ax(x.rows());
    a_vector y(1);
    ADFun<double> gr;

    for (int i = 0; i < x.rows(); ++i)
	ax(i) = x(i);

    Independent(ax);
    y(0) = fx(ax);
    gr.Dependent(ax, y);

    return gr.Jacobian(x);
}

