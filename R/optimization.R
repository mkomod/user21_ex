Rcpp::sourceCpp("../src/optimization.cpp")

x <- matrix(c(1, 0.2))
fx(x)
fp(x)
optim(x, fx, fp, method="CG", control=list(fnscale=-1))

f <- \(x1, x2) sin(x1 + x2) + cos(x1 + x2)
f <- function(x) { r <- sqrt(sum(x^2)); 3 * sin(r)/r }
f <- function(x) { r <- sqrt(sum(x^2)); sin(r)/r }

f <- function(x, y) { r <- sqrt(x^2+y^2); sin(r)/r }
x1 <- seq(-5, 5, 0.1)
x2 <- seq(-5, 5, 0.1)
Z <- matrix(0, length(x1), length(x2))

for (i in seq_along(x1))
    for (j in seq_along(x2))
	Z[i, j] <- f(x1[i], x2[j])

persp(x1, x2, Z)
