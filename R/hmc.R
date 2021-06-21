Rcpp::sourceCpp("../src/hmc.cpp")
library(MASS)

# ----------------------------------------
# Generate test data
# ----------------------------------------
set.seed(1)
x <- matrix(c(rep(1, 20), seq(0, 4, length.out=20)) , ncol=2)
b <- matrix(c(0, 2))
y <- x %*% b + rnorm(nrow(x))

plot(seq(0, 4, length.out=20), y)

# ----------------------------------------
# Test our C++ implementation
# ----------------------------------------
mu_beta <- matrix(rep(0, 2))
s_beta <- matrix(rep(5, 2))
L(b, y, x, 1, mu_beta, s_beta) == log(prod(dnorm(y, x %*% b, 1)) * prod(dnorm(c(b), mu_beta, s_beta)))
# if this works then our gradient function is also correct

# ----------------------------------------
# HMC
# ----------------------------------------
# See: Neal (2011)
HMC <- function (U, grad_U, epsilon, L, current_q)
{
    q <- current_q
    p <- rnorm(length(q),0,1)  # independent standard normal variates
    current_p <- p

    # Make a half step for momentum at the beginning
    p <- p - epsilon * grad_U(q) / 2

    # Alternate full steps for position and momentum
    # Leapfrog
    for (i in 1:L) {
	q <- q + epsilon * p
	if (i!=L) p <- p - epsilon * grad_U(q)
    }

    p = p - epsilon * grad_U(q) / 2
    p = -p

    current_U <-  U(current_q)
    current_K <- sum(current_p^2) / 2
    proposed_U <-  U(q)
    proposed_K <- sum(p^2) / 2

    if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K)) {
	return (q)  # accept
    } else {
	return (current_q)  # reject
    }
}

# Run sampler
B <- matrix(0, nrow=2, ncol=1e4)
for (iter in 1:1e4) {
    b <- HMC(function(b) -L(b, y, x, 1, mu_beta, s_beta), 
	function(b) -grad_L(b, y, x, 1, mu_beta, s_beta), 
	0.05, 7, b)
    B[ , iter] <- b
}


# ----------------------------------------
# Plots and tables
# ----------------------------------------
plot(B[1, ])
plot(B[2, ])

apply(B, 1, mean)
apply(B, 1, sd)

f <- MASS::kde2d(B[2,], B[1,], lims=c(1,3,-1,1), n=250)
png("hmc.png", width=800, height=800)
par(mfrow=c(2,2), mar=c(2,2,2,2))
plot(density(B[1, ]), col="purple", lwd=2, main=expression(beta[1]))
image(f, useRaster=T, col=hcl.colors(12, "purples", rev=T))
plot.new()
plot(density(B[2, ]), col="purple", lwd=2, main=expression(beta[2]))
dev.off()


# ----------------------------------------
# MAP
# ----------------------------------------
optim(par=c(1, 1), 
    fn=function(b) -L(b, y, x, 1, mu_beta, s_beta), 
    gr=function(b) -grad_L(b, y, x, 1, mu_beta, s_beta), method="CG")



