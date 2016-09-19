## p( x | Ci ) = Normal( mean = mu_i, Cov = Sigma_c )

from numpy import exp, vectorize

sigmoid = lambda z: 1/( 1 + exp(-z) )
sigmoid = vectorize(sigmoid)
