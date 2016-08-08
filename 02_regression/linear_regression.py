import numpy as np
from numpy import exp, dot, linspace, power
from numpy.linalg import norm, solve, inv, svd
from matplotlib.pylab import plot, figure, show, scatter

def covariance( x_1, x_2 ):
    # Square Exponential Kernel
    return exp( - power(norm(x_1 - x_2), 1.5) * 0.5 )

def feature_space(xlist):
    # Polynomial expansion
    fx = []
    for xx in xlist:
        fx.append( [1, xx, xx*xx, xx*xx*xx] )
    return np.array( fx ).T
def feature_space_test(x, xlist):
    return np.array( [1, x, x*x, x*x*x] )

#def feature_space(xlist):
#    fx = []
#    for x_i in xlist:
#        fx.append( [ covariance( x_i, x_j ) for x_j in xlist ] )
#    return np.array(fx)
#def feature_space_test(x, xlist):
#    return np.array( [covariance(x, x_i) for x_i in xlist] )

def covMat(xlist):
    N = xlist.shape[0]
    C = np.eye( N )
    for i in range(N):
        for j in range(N):
            C[ i, j ] = covariance( xlist[i], xlist[j] )
    return C

x = np.array( [1,2,3,4,5] )
y = np.sin(x)
#feature_x = np.array( [feature_space(x_i) for x_i in x] ).T
feature_x = feature_space(x)
C = covMat( feature_x )

kernel = dot( feature_x.T, dot( C, feature_x ) )

# prediction
x_star = linspace(1, 5, 1000)
y_star = np.zeros(x_star.size)

sigma = 0.01
w = dot( dot( C, feature_x ), solve( kernel + sigma*sigma*np.eye(kernel.shape[0]), y ) )
for i in range(x_star.size):
    y_star[i] = dot( feature_space_test(x_star[i], x), w )

figure()
plot(x_star, y_star)
plot(x_star, np.sin(x_star))
for i in range(x.size):
    scatter( x[i], y[i] )
show()
