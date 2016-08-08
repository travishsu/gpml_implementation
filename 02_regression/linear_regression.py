import numpy as np
from numpy import exp, dot, linspace, power, eye
from numpy.linalg import norm, solve, inv, svd
from matplotlib.pylab import plot, figure, show, scatter

def covariance( x_1, x_2 ):
    # Square Exponential Kernel
    return exp( - power(norm(x_1 - x_2), 3) * 0.5 )

def feature(x):
    #return np.array( [1, x, x*x/10, x*x*x/100, x*x*x*x/1000, x*x*x*x*x/10000] )
    return np.array( [1, np.sin(x), np.cos(x)] )

def kerMat(xlist, ylist):
    ker = []
    for x in xlist:
        ker.append( [covariance( feature(x), feature(y) ) for y in ylist] )
    return np.array(ker)

x = np.array( [1,2,3,4,5] )
y = np.sin(x)

# prediction
x_star = linspace(1, 5, 1000)

sigma = 0.01
w = solve( kerMat(x, x) + sigma*sigma*eye(x.size), y )

y_star = dot( kerMat(x_star, x), w )

figure()
plot(x_star, y_star)
plot(x_star, np.sin(x_star))
for i in range(x.size):
    scatter( x[i], y[i] )
show()
