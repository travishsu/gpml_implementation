from numpy import exp, dot, linspace, power, eye, array, sin, cos
from numpy.linalg import norm, solve, inv, svd
from matplotlib.pylab import plot, figure, show, scatter

def covariance( x_1, x_2 ):
    # Square Exponential Kernel
    return exp( - power(norm(x_1 - x_2), 2  ) * 0.5 )

def feature(x):
    return array( [ x ] )
    #return array( [1, sin(x), cos(x)] )

def kerMat(xlist, ylist):
    ker = []
    for x in xlist:
        ker.append( [covariance( feature(x), feature(y) ) for y in ylist] )
    return array(ker)

# Generating Training Data
x = array( [1,2,3,4,5] )
y = sin(x)

# Modeling: Parametes estimation
sigma = 0.0001
w = solve( kerMat(x, x) + sigma*sigma*eye(x.size), y )

# Prediction
x_star = linspace(1, 5, 1000)
y_star = dot( kerMat(x_star, x), w )

# Plotting
figure()
plot(x_star, y_star)
plot(x_star, sin(x_star))
for i in range(x.size):
    scatter( x[i], y[i] )
show()
