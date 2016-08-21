from numpy import exp, dot, linspace, power, eye, array, sin, cos, pi, diag, log, sum, sqrt
from numpy.linalg import norm, solve, inv, svd, cholesky
from matplotlib.pylab import plt, scatter

def covariance( x_1, x_2 ):
    # Squared Exponential covariance function
    l = 0.064 # length-scale parameter
    return exp( - power(norm(x_1 - x_2)/l, 2  ) * 0.5 )

def feature(x):
    return array( [ 1, (x-1)*(x-5)/2, x] )
    #return array( [1, sin(x), cos(x)] )

def equivKer(x, xlist):
    x_feature = array( [feature(xi) for xi in xlist] )
    ker = array( [covariance(feature(x), x_i) for x_i in x_feature] )
    weighted_ker = ker / sum(ker)
    return weighted_ker

# Generating Training Data
x = linspace(0, 2*pi, 50)
y = sin(x)

# Modeling: Parameters estimation
noise_level = 0

# Prediction
x_star = linspace(0, 2*pi, 1000)
y_star = linspace(0, 2*pi, 1000)
for i in range(1000):
    y_star[i] = dot( equivKer(x_star[i], x), y )

# Plotting
plt.plot(x_star, y_star)
#plt.plot(x_star, y_star+100*diag(var_star))
#plt.plot(x_star, y_star-100*diag(var_star))
plt.plot(x_star, sin(x_star))
for i in range(x.size):
    plt.scatter( x[i], y[i] )
plt.show()
