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

def kerMat(xlist, ylist):
    ker = []
    for x in xlist:
        ker.append( [covariance( feature(x), feature(y) ) for y in ylist] )
    return array(ker)

def fit(x, y, noise_level):
    L = cholesky( kerMat(x, x) + noise_level * eye(x.size) )
    w = solve( L.T, solve( L, y ) )
    log_marginal_likelihood = - dot(y, w)/2 - sum( log( diag( L ) ) ) - x.size*log(2*pi)/2
    return w, fit_score

def predict(x, X, w, L):
    f = dot( kerMat(x, X), w )
    v = solve(L, kerMat(x, X))
    var = kerMat( x, x ) - dot(v.T, v)
    return f, var

# Generating Training Data
x = linspace(0, 2*pi, 50)
y = sin(x)

# Modeling: Parameters estimation
noise_level = 0.01
w = solve( kerMat(x, x) + noise_level * eye(x.size), y )
log_marginal_likelihood = - dot(y, w)/2 - sum( log( diag( cholesky(kerMat(x, x) + noise_level * eye(x.size)) ) ) ) - x.size*log(2*pi)/2
print log_marginal_likelihood
# Prediction
x_star = linspace(0, 2*pi, 1000)
y_star = dot( kerMat(x_star, x), w )
var_star = kerMat( x_star, x_star ) - dot( kerMat( x_star, x ), solve( kerMat( x, x ), kerMat( x, x_star ) ) )

# Plotting
plt.plot(x_star, y_star)
#plt.plot(x_star, y_star+100*diag(var_star))
#plt.plot(x_star, y_star-100*diag(var_star))
plt.plot(x_star, sin(x_star))
for i in range(x.size):
    plt.scatter( x[i], y[i] )
plt.show()
