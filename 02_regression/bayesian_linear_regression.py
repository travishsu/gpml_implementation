from numpy import exp, dot, linspace, power, eye, array, sin, cos, pi, diag, log, sum, sqrt
from numpy.linalg import norm, solve, inv, svd, cholesky
from matplotlib.pylab import plt, scatter

class GaussianProcessRegression:
    def __init__( self, X, y, noise_level ):
        self.X = X
        self.y = y
        self.noise_level = noise_level

        self.x_feature = array([self.feature(xi) for xi in self.X])
        self.L = cholesky( self.kerMat(self.X, self.X) + self.noise_level * eye(self.X.size) )
        self.w = solve( self.L.T, solve(  self.L,  self.y ) )
        self.log_marginal_likelihood = - dot(self.y, self.w)/2 - sum( log( diag( self.L ) ) ) - self.X.size*log(2*pi)/2

    def covariance( self, x_1, x_2 ):
        # Squared Exponential covariance function
        l = 1 # length-scale parameter
        return exp( - power(norm(x_1 - x_2)/l, 2  ) * 0.5 )

    def feature( self, x ):
        return array( [ 1, (x-1)*(x-5)/2, x] )
        #return array( [1, sin(x), cos(x)] )

    def kerMat( self, xlist, ylist ):
        ker = []
        for x in xlist:
            ker.append( [self.covariance( self.feature(x), self.feature(y) ) for y in ylist] )
        return array(ker)

    #def fit(self, x, y, noise_level):
    #    L = cholesky( kerMat(x, x) + noise_level * eye(x.size) )
    #    w = solve( L.T, solve( L, y ) )
    #    log_marginal_likelihood = - dot(y, w)/2 - sum( log( diag( L ) ) ) - x.size*log(2*pi)/2
    #    return w, fit_score

    def predict(self, xtest):
        self.xtest = xtest;
        self.f = dot( self.kerMat(self.xtest, self.X), self.w )
        #v = solve(self.L, self.kerMat(self.X, self.xtest))
        #print v.shape
        #self.var = self.kerMat( self.xtest, self.X ) - dot(v.T, v)
        return self.f#, self.var

# Generating Training Data
x = linspace(0, 2*pi, 50)
y = sin(x)

# Modeling: Parameters estimation
noise_level = 0.01
model = GaussianProcessRegression(x, y, noise_level)
# Prediction
x_star = linspace(0, 2*pi, 1000)
y_star = model.predict(x_star)

# Plotting
plt.plot(x_star, y_star)
#plt.plot(x_star, y_star+100*diag(var_star))
#plt.plot(x_star, y_star-100*diag(var_star))
plt.plot(x_star, sin(x_star))
for i in range(x.size):
    plt.scatter( x[i], y[i] )
plt.show()
