import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # This is needed to define a colormap
from random import random, seed
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
    
n = 100
xin = np.random.rand(n,1) # x and y data values within 0 and 1 with normal distribution 1 
yin = np.random.rand(n,1) 
#print(x.shape)
x,y = np.meshgrid(xin,yin)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures (degree=5)
#XY = poly.fit_transform(np.c_[x.ravel(),y.ravel()])

#a=np.c_[x.ravel()]
#b=np.c_[y.ravel()]
a=x.ravel()
b=y.ravel()

XY = np.c_[np.ones((n*n,1)), a, b, a*a, b*b, a*b, a**3, b**3, a**2*b, a*b**2, a**4, b**4, a**3*b, a**2*b**2, a*b**3, a**5, b**5, a**4*b, a**3*b**2, a**2*b**3, a*b**4, 
                   np.exp(a), np.exp(b), np.exp((a*a)+(b*b)), np.exp((a*a)+b)]

#XY = np.c_[np.ones((n*n,1)), a, b, np.exp(a), np.exp(b), np.exp(a*b), np.exp((a+b)**2), np.exp(a**2), np.exp(b**2), np.exp(a**2+b**2)] 

error = 0.001

z = FrankeFunction(x.reshape(-1,1), y.reshape(-1,1))+error*np.random.randn(n*n,1)

#z = FrankeFunction(np.c_[x.ravel(), y.ravel()]) + error*np.random.randn(n*n,1)

#z = FrankeFunction(x, y) + error*np.random.randn(n*n,1)


from sklearn.linear_model import LinearRegression

clf2 = LinearRegression()
clf2.fit(XY, z)

xinplot = np.linspace(0.0,1.0, 100)
yinplot = np.linspace(0.0,1.0, 100)

xplot, yplot = np.meshgrid(xinplot, yinplot) 

polynew = PolynomialFeatures (degree=5)

c=xplot.ravel()
d=yplot.ravel()

#XYplot = np.c_[np.ones((n*n,1)), c, d, np.exp(c), np.exp(d), np.exp(c*d), np.exp((c+d)**2), np.exp(c**2), np.exp(d**2), np.exp(c**2+d**2)] 

XYplot = np.c_[np.ones((n*n,1)), c, d, c*c, d*d, c*d, c**3, d**3, c**2*d, c*d**2, c**4, d**4, c**3*d, c**2*d**2, c*d**3, c**5, d**5, 
                   c**4*d, c**3*d**2, c**2*d**3, c*d**4, np.exp(c), np.exp(d),
                   np.exp((c*c)+(d*d)), np.exp((c*c)+d)]


#XYplot = polynew.fit_transform(np.c_[xplot.ravel(),yplot.ravel()])


zplot = FrankeFunction(xplot, yplot)

zsklearn = clf2.predict(XYplot) # 1st way zresult


fig = plt.figure()
ax = fig.gca(projection='3d')
print(xplot)

# Plot the surface.
#surf = ax.plot_surface(xplot.reshape(10,10), yplot.reshape(10,10), zsklearn.reshape(10,10), cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

#surf = ax.plot_surface(xplot, yplot, zsklearn, cmap=cm.coolwarm,linewidth=0, antialiased=False)

surf = ax.plot_surface(xplot.reshape(100,100), yplot.reshape(100,100), zsklearn.reshape(100,100), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
                       
#surf = ax.plot_surface(xplot, yplot, zplot, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
# 1st way true curve
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

znew=zplot.reshape(10000,1)
zsknew=zsklearn.reshape(10000,1)

plt.show()

print("Mean squared error (ysklearn):", mean_squared_error(znew, zsknew))
print("R^2 score (ysklearn):", r2_score(znew, zsknew))
