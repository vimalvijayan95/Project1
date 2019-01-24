from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x=np.random.rand(100,1)
y=np.random.rand(100,1)
#x, y = np.meshgrid(x,y)



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.4, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()

#Fit the power of 3
poly = PolynomialFeatures (degree=3)
X=np.c_[x,y]
#print(X)
Xplot = poly.fit_transform(X)
linreg = LinearRegression()
linreg.fit(Xplot,z)

xnew = np.arange(0, 1, 0.05)
ynew = np.arange(0, 1, 0.05)
xnew, ynew = np.meshgrid(xnew,ynew)

#ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
Xnew = np.c_[xnew.ravel(), ynew.ravel()]
Zpredict = linreg.predict(poly.fit_transform(Xnew))

#print(Zpredict.shape)

#reshape - to make (400,1) be (20,20) corresponding to (xnew,ynew)
surf = ax.plot_surface(xnew, ynew, Zpredict.reshape(20,20), cmap=cm.coolwarm, linewidth=0, antialiased=False)

