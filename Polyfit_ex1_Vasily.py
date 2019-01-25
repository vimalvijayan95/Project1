from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



# Make data
#uniform distribution creation (100 rows,1 column), numbers in [0,1]
xtrain=np.random.rand(100,1)
ytrain=np.random.rand(100,1)
xtrain, ytrain = np.meshgrid(xtrain,ytrain)
xtrain, ytrain = xtrain.reshape(-1,1), ytrain.reshape(-1,1)
ztrain = FrankeFunction(xtrain, ytrain) + 0.1*np.random.randn(len(xtrain), 1)

#degree 3
poly = PolynomialFeatures(degree=3)
Xtrain = poly.fit_transform(np.c_[xtrain,ytrain])
linreg = LinearRegression()
linreg.fit(Xtrain,ztrain)

xnew = np.arange(0, 1, 0.1)
ynew = np.arange(0, 1, 0.1)
xnew, ynew = np.meshgrid(xnew,ynew)
#ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
#matrix with 2 columns and 100 rows matching two coordinate points (unravelment)
#concatenation tranposes and merges
Znew = poly.fit_transform(np.c_[xnew.ravel(), ynew.ravel()])
Zpredict = linreg.predict(Znew)
Ztrue = FrankeFunction(xnew, ynew).reshape(-1, 1)

#Zpredict plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
print('Polynomial fit plot (prediction)')
surf = ax.plot_surface(xnew, ynew, Ztrue.reshape(*xnew.shape), cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.4, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#Ztrue plot

fig = plt.figure()
ax = fig.gca(projection='3d')
print('Franke function plot')
surf = ax.plot_surface(xnew, ynew, Zpredict.reshape(*xnew.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.4, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

print("MSE for Zpredict_degree3:", mean_squared_error(Ztrue,Zpredict))
print("R2score for Zpredict_degree3:", r2_score(Ztrue,Zpredict))

#degree 4
poly = PolynomialFeatures(degree=4)
Xtrain = poly.fit_transform(np.c_[xtrain,ytrain])
linreg = LinearRegression()
linreg.fit(Xtrain,ztrain)

xnew = np.arange(0, 1, 0.1)
ynew = np.arange(0, 1, 0.1)
xnew, ynew = np.meshgrid(xnew,ynew)
#ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
#matrix with 2 columns and 100 rows matching two coordinate points (unravelment)
#concatenation tranposes and merges
Znew = poly.fit_transform(np.c_[xnew.ravel(), ynew.ravel()])
Zpredict = linreg.predict(Znew)
#Zpredict plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
print('Polynomial fit plot (prediction)')
surf = ax.plot_surface(xnew, ynew, Ztrue.reshape(*xnew.shape), cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.4, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
print("MSE for Zpredict_degree4:", mean_squared_error(Ztrue,Zpredict))
print("R2score for Zpredict_degree4:", r2_score(Ztrue,Zpredict))

#degree5
poly = PolynomialFeatures(degree=5)
Xtrain = poly.fit_transform(np.c_[xtrain,ytrain])
linreg = LinearRegression()
linreg.fit(Xtrain,ztrain)

xnew = np.arange(0, 1, 0.1)
ynew = np.arange(0, 1, 0.1)
xnew, ynew = np.meshgrid(xnew,ynew)
#ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
#matrix with 2 columns and 100 rows matching two coordinate points (unravelment)
#concatenation tranposes and merges
Znew = poly.fit_transform(np.c_[xnew.ravel(), ynew.ravel()])
Zpredict = linreg.predict(Znew)
#Zpredict plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
print('Polynomial fit plot (prediction)')
surf = ax.plot_surface(xnew, ynew, Ztrue.reshape(*xnew.shape), cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.4, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
print("MSE for Zpredict_degree5:", mean_squared_error(Ztrue,Zpredict))
print("R2score for Zpredict_degree5:", r2_score(Ztrue,Zpredict))