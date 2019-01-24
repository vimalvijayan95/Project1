# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:29:59 2019

@author: andrea
"""
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import Lasso
#from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


n = 100 # 1st way = data points
n_x = 100 # 2nd way = data points
n_y = 100 # 2nd way = data points



error = 0.1

#x = np.random.rand(n,1) # 1st way x data values
#y = np.random.rand(n,1) # 1st way x data values


nfit=100
# Make data.
xplot = np.arange(0, 1, 0.05) # 1st way grid
yplot = np.arange(0, 1, 0.05) # 1st way grid
xplot, yplot = np.meshgrid(xplot,yplot)

#sort_inds_xplot = np.argsort(x) # 2nd way for grid points 
#sort_inds_yplot = np.argsort(y) # 2nd way for grid points

#xplot = x[sort_inds_xplot] # 2nd way for grid
#yplot = y[sort_inds_yplot] # 2nd way for grid 



#x = np.random.rand(n_x) #2nd way x data values
XY = []
for x in np.random.rand(n_x):
    for y in np.random.randn(n_y): #2nd way y data values
        XY.append([1, x, y, x*x, x*y, y*y])
XY = np.asarray(XY)

x, y = XY[:, [1,2]].T


z = FrankeFunction(x.reshape(-1,1), y.reshape(-1,1))+error*np.random.randn(n*n,1)

#    XY = np.c_[np.ones(((n_x*n_y),1)), x, y, x*x, y*y]


from sklearn.linear_model import LinearRegression

clf2 = LinearRegression()
clf2.fit(XY, z)

#ols = linear_model.LinearRegression()
#ols.fit(X_train, y_train)


zplot=FrankeFunction(xplot, yplot) # 1st way ztrue

XYplot = np.c_[np.ones(((n_x*n_y),1)), xplot, xplot**2] # 1st way XYplot

zsklearn = clf2.predict(XYplot) # 1st way zresult



fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(xplot, yplot, zsklearn, cmap=cm.coolwarm,
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

plt.show()

