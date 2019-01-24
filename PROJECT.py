# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:51:24 2019

@author: andrea
"""

#PROJECT 1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Make data.
n=100
nmax=1
error=0.1
degree=2
x = np.random.rand(n,nmax);
y = np.random.rand(n,nmax);
xx, yy = np.meshgrid(x, y)

V=np.c_[xx.ravel(), yy.ravel()]
poly = PolynomialFeatures(degree)
X=poly.fit_transform(V)

#Ahora que ya tengo mi matriz X con los valores para ajustar puedo usar
#el método de rigresión linear


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Generamos los valores pero añadiendo error


#Matriz con los funciones de mi polinomio:
z = FrankeFunction(xx.ravel(), yy.ravel())+error*np.random.randn(n,xmax);
H=X.T @ X
beta=(np.linalg.inv(H)) @ X.T @ z

#Now we compare our results with the ones from the original function. To do that we define
#two sets of points: 
nfit = 100
xplot = np.linspace(0.0,1.0, num=nfit)
yplot = np.linspace(0.0,1.0, num=nfit)
xxplot, yyplot = np.meshgrid(xplot.reshape((100,1)), yplot.reshape((100,1)))

Vplot=np.c_[xxplot.ravel(), yyplot.ravel()]
poly = PolynomialFeatures(degree)
Xplot=poly.fit_transform(Vplot)


#Vplot=np.c_[xplot,yplot]
#Xplot=poly.fit_transform(Vplot)


#Creamos los valores que nos predice. 

zpredict = Xplot.dot(beta)
xplot, yplot = np.meshgrid(xplot,yplot)
ztrue= FrankeFunction(xplot, yplot)
#Ahora tenemos que hacer 3 PLOTS:
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = plt.axes(projection='3d')

#x, y = np.meshgrid(x,y)

ax.scatter3D(x, y, z, s=5, color='r',marker='o')

#surf = ax.scatter(x, y, z, cmap=cm.coolwarm,
 #               linewidth=0, antialiased=False, label="$z_{\mathrm{Franzkie}}$")
surf = ax.plot_surface(xplot, yplot, zpredict, cmap=cm.coolwarm,
 linewidth=0, antialiased=False, label="$z_{\mathrm{predict}}$")

#surf = ax.plot_surface(xplot, yplot, ztrue, cmap=cm.viridis,
 # linewidth=0, antialiased=False, label="$z_{\mathrm{predict}}$")

#ax.contour3D(x, y, z, 50, cmap='binary')                      
#ax.scatter3D(xplot, yplot, ztrue, s=5, color='r',marker='o')


#ax.contour3D(xplot, yplot, zpredict, 50, cmap='binary')
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()







