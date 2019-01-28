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
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from tabulate import tabulate

# Make data.
#Fix random number for every run. 
np.random.seed(3155)
degrees=np.arange(2,6,1)
k = 5
mse =np.zeros(len(degrees))
r2=np.zeros(len(degrees))
mse_cv= np.zeros((k,len(degrees)))
r2_cv= np.zeros((k,len(degrees)))
mean_mse_cv= np.zeros(len(degrees))
mean_r2_cv= np.zeros(len(degrees))

#We are going to plot our results in a table
from prettytable import PrettyTable
t = PrettyTable()
t.field_names = ["degree", "mse", "r2"]

for degree in degrees:
 n=100
 nmax=1
 error=0.1


#x_unsort = np.random.rand(n,nmax);
#y_unsort = np.random.rand(n,nmax);

#Sorten the data:
#ind_sort_x = np.argsort(x_unsort)
#ind_sort_y = np.argsort(y_unsort)

#x = x_unsort[ind_sort_x]
#y = y_unsort[ind_sort_y]

 x = np.random.rand(n,nmax);
 y = np.random.rand(n,nmax);
 xx, yy = np.meshgrid(x, y)

#Generate the values of the function:
 def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

 z = FrankeFunction(xx, yy)+error*np.random.randn(n,n)

#Generate the design matrix:
 poly = PolynomialFeatures(degree)
 XY=poly.fit_transform(np.c_[xx.ravel(), yy.ravel()])

## Fit the data with Linear Regression. 
 linreg = LinearRegression()
 linreg.fit(XY, z.ravel())


#Now we compare our results with the ones from the original function. 
#To do that we define two new sets of points: 
 nfit = 100
 xplot = np.linspace(0.0,1.0, n)
 yplot = np.linspace(0.0,1.0, n)
 xxplot, yyplot = np.meshgrid(xplot, yplot)
 XYplot=poly.fit_transform(np.c_[xxplot.ravel(), yyplot.ravel()])

#Determine the predicted and true values
 zpredict = linreg.predict(XYplot)
 ztrue= FrankeFunction(xxplot, yyplot)

#Make the different plots:

 fig = plt.figure()
 ax = fig.gca(projection='3d')
#ax = fig.add_subplot(1, 2, 1, projection='3d')
 surf = ax.plot_surface(xxplot, yyplot, ztrue, cmap=cm.viridis, linewidth=0, antialiased=False)
 fig.colorbar(surf, shrink=0.5, aspect=5)
 plt.title('Franke' )

 plt.show()

 fig = plt.figure()
#ax = fig.add_subplot(1, 2, 2, projection='3d')
 ax = fig.gca(projection='3d')
 surf = ax.plot_surface(xxplot, yyplot, zpredict.reshape(n,n), cmap=cm.viridis, linewidth=0, antialiased=False)
 fig.colorbar(surf, shrink=0.5, aspect=5)
 plt.title('Fitted Franke degree=%d' %degree)

 plt.show()

 from sklearn.metrics import mean_squared_error
 mse[degree-2]= mean_squared_error(ztrue.reshape(-1,1), zpredict)
 #print(mse[degree-2])
 #print("Mean squared error (ysklearn):", mean_squared_error(ztrue.reshape(-1,1), zpredict))
 from sklearn.metrics import r2_score
 #print("R^2 score (ysklearn):", r2_score(ztrue.reshape(-1,1), zpredict.ravel()))
 r2[degree-2]=r2_score(ztrue.reshape(-1,1), zpredict)
 
 #Perform cross validation for each data set:
 #------------------------------------------
 from sklearn.model_selection import cross_val_score
 #from sklearn.model_selection import KFold
 #kfold = KFold(n_splits = k)
 mse_cv[:,degree-2] = cross_val_score(linreg, XY, z.ravel(), scoring='neg_mean_squared_error', cv=k)
 r2_cv[:,degree-2] = cross_val_score(linreg, XY, z.ravel(), scoring='r2', cv=k)

  #For each degree determine the mean value of the mse value .
 mean_mse_cv[degree-2] = np.mean(-mse_cv[:,degree-2])
 mean_r2_cv[degree-2] = np.mean(r2_cv[:,degree-2])
 #print('degree: %d', mean_mse_cv[degree-2] %degree)
 t.add_row([degree, mean_mse_cv[degree-2], mean_r2_cv[degree-2]])

print()
print()
print(t)
 















