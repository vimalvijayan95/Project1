# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:59:02 2019

@author: andrea
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge

#RIDGE method
from sklearn import linear_model
alphav=np.array([0.001,0.01,0.1,1,10,100])
np.random.seed(3155)
degrees=np.arange(2,6,1)
#mse_cv= np.zeros((k,len(degrees)))
#r2_cv= np.zeros((k,len(degrees)))
mean_mse_cv= np.zeros((len(alphav),len(degrees)))
mean_r2_cv= np.zeros((len(alphav),len(degrees)))

#Now we apply the method to each polinomial degree (same as in 1a)):

for degree in degrees:
 n=100
 nmax=1
 error=0.1
 
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

 from sklearn.metrics import mean_squared_error
 from sklearn.metrics import r2_score
 from sklearn.model_selection import cross_val_score
#In this case we fit with Ridge for EACH value of alpha
 i=0
 for v in alphav:
     ridge = Ridge(alpha = v)
       
     #ridge.fit(XY,z.reshape(-1,1))
     #This vector contains all the mse values for each of the k-fold set.
     mse_cv= cross_val_score(ridge, XY, z.ravel(), scoring='neg_mean_squared_error', cv=k).reshape(-1,1)
     #In this matrix we save the mean of the above values (overall mse estimation) for each polinomial degree
     mean_mse_cv[i,degree-2] = np.mean(-mse_cv)
     
     r2_cv = cross_val_score(ridge, XY, z.ravel(), scoring='r2', cv=k).reshape(-1,1)
     mean_r2_cv[i,degree-2] = np.mean(r2_cv)
     i += 1



