# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:22:02 2019

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
from sklearn.linear_model import Lasso

#RIDGE method
alphav=np.array([0.001,0.01,0.1,1,10,100])
np.random.seed(3155)
degrees=np.arange(2,6,1)
#mse_cv= np.zeros((k,len(degrees)))
#r2_cv= np.zeros((k,len(degrees)))
mean_mse_cv= np.zeros((len(alphav),len(degrees)))
mean_mse_bs= np.zeros((len(alphav),len(degrees)))
mean_r2_cv= np.zeros((len(alphav),len(degrees)))
best_mse_cv=np.zeros(len(degrees))
vbest_mse_cv=np.zeros(len(degrees))
best_r2_cv = np.zeros(len(degrees))
vbest_r2_cv = np.zeros(len(degrees))
 


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
 
 def bootstrap(n, n_bootstrap, n_train=1):
    idx=list(range(n))
    n_train = int(n_train*n)
    for _ in range(n_bootstrap):
        train_index=np. random.choice (idx, replace=True, size=n_train)
        test_index=list(set(idx)-set(train_index))
        yield train_index, test_index
 
    #bs = bootstrap(n,5, 0.8)
    
 #from sklearn import cross_validation
 #bs = cross_validation.Bootstrap(n, n_bootstraps=n, n_train=8000, n_test=2000, random_state=0)
 i=0
 k=5
 from prettytable import PrettyTable
 t = PrettyTable()
 t.field_names = ["degree", "landa_mse", "best mse", "landa_r2", "best r2"]
     
 for v in alphav:
     lasso = Lasso(alpha = v)
       
     #ridge.fit(XY,z.reshape(-1,1))
     #This vector contains all the mse values for each of the k-fold set.
     mse_cv= cross_val_score(lasso, XY, z.ravel(), scoring='neg_mean_squared_error', cv=k).reshape(-1,1)
     #In this matrix we save the mean of the above values (overall mse estimation) for each polinomial degree
     mean_mse_cv[i,degree-2] = np.mean(-mse_cv)
     mse_bs= cross_val_score(lasso, XY, z.ravel(), scoring='neg_mean_squared_error', cv=bootstrap(n=100, n_bootstrap=5)).reshape(-1,1)
     mean_mse_bs[i,degree-2] = np.mean(-mse_bs)

     
     r2_cv = cross_val_score(lasso, XY, z.ravel(), scoring='r2', cv=k).reshape(-1,1)
     mean_r2_cv[i,degree-2] = np.mean(r2_cv)
     i += 1

#Now I determine he best mse values obtained for each polinomial degree and the corresponding landa:
 best_mse_cv[degree-2] = np.min(mean_mse_cv[:,degree-2])
 vbest_mse_cv[degree-2] = alphav[np.argmin(mean_mse_cv[:,degree-2])]
 best_r2_cv[degree-2] = np.max(mean_r2_cv[:,degree-2])
 vbest_r2_cv[degree-2] = alphav[np.argmax(mean_r2_cv[:,degree-2])]
 
 
 t.add_row([degree, vbest_mse_cv[degree-2], best_mse_cv[degree-2], vbest_r2_cv[degree-2], best_r2_cv[degree-2]])
     
 b='    '
 #print(5*b,        'degree=%d' %degree)    
 print(t)
     