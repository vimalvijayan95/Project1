import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # This is needed to define a colormap
from random import random, seed
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



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
XY = poly.fit_transform(np.c_[x.ravel(),y.ravel()])

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
XYplot = polynew.fit_transform(np.c_[xplot.ravel(),yplot.ravel()])

#zplot = FrankeFunction(xplot, yplot)

zplot = FrankeFunction(xplot.reshape(-1,1), yplot.reshape(-1,1))+error*np.random.randn(n*n,1)

zsklearn = clf2.predict(XYplot) # 1st way zresult


print(xplot)

fig = plt.figure()
ax = fig.gca(projection='3d')

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

plt.show()

znew=zplot.reshape(10000,1)
zsknew=zsklearn.reshape(10000,1)
znoise=z.reshape(10000,1) # with noise and randomly distributed values


print("Mean squared error (zsklearn):", mean_squared_error(znew, zsknew))
print("R^2 score (zsklearn):", r2_score(znew, zsknew))
#print("Mean squared error (zsklearn):", mean_squared_error(znoise, zsknew))
#print("R^2 score (zsklearn):", r2_score(znoise, zsknew)) # you shouldn't plot this one because the points are not ordered




# Ridge and Lasso

lmb_values = [1e-4, 1e-3, 1e-2, 10, 1e2, 1e4]
num_values = len(lmb_values)

ridge= Ridge(alpha=1.0)
ridge.fit(XY, z)
pred_ridge=ridge.predict(XYplot)

#sort_ind = np.argsort(x[:,0])
#sort_ind = np.argsort(y[:,0])

#xplot = x[sort_ind,0]
#yplot = y[sort_ind,0]

#pred_ridge_plot = pred_ridge[sort_ind,:]

pred_ridge_scikit =  np.zeros((n**2,num_values))
for i,lmb in enumerate(lmb_values):
    pred_ridge_scikit[:,i] = (Ridge(alpha=lmb,fit_intercept=False).fit(XY,z).predict(XYplot)).flatten() # fit_intercept=False fordi bias er allerede i X

#plt.figure()

#for i in range(num_values):
#    plt.plot(xplot, yplot, pred_ridge_scikit[sort_ind,i],label='scikit-ridge, lmb=%g'%lmb_values[i])

#plt.plot(x,y,'ro')
#plt.legend()
#plt.title('linear regression using scikit')

#plt.show()

### R2-score of the results
for i in range(num_values):
    print('lambda = %g'%lmb_values[i])
    print('r2 for scikit: %g'%r2_score(znew,pred_ridge_scikit[:,i]))
    



lasso= Lasso(alpha=1.0)
lasso.fit(XY, z)
pred_lasso=lasso.predict(XYplot)

#sort_ind = np.argsort(x[:,0])
#sort_ind = np.argsort(y[:,0])

#xplot = x[sort_ind,0]
#yplot = y[sort_ind,0]

#pred_ridge_plot = pred_ridge[sort_ind,:]

pred_lasso_scikit =  np.zeros((n**2,num_values))
for i,lmb in enumerate(lmb_values):
    pred_lasso_scikit[:,i] = (Lasso(alpha=lmb,fit_intercept=False).fit(XY,z).predict(XYplot)).flatten() # fit_intercept=False fordi bias er allerede i X

#plt.figure()

#for i in range(num_values):
#    plt.plot(xplot, yplot, pred_ridge_scikit[sort_ind,i],label='scikit-ridge, lmb=%g'%lmb_values[i])

#plt.plot(x,y,'ro')
#plt.legend()
#plt.title('linear regression using scikit')

#plt.show()

### R2-score of the results
for i in range(num_values):
    print('lambda = %g'%lmb_values[i])
    print('r2 for scikit: %g'%r2_score(znew,pred_lasso_scikit[:,i]))
    


'''    
    
#Lasso


xplot = x[sort_ind,0]
pred_lasso_plot = pred_lasso[sort_ind,:]

pred_lasso_scikit =  np.zeros((n,num_values))
for i,lmb in enumerate(lmb_values):
    pred_lasso_scikit[:,i] = (Lasso(alpha=lmb,fit_intercept=False).fit(X,y).predict(X)).flatten() # fit_intercept=False fordi bias er allerede i X

plt.figure()

for i in range(num_values):
    plt.plot(xplot,pred_lasso_scikit[sort_ind,i],label='scikit-ridge, lmb=%g'%lmb_values[i])

plt.plot(x,y,'ro')
plt.legend()
plt.title('linear regression using scikit')

plt.show()

### R2-score of the results
for i in range(num_values):
    print('lambda = %g'%lmb_values[i])
    print('r2 for scikit: %g'%r2_score(y,pred_lasso_scikit[:,i]))
'''

'''
# cross validation 

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
#poly = PolynomialFeatures(degree = 6)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))

i = 0
#xflatten = 0
for lmb in lambdas:
#    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(x,y):
        xtrain = x[train_inds]
        ytrain = y[train_inds]
        ztrain = z[train_inds]


        xtest = x[test_inds]
        ytest = y[test_inds]
        ztest = z[test_inds]

        #Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        
        XYtrain = poly.fit_transform(np.c_[xtrain.ravel(),ytrain.ravel()])
        
        #ridge.fit(Xtrain, ytrain[:, np.newaxis])
        clf2 = LinearRegression()
        clf2.fit(XYtrain, ztrain)

        #Xtest = poly.fit_transform(xtest[:, np.newaxis])
        
        XYtest = poly.fit_transform(np.c_[xtest.ravel(),ytest.ravel()])
        
        #ypred = ridge.predict(Xtest)
        zpred = clf2.predict(XYtest)

        scores_KFold[i,j] = np.sum((zpred - ztest)**2)/np.size(zpred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
#i = 0
#for lmb in lambdas:
#    ridge = Ridge(alpha = lmb)

#    XYpred = poly.fit_transform(XY)
#    estimated_mse_folds = cross_val_score(XY, z, scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
#    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

#    i += 1

## Plot and compare the slightly different ways to perform cross-validation

plt.figure()

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()

'''