import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from numpy.linalg import inv


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def polyfit(xtrain, ytrain, ztrain, X_c, f):
    #sigma is changed when we apply noise
    #resampling - we get betas variance

    Xtrain = poly.fit_transform(X_c)
    linreg.fit(Xtrain,ztrain)
    Zpredict = linreg.predict(Xtrain)
#variance-covariance matrix, we need only diagonal elements - covcov
    Beta_varcovar = inv(np.dot(Xtrain.T,Xtrain))*((10**(-f))**2)
    Beta_var = Beta_varcovar.diagonal()
#confidence interval estimation with 95% probability    
    conf_int = []
    for i in range(len(Beta_var)):
        conf_int.append(1.96*(Beta_var[i]/len(Xtrain))**(1/2))
        i += 1

    feature_names = poly.get_feature_names()
    lin_coef = linreg.coef_.tolist()
    lin_intercept = linreg.intercept_
    lin_int_coef = []
    for i in range(len(lin_coef[0])):
        if i == 0:
            lin_int_coef.append(lin_intercept[0])
        else:
            lin_int_coef.append(lin_coef[0][i]) 

    str1 = [str(i) for i in feature_names]
    str2 = [str(i) for i in lin_int_coef]
    str3 = [str(i) for i in conf_int]
    Beta_var_degrees = np.stack((str1,str2,str3),axis=-1)
    
    print("\nFeatures/Betas/Conf.intervals:\n", Beta_var_degrees)
    print("\nMSE for Zpredict:", mean_squared_error(ztrain,Zpredict))
    print("R2_score for Zpredict:", r2_score(ztrain,Zpredict))
    

    return;
    

def plot(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface
    surf = ax.plot_surface(x, y, z.reshape(*x.shape), cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-0.4, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if str(z) == str(ztrue):       
        ax.set_title('Franke function (true)')
    elif str(z) == str(zpredict):
        ax.set_title('Franke function (training)')
        
    plt.show()
    return;
    

def LinregResample(k, X_c, xtrain, ytrain, ztrain):
    kfold = KFold(n_splits = k)
    mse_KFold = np.zeros((k))
    R2_KFold = np.zeros((k))
    j = 0
    for train_inds, test_inds in kfold.split(X_c):
            xtrain_cv = X_c[train_inds]
            ztrain_cv = ztrain[train_inds]
            
            xtest_cv = X_c[test_inds]
            ztest_cv = ztrain[test_inds]
            
            Xtrain_fit = poly.fit_transform(xtrain_cv)
            linreg.fit(Xtrain_fit, ztrain_cv)
            
            Xtest_fit = poly.fit_transform(xtest_cv)
            zpred = linreg.predict(Xtest_fit)
            
            mse_KFold[j] = mean_squared_error(zpred,ztest_cv)
            R2_KFold[j] = r2_score(ztest_cv,zpred)  
            
            j += 1
              
    estimated_mse_KFold = np.mean(mse_KFold)
    estimated_R2_KFold = np.mean(R2_KFold)   

    print("\nMSE for Zpredict= %s" % str(estimated_mse_KFold),
          "\nR2_score for Zpredict= %s" % str(estimated_R2_KFold))
    return;
        
    
def RidgeResample(k, lambdas, X_c, xtrain, ytrain, ztrain):
    kfold = KFold(n_splits = k)
    # Perform the cross-validation to estimate MSE
    #divided 5 times in sets

    nlambdas = len(lambdas)
    mse_KFold = np.zeros((nlambdas, k))
    R2_KFold = np.zeros((nlambdas, k))
    
    l = 0
    for lmb in lambdas:
        print("\nLambda:%s" % str(lambdas[l]))
        ridge = Ridge(alpha = lmb)
        j = 0
        for train_inds, test_inds in kfold.split(X_c):
            xtrain_cv = X_c[train_inds]
            ztrain_cv = ztrain[train_inds]
            
            xtest_cv = X_c[test_inds]
            ztest_cv = ztrain[test_inds]
            
            Xtrain_fit = poly.fit_transform(xtrain_cv)[:,1:]
            ridge.fit(Xtrain_fit, ztrain_cv)
            
            Xtest_fit = poly.fit_transform(xtest_cv)[:,1:]
            zpred = ridge.predict(Xtest_fit)
            
            mse_KFold[l,j] = mean_squared_error(zpred,ztest_cv)
            R2_KFold[l,j] = r2_score(ztest_cv,zpred)
            
            j += 1
        estimated_mse_KFold = np.mean(mse_KFold, axis = 1)
        estimated_R2_KFold = np.mean(R2_KFold, axis = 1)
        print("\nMSE for Zpredict(Ridge)= %s" % str(estimated_mse_KFold[l]),
          "\nR2_score for Zpredict(Ridge)= %s" % str(estimated_R2_KFold[l]))
        l += 1
        
    best_mse = np.argmin(estimated_mse_KFold)
    best_lambda = lambdas[best_mse]
    print("\nBest lambda(Ridge):%s\n" % str(best_lambda))    
    return;
    
def LassoResample(k, lambdas, X_c, xtrain, ytrain, ztrain):
    kfold = KFold(n_splits = k)
    nlambdas = len(lambdas)
    mse_KFold = np.zeros((nlambdas, k))
    R2_KFold = np.zeros((nlambdas, k))
    
    l = 0
    for lmb in lambdas:
        print("\nLambda:%s" % str(lambdas[l]))
        lasso = Lasso(alpha = lmb)
        j = 0
        for train_inds, test_inds in kfold.split(X_c):
            xtrain_cv = X_c[train_inds]
            ztrain_cv = ztrain[train_inds]
            
            xtest_cv = X_c[test_inds]
            ztest_cv = ztrain[test_inds]
            
            Xtrain_fit = poly.fit_transform(xtrain_cv)[:,1:]
            lasso.fit(Xtrain_fit, ztrain_cv)
            
            Xtest_fit = poly.fit_transform(xtest_cv)[:,1:]
            zpred = lasso.predict(Xtest_fit)
            
#            mse_KFold[l,j] = np.sum((zpred - ztest_cv)**2)/np.size(zpred)
            mse_KFold[l,j] = mean_squared_error(zpred,ztest_cv)
            R2_KFold[l,j] = r2_score(ztest_cv,zpred)
            
            j += 1
        estimated_mse_KFold = np.mean(mse_KFold, axis = 1)
        estimated_R2_KFold = np.mean(R2_KFold, axis = 1)
        print("\nMSE for Zpredict(Lasso)= %s" % str(estimated_mse_KFold[l]),
          "\nR2_score for Zpredict(Lasso)= %s" % str(estimated_R2_KFold[l]))
        l += 1
        
    best_R2_score = np.argmax(estimated_R2_KFold)
    best_lambda = lambdas[best_R2_score]
    print("\nBest lambda(Lasso):%s\n" % str(best_lambda))    
    return;  
    
# Make data
#uniform distribution creation (100 rows,1 column), numbers in [0,1]
linreg = LinearRegression()
xtrain=np.random.rand(200,1)
ytrain=np.random.rand(200,1)
X_c=np.c_[xtrain,ytrain]
noise=np.random.randn(len(xtrain), 1)

# Initialize a KFold instance
k = 5
##b) Ridge + cross-validation
lambdas = [1e-3, 1e-2, 1e-1, 10]

#for plots
xnew = np.arange(0, 1, 0.1)
ynew = np.arange(0, 1, 0.1)
xnew, ynew = np.meshgrid(xnew,ynew)
ztrue = FrankeFunction(xnew,ynew)

for d in range (3,6):
    print("\nDegree:%s" % str(d))
    poly = PolynomialFeatures(degree=d)
    for f in range(1,4):
        print("\nNoise ~ 0.%s1" %(str("0")*(f-1)))
        ztrain = FrankeFunction(xtrain, ytrain) + (10**(-f))*noise

        if f == 1:
            print("\nOrdinary Least Squares:")
            polyfit(xtrain,ytrain,ztrain,X_c,f)
            print("\nOrdinary Least Squares with %s-fold Cross-Validation:" %str(k))
            LinregResample(k, X_c, xtrain, ytrain, ztrain)  
            
            #ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
            #matrix with 2 columns and 100 rows matching two coordinate points (unravelment)
            #concatenation tranposes and merges
            #y fixed,then running through all X values
            Xnew = poly.fit_transform(np.c_[xnew.ravel(), ynew.ravel()])
            zpredict = linreg.predict(Xnew)

            plot(xnew,ynew,zpredict)
            plot(xnew,ynew,ztrue)
            
            print("\nRidge regression with %s-fold Cross-Validation, noise ~ 0.%s1:" 
                  % (str(k), str("0")*(f-1)))  
            RidgeResample(k, lambdas, X_c, xtrain, ytrain, ztrain)
            print("\nLasso regression with %s-fold Cross-Validation, noise ~ 0.%s1:" % (str(k), str("0")*(f-1)))
            LassoResample(k, lambdas, X_c, xtrain, ytrain, ztrain)
        else:
            print("\nRidge regression with %s-fold Cross-Validation, noise ~ 0.%s1:" 
                  % (str(k), str("0")*(f-1))) 
            RidgeResample(k, lambdas, X_c, xtrain, ytrain, ztrain)
            print("\nLasso regression with %s-fold Cross-Validation, noise ~ 0.%s1:" % (str(k), str("0")*(f-1)))
            LassoResample(k, lambdas, X_c, xtrain, ytrain, ztrain)

    d += 1

