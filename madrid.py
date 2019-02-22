import matplotlib.pyplot as plt
from matplotlib import cm
from imageio import imread
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

def model_fit(xtrain, ytrain, ztrain, X_c, model):
    # sigma is changed when we apply noise
    # resampling - we get betas variance
    if str(model) == str(linreg):
        Xtrain = poly.fit_transform(X_c)
    else:
        Xtrain = poly.fit_transform(X_c)[:,1:]
    model.fit(Xtrain, ztrain)
    Zpredict = model.predict(Xtrain)
    MSE = mean_squared_error(ztrain, Zpredict)
    r2 = r2_score(ztrain, Zpredict) 
    print("\nMSE for Zpredict = %s" % str(MSE))
    print("\nR2 score for Zpredict = %s" % str(r2))


def plot(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Plot the surface
    surf = ax.plot_surface(
        x, y, z.reshape(*x.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Customize the z axis.
    ax.set_zlim(-1, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if str(z) == str(ztrue):
        ax.set_title("Franke function (true)")
    elif str(z) == str(zpredict):
        ax.set_title("Franke function (training)")

    plt.show()
    return

def linreg_resample(X_c, xtrain, ytrain, ztrain):
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

        mse_KFold[j] = mean_squared_error(ztest_cv, zpred)
        R2_KFold[j] = r2_score(ztest_cv, zpred)

        j += 1

    estimated_mse_KFold = np.mean(mse_KFold)
    estimated_R2_KFold = np.mean(R2_KFold)

    print(
        "\nMSE for Zpredict= %s" % str(estimated_mse_KFold),
        "\nR2_score for Zpredict= %s" % str(estimated_R2_KFold),
    )

#loop over different values of lambda is left in case there's need to check it again
#that's why this function is not combined with the previous function as it would
#require many "if" conditions
    
def model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, model):

    nlambdas = len(lambdas)
    mse_KFold = np.zeros((nlambdas, k))
    R2_KFold = np.zeros((nlambdas, k))

    l = 0
    for lmb in lambdas:
        print("\nLamda:%s" % str(lambdas[l]))
        model_type = model(alpha=lmb)
        j = 0
        for train_inds, test_inds in kfold.split(X_c):
            xtrain_cv = X_c[train_inds]
            ztrain_cv = ztrain[train_inds]

            xtest_cv = X_c[test_inds]
            ztest_cv = ztrain[test_inds]

            Xtrain_fit = poly.fit_transform(xtrain_cv)[:, 1:]
            model_type.fit(Xtrain_fit, ztrain_cv)

            Xtest_fit = poly.fit_transform(xtest_cv)[:, 1:]
            zpred = model_type.predict(Xtest_fit)

            mse_KFold[l, j] = mean_squared_error(ztest_cv, zpred)
            R2_KFold[l, j] = r2_score(ztest_cv, zpred)

            j += 1
             
        estimated_mse_KFold = np.mean(mse_KFold, axis=1)
        estimated_R2_KFold = np.mean(R2_KFold, axis=1)
        print(
            "\nMSE for Zpredict (%s) = %s"
            % (str(model.__name__), str(estimated_mse_KFold[l])),
            "\nR2 score for Zpredict (%s) = %s"
            % (str(model.__name__), str(estimated_R2_KFold[l])),
        )
        l += 1

# Make data
np.random.seed(42)
xtrain = np.random.rand(3601, 1)
ytrain = np.random.rand(3601, 1)
ztrain = imread('SRTM_data_Madrid.tif')
X_c = np.c_[xtrain, ytrain]

d = 100
k = 5
lambdas = [1e-6]

linreg = LinearRegression()
kfold = KFold(n_splits=k)
ridge = Ridge()
lasso = Lasso()
poly = PolynomialFeatures(degree=d)

print("\nDegree:%s" % str(d))
print("\nLambda:%s" % str(lambdas[0]))

#print("\nOrdinary Least Squares (no resampling):")
#model_fit(xtrain, ytrain, ztrain, X_c, linreg)

#print("\nOrdinary Least Squares with %s-fold Cross-Validation:" % str(k))
#linreg_resample(X_c, xtrain, ytrain, ztrain)

#print("\nRidge (no resampling):")
#model_fit(xtrain, ytrain, ztrain, X_c, ridge)

#print("\nRidge regression with %s-fold Cross-Validation:"
#    % str(k))
#model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, ridge)

print("\Lasso (no resampling):")
model_fit(xtrain, ytrain, ztrain, X_c, lasso)

#print("\nLasso regression with %s-fold Cross-Validation:"
#    % str(k))
#model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, lasso)
