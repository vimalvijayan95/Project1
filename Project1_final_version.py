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

# Functions
def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4

def poly_fit(xtrain, ytrain, ztrain, X_c, f):
    # sigma is changed when we apply noise
    # resampling - we get betas variance

    Xtrain = poly.fit_transform(X_c)
    linreg.fit(Xtrain, ztrain)
    Zpredict = linreg.predict(Xtrain)
    # variance-covariance matrix, we need only diagonal elements - covcov
    Beta_varcovar = inv(np.dot(Xtrain.T, Xtrain)) * ((10 ** (-f)) ** 2)
    Beta_var = Beta_varcovar.diagonal()
    # confidence interval estimation with 95% probability
    conf_int = []
    conf_int = [1.96 * (beta / len(Xtrain)) ** 0.5 for beta in Beta_var]

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
    Beta_var_degrees = np.stack((str1, str2, str3), axis=-1)

    print("\nFeatures/Betas/Conf.intervals:\n", Beta_var_degrees)
    print(
        "\nMSE for Zpredict (without resampling):", mean_squared_error(ztrain, Zpredict)
    )
    print("R2_score for Zpredict (without resampling):", r2_score(ztrain, Zpredict))

def plot(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Plot the surface
    surf = ax.plot_surface(
        x, y, z.reshape(*x.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    # Customize the z axis.
    ax.set_zlim(-0.4, 1.40)
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

    best_mse = np.argmax(estimated_R2_KFold)
    best_lambda = lambdas[best_mse]
    print("\nBest lambda (%s):%s\n" % (str(model.__name__), str(best_lambda)))

# Make data
# uniform distribution creation (100 rows,1 column), numbers in [0,1]
xtrain = np.random.rand(200, 1)
ytrain = np.random.rand(200, 1)
X_c = np.c_[xtrain, ytrain]
noise = np.random.randn(len(xtrain), 1)

k = 5
lambdas = [1e-5, 1e-3, 1e-1, 10]

linreg = LinearRegression()
kfold = KFold(n_splits=k)

# for plots
xnew = np.linspace(0, 1, 10)
ynew = np.linspace(0, 1, 10)
xnew, ynew = np.meshgrid(xnew, ynew)
ztrue = franke_function(xnew, ynew)

for d in range(3, 6):
    poly = PolynomialFeatures(degree=d)
    for f in range(0, 3):
        ztrain = franke_function(xtrain, ytrain) + (10 ** (-f)) * noise
        print("\nDegree:%s ,sigma(noise) = %s" % (str(d),str(float(10 ** (-f)))))
        if f == 0 or f == 1:
            print("\nOrdinary Least Squares:")
            poly_fit(xtrain, ytrain, ztrain, X_c, f)
            print("\nOrdinary Least Squares with %s-fold Cross-Validation:" % str(k))
            linreg_resample(X_c, xtrain, ytrain, ztrain)
            # ravel - makes correspondance between xnew and ynew (otherwise they're not connected)
            # matrix with 2 columns and 100 rows matching two coordinate points (unravelment)
            # concatenation tranposes and merges
            # y fixed,then running through all X values
            Xnew = poly.fit_transform(np.c_[xnew.ravel(), ynew.ravel()])
            zpredict = linreg.predict(Xnew)

            plot(xnew, ynew, zpredict)
            plot(xnew, ynew, ztrue)
            
            print(
                "\nRidge regression with %s-fold Cross-Validation, sigma(noise) = %s:"
                % (str(k), str(float(10 ** (-f))))
            )
            model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, Ridge)
            print(
                "\nLasso regression with %s-fold Cross-Validation, sigma(noise) = %s:"
                % (str(k), str(float(10 ** (-f))))
            )
            model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, Lasso)
        else:
            print(
                "\nRidge regression with %s-fold Cross-Validation, sigma(noise) = %s:"
                % (str(k), str(float(10 ** (-f))))
            )
            model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, Ridge)
            print(
                "\nLasso regression with %s-fold Cross-Validation, sigma(noise) = %s:"
                % (str(k), str(float(10 ** (-f))))
            )
            model_resample(k, lambdas, X_c, xtrain, ytrain, ztrain, Lasso)