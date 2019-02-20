# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:18:26 2019

@author: andrea
"""


import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#-----------------
# Part A: SLICES.|
#-----------------

# 1) For a given x:
# ----------------

# a) Create time and position with the number of points corresponding
# to each of the employed methods.
Nx = 100
x_np = np.linspace(0,1,Nx+1)
Nt = 2000
t_np = np.linspace(0,1,Nt+1)


Nxe = 100
x_npe = np.linspace(0,1,Nxe+1)
Nte = 60000
t_npe = np.linspace(0,1,Nte+1)

# b)Selecting the slice. 
indx1 = int(Nx/2)
indx1e = int(Nxe/2)
x1 = x_np[indx1]
x1e = x_npe[indx1]

# c)Loading the data. 
G_Euler = np.load('Euler_60000.npy')
G_dnn = np.load('Dnn_2000.npy')
G_analytic = np.load('An_dnn_2000.npy')
diff_Euler= np.load('diff_Euler_60000.npy')
diff_dnn= np.abs(G_analytic-G_dnn)


# d)Slicing the functions. 
slice_Euler_x1 = G_Euler[:,indx1e]
slice_dnn_x1 = G_dnn[:,indx1]
slice_an_x1 = G_analytic[:,indx1]



plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 24

# e)Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at x = %g"%x1,fontsize=25)
plt.plot(t_np, slice_dnn_x1,'g', linewidth=2.0 )
plt.plot(t_npe, slice_Euler_x1,'r', linewidth=2.0)
plt.plot(t_np,slice_an_x1,'b--', linewidth=2.0)
plt.xlabel('Time t',fontsize=25)
plt.ylabel('U(x, $t_0$)',fontsize=25)
plt.xlim(0,1)
plt.legend(['dnn','Euler','analytical'],fontsize=26)

plt.show()

# 1) For different times:
# -----------------------

tval=[0, 0.1, 0.2, 0.5, 0.8, 1]
slice_dnn_t = np.zeros((len(tval),len(x_np)))
slice_an_t = np.zeros((len(tval),len(x_np)))
slice_Euler_t = np.zeros((len(tval),len(x_np)))

# a)Creating the slices. 
i=0
for tv in tval:
    indt=np.where(t_np == tv)
    indte=np.where(t_npe == tv)
    slice_dnn_t[i,:] = G_dnn[indt,:]
    slice_an_t[i,:] = G_analytic[indt,:]
    slice_Euler_t[i,:] = G_Euler[indte,:]
    i+=1

# b)Plot the slices    
plt.figure(figsize=(10,10))
i=0
#pos=[0.02,0.045,0.075,0.15,0.39,1.]
pos=[1, 0.39, 0.15, 0.075, 0.045, 0.02]
#pos=[1, 0.625, 0.35, 0.13, 0.093, 0.065]
for tv in tval:
    #plt.title("Computed solutions at time = %g"%tv,fontsize=20)
    plt.plot(x_np, slice_dnn_t[i,:],'r', linewidth=3.0)
    plt.plot(x_np, slice_an_t[i,:],'b--')
    plt.text(x_np[int(Nx/2)],pos[i],'t = %.1f'%tv, fontsize=17)
    i+=1

#plt.text(0.05,0.95,'$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.7))
plt.legend(['dnn','analytical'],fontsize=26,loc=1)
plt.xlabel('Position x',fontsize=25)
plt.ylabel('U(x, $t_0$)',fontsize=25)
#plt.title("Solutions at $\Delta$ x = 1/10 and num_iter = 10000",fontsize=22)
plt.show()

plt.figure(figsize=(10,10))
i=0
pos=[1, 0.39, 0.15, 0.075, 0.045, 0.02]
#pos=[1, 0.625, 0.35, 0.13, 0.093, 0.065]
for tv in tval:
    #plt.title("Computed solutions at time = %g"%tv,fontsize=20)
    plt.plot(x_npe, slice_Euler_t[i,:],'r', linewidth=3.0)
    plt.plot(x_np, slice_an_t[i,:],'b--')
    plt.text(x_np[int(Nx/2)],pos[i],'t = %.1f'%tv, fontsize=17)
    i+=1

#plt.text(0.05,0.95,'$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.7))
plt.legend(['Euler','analytical'],fontsize=26,loc=1)
plt.xlabel('Position x',fontsize=25)
plt.ylabel('U(x, $t_0$)',fontsize=25)
#plt.title("Solutions at $\Delta$ x = 1/10 and num_iter = 10000",fontsize=22)
plt.show()

#-----------------
# Part A: 3D PLOTS.|
#-----------------


X,T = np.meshgrid(x_np, t_np)
Xe,Te = np.meshgrid(x_npe, t_npe)


fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
ax.set_title('Euler method')
s = ax.plot_surface(Xe,Te,G_Euler,linewidth=0,antialiased=False,cmap=cm.hot)
ax.set_xlabel('Position $x$',fontsize=25,labelpad=17)
ax.set_ylabel('Time $t$',fontsize=25,labelpad=17)
ax.set_zlabel('Temperature',fontsize=25, labelpad=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(s, shrink=0.5, aspect=7, pad=0.04, cmap=cm.hot)


fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
ax.set_title('Deep neural network')
s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.hot)
ax.set_xlabel('Position $x$',fontsize=25,labelpad=17)
ax.set_ylabel('Time $t$',fontsize=25,labelpad=17)
ax.set_zlabel('Temperature',fontsize=25, labelpad=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(s, shrink=0.5, aspect=7, pad=0.04, cmap=cm.hot)


fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
ax.set_title('Analytic solution')
s = ax.plot_surface(X,T,G_analytic,linewidth=0,antialiased=False,cmap=cm.hot)
ax.set_xlabel('Position $x$',fontsize=25,labelpad=17)
ax.set_ylabel('Time $t$',fontsize=25,labelpad=17)
ax.set_zlabel('Temperature',fontsize=25, labelpad=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(s, shrink=0.5, aspect=7, pad=0.04, cmap=cm.hot)


fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference between Euler method and analytical solution')
s = ax.plot_surface(Xe,Te,diff_Euler.reshape(Nte+1,Nxe+1),linewidth=0,antialiased=False,cmap=cm.rainbow)
ax.set_xlabel('Position $x$',fontsize=25,labelpad=17)
ax.set_ylabel('Time $t$',fontsize=25,labelpad=17)
ax.set_zlabel('Difference',labelpad=15,fontsize=25)
ax.ticklabel_format(axis='z',style='sci',scilimits=(0,0), useOffset=True)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.5))
#fig.colorbar(s, shrink=0.5, aspect=7, pad=0.04)
plt.show()


fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference between DNN and analytical solution')
s = ax.plot_surface(X,T,diff_dnn.reshape(Nt+1,Nx+1),linewidth=0,antialiased=False,cmap=cm.rainbow)
ax.set_xlabel('Position $x$',fontsize=25,labelpad=17)
ax.set_ylabel('Time $t$',fontsize=25,labelpad=17)
ax.set_zlabel('Difference',labelpad=15,fontsize=25)
ax.ticklabel_format(axis='z',style='sci',scilimits=(0,0), useOffset=True)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.5))
#fig.colorbar(s, shrink=0.5, aspect=7, pad=0.04)
plt.show()

#%%
import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


maxdiff_sigmoid = np.load('maxdiff_sigmoid.npy')
maxdiff_tanh = np.load('maxdiff_tanh.npy')
maxdiff_elu = np.load('maxdiff_elu.npy')

learning_rate = np.linspace(1e-4,1e-1,100)


plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 24
plt.figure(figsize=(10,10))

plt.plot(learning_rate, maxdiff_sigmoid ,'r', linewidth=2.0 )
#plt.plot(learning_rate, maxdiff_tanh ,'b', linewidth=2.0 )
#plt.plot(learning_rate, maxdiff_elu ,'g', linewidth=2.0 )
plt.xlabel('Learning rate',fontsize=25)
plt.ylabel('Max. difference',fontsize=25)
plt.xlim(0,0.1)
plt.ylim(0,1)

#plt.legend(['sigmoid','tanh','exponential linear'],fontsize=26)
plt.legend(['sigmoid'],fontsize=26)


plt.show()



