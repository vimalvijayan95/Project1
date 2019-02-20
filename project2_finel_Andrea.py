# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:39:21 2019

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#Program to solve by the finite element method de diffusion (heat equation).

def analytic(x,t):
    term1 = np.exp(-(np.pi**2)*t)
    term2 = np.sin(np.pi*x)
    return term1*term2

#---------------------|
# 1st: \Delta x=1/10; |
#-------------------- |   

#Fisrt we define our intervals:
L=1; #Longitud of our rod (1D problem)
#Initial and final positions:
xa=0;
xb=L;
Nx=100;
x=np.zeros(Nx+1)
x[0]=xa
#Temporal interval:
T=1;
ta=0;
tb=T;
#Nt=20000
#Ntv=np.arange(20000,2000100,100)
Nt=60000


t=np.zeros(Nt+1)
t[0]=ta
#The difference:
dx=(xb-xa)/Nx;
dt=(tb-ta)/Nt;

#Create the different elements we will use:
u=np.zeros((Nx+1,Nt+1))

for i in range(Nx):
    x[i+1]=xa+dx*(i+1)
for j in range(Nt):
    t[j+1]=ta+dt*(j+1)

#Amplitude:
    
u[:,0]=np.sin(np.pi * x)
#Boundary conditions: 
u[0,:]=0;
u[Nx,:]=0;

#Use finite elements method to solve the problem:
d=dt/(dx**2)
#d=0.1
#To make our equation stable we need that s <= 1/2

for j in range(Nt):
  for i in range(1,Nx):
      u[i,j+1]=u[i,j]+d*(u[i+1,j]+u[i-1,j]-2*u[i,j])
        #u[j+1,i]=u[j,i]+d*(u[j,i+1]+u[j,i-1]-2*[j,i])
        #u[j+1,i] = u[j,i] + s*(T[j,i-1] - 2*T[j,i] + T[j,i+1]) 
        #print(u[i,j+1])

#Transpose our matrix to match values with analytical solution:
ut=np.zeros((Nt+1,Nx+1))
for j in range(Nt+1):
  for i in range(Nx+1):
#Compare with the analytical solution:
     ut[j,i]=u[i,j]

#np.save('finel_1_10_500points',ut)

#Evaluate the difference between the numerical and analytical solution: 
X,T=np.meshgrid(x,t)
u_ann=analytic(X,T)
diff = np.abs(u_ann - ut)
print('Max. difference between Euler method and analytical solution:',np.max(diff))
  

plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 17


# Plot the slices
##SLICES.
#----------------------------

#a) For a given x:
#First, we study the curvature of our function for a given postion. This will help to 
#understand better its time evolotuion and to peak the correct time values.

indx1 = int(Nx/2)
x1 = x[indx1]
#t2 = t_np[indx2]
#t3 = t_np[indx3]

slice_fem_x1 = ut[:,indx1]
slice_ann_x1 = u_ann[:,indx1]
#res2 = G_dnn[indx2,:]
#res3 = G_dnn[indx3,:]


plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 17

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at position = %g"%x1,fontsize=20)
plt.plot(t, slice_fem_x1,'r', linewidth=3.0)
plt.plot(t,slice_ann_x1,'b--')
plt.legend(['fem','analytical'],fontsize=20)

plt.show()

#b)At different times: 
tval=[0, 0.1, 0.2, 0.5, 0.8, 1]
slice_fem_t = np.zeros((len(tval),len(x)))
slice_an_t = np.zeros((len(tval),len(x)))
i=0
for tv in tval:
    indt=np.where(t == tv)
    slice_fem_t[i,:] = ut[indt,:]
    slice_an_t[i,:] = u_ann[indt,:]
    i+=1
    
# Plot the slices
plt.figure(figsize=(10,10))
i=0
pos=[1, 0.39, 0.15, 0.075, 0.045, 0.02]
for tv in tval:
    #plt.title("Computed solutions at time = %g"%tv,fontsize=20)
    plt.plot(x, slice_fem_t[i,:],'r', linewidth=3.0)
    plt.plot(x, slice_an_t[i,:],'b--')
    plt.text(x[int(Nx/2)],pos[i],'t = %.1f'%tv, fontsize=17)
    i+=1

#plt.text(0.05,0.95,'$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.7))
plt.legend(['fem','analytical'],fontsize=25, loc=1)
plt.xlabel('Position x')
plt.ylabel('U(x, $t_0$)')
#plt.title("Solutions at $\Delta$ x = 1/10",fontsize=22)
plt.show()


##3D PLOTS
##--------

fig = plt.figure(figsize=(11,7))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,ut, cmap='hot')
ax.set_title('Finite element method',fontsize=22)
ax.set_xlabel('Position x',fontsize=15,labelpad=10)
ax.set_ylabel('Time t',fontsize=15,labelpad=10)
ax.set_zlabel('Temperature',fontsize=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.7))
#ax.set_zticklabels(ax.get_zticks(),fontsize=15)
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
#plt.show()
fig.savefig('1_10_FEM.png',Transparent=True)


fig = plt.figure(figsize=(7,5))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,u_ann, cmap='hot')
ax.set_title('Analytical solution',fontsize=17)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('t',fontsize=15)
ax.set_zlabel('Temperature',fontsize=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
fig.savefig('1_10_AN.png',Transparent=True)
#plt.show()

fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,diff, cmap='rainbow')
ax.set_title('Difference Euler method and analytical solution',fontsize=22)
ax.set_xlabel('Position x',fontsize=15,labelpad=15)
ax.set_ylabel('Time t',fontsize=15,labelpad=15)
ax.set_zlabel('Difference',labelpad=28,fontsize=15)
ax.tick_params(axis='z', which='major', pad=13)
xes.ticklabel_format(style='sci')
ax.text(0.5,0.95, 0.005, '$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.5))
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
fig.savefig('1_10_diff.png',Transparent=True)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#Program to solve by the finite element method de diffusion (heat equation).

def analytic(x,t):
    term1 = np.exp(-(np.pi**2)*t)
    term2 = np.sin(np.pi*x)
    return term1*term2
#---------------------|
# 2nd: \Delta x=1/100; |
#-------------------- | 
L=1; #Longitud of our rod (1D problem)
#Initial and final positions:
xa=0;
xb=L;
Nx=100;
x=np.zeros(Nx+1)
x[0]=xa
#Temporal interval:
T=1;
ta=0;
tb=T;
Nt=20000;
t=np.zeros(Nt+1)
t[0]=ta
#The difference:
dx=(xb-xa)/Nx;
dt=(tb-ta)/Nt;

#Create the different elements we will use:
u=np.zeros((Nx+1,Nt+1))

for i in range(Nx):
    x[i+1]=xa+dx*(i+1)
for j in range(Nt):
    t[j+1]=ta+dt*(j+1)

#Amplitude:
    
u[:,0]=np.sin(np.pi * x)
#Boundary conditions: 
u[0,:]=0;
u[Nx,:]=0;

#Use finite elements method to solve the problem:
d=dt/(dx**2)
#d=0.1
#To make our equation stable we need that s <= 1/2

for j in range(Nt):
    for i in range(1,Nx):
        u[i,j+1]=u[i,j]+d*(u[i+1,j]+u[i-1,j]-2*u[i,j])
        #u[j+1,i]=u[j,i]+d*(u[j,i+1]+u[j,i-1]-2*[j,i])
        #u[j+1,i] = u[j,i] + s*(T[j,i-1] - 2*T[j,i] + T[j,i+1]) 
        #print(u[i,j+1])

#Transpose our matrix to match values with analytical solution:
ut=np.zeros((Nt+1,Nx+1))
for j in range(Nt+1):
    for i in range(Nx+1):
#Compare with the analytical solution:
       ut[j,i]=u[i,j]

#np.save('finel_1_100_50000',ut)




X,T=np.meshgrid(x,t)
u_ann=analytic(X,T)
diff = np.abs(u_ann - ut)
print('Max absolute difference between analytical solution and FEM = ',np.max(diff))




plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 13

##SLICES.
#----------------------------

#a) For a given x:
#First, we study the curvature of our function for a given postion. This will help to 
#understand better its time evolotuion and to peak the correct time values.

indx1 = int(Nx/2)
x1 = x[indx1]
#t2 = t_np[indx2]
#t3 = t_np[indx3]

slice_fem_x1 = ut[:,indx1]
slice_ann_x1 = u_ann[:,indx1]
#res2 = G_dnn[indx2,:]
#res3 = G_dnn[indx3,:]


plt.rcParams['font.family'] = "Times new roman"
plt.rcParams['font.size'] = 24

# Plot the slices
plt.figure(figsize=(10,10))
plt.title("Computed solutions at position = %g"%x1,fontsize=20)
plt.plot(t, slice_fem_x1,'r', linewidth=3.0)
plt.plot(t,slice_ann_x1,'b--')
plt.legend(['fem','analytical'],fontsize=20)

plt.show()

#b)At different times: 
tval=[0, 0.1, 0.2, 0.5, 0.8, 1]
slice_fem_t = np.zeros((len(tval),len(x)))
slice_ann_t = np.zeros((len(tval),len(x)))
i=0
for tv in tval:
    indt=np.where(t == tv)
    slice_fem_t[i,:] = ut[indt,:]
    slice_ann_t[i,:] = u_ann[indt,:]
    i+=1
    
# Plot the slices
plt.figure(figsize=(10,10))
i=0
pos=[0.02,0.045,0.075,0.15,0.39,1.]
for tv in tval:
    #plt.title("Computed solutions at time = %g"%tv,fontsize=20)
    plt.plot(x, slice_fem_t[i,:],'r', linewidth=3.0)
    plt.plot(x, slice_ann_t[i,:],'b--')
    plt.text(x[int(Nx/2)],pos[i],'t = %.1f'%tv, fontsize=17)
    i+=1

#plt.text(0.05,0.95,'$\Delta$ x=1/10', bbox=dict(facecolor='white', alpha=0.7))
plt.legend(['fem','analytical'],fontsize=22)
plt.xlabel('Position x')
plt.ylabel('U(x, $t_0$)')
#plt.title("Solutions at $\Delta$ x = 1/100",fontsize=22)
plt.show()



fig = plt.figure(figsize=(11,7))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,ut, cmap='hot')
ax.set_title('Finite element method',fontsize=17)
ax.set_xlabel('Position $x$',fontsize=15,labelpad=10)
ax.set_ylabel('Time $t$',fontsize=15,labelpad=10)
ax.set_zlabel('Temperature',fontsize=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.7))
#ax.set_zticklabels(ax.get_zticks(),fontsize=15)
fig.colorbar(surf, shrink=0.5, aspect=7, pad=0.04)
plt.show()
#fig.savefig('1_100_FEM.png',Transparent=True)


fig = plt.figure(figsize=(7,5))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,u_ann, cmap='hot')
ax.set_title('Analytical solution',fontsize=17)
ax.set_xlabel('Position $x$',fontsize=15)
ax.set_ylabel('Time $t$',fontsize=15)
ax.set_zlabel('Temperature',fontsize=15)
ax.text(0.5,0.95, 0.8, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
#fig.savefig('1_100_AN.png',Transparent=True)
plt.show()

fig = plt.figure(figsize=(11,7))
ax = fig.gca(projection='3d')
surf=ax.plot_surface(X,T,diff, cmap='rainbow')
ax.set_title('Difference between FEM and analytical solution',fontsize=22)
ax.set_xlabel('Position x',fontsize=15,labelpad=10)
ax.set_ylabel('Time t',fontsize=15,labelpad=10)
ax.set_zlabel('Difference',labelpad=24,fontsize=15)
ax.tick_params(axis='z', which='major', pad=13)
ax.text(0.5,0.95, 0.00005, '$\Delta$ x=1/100', bbox=dict(facecolor='white', alpha=0.5))
fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
fig.savefig('1_100_diff.png',Transparent=True)
plt.show()
#%%


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#Program to solve by the finite element method de diffusion (heat equation).

def analytic(x,t):
    term1 = np.exp(-(np.pi**2)*t)
    term2 = np.sin(np.pi*x)
    return term1*term2

L=1; #Longitud of our rod (1D problem)
#Initial and final positions:
xa=0;
xb=L;
Nx=100;
x=np.zeros(Nx+1)
x[0]=xa
#Temporal interval:
T=1;
ta=0;
tb=T;
#Nt=20000
Ntv=np.arange(20000,210000,10000)
k=0;
maxdiff=np.zeros(len(Ntv))

for Nt in Ntv:
  t=np.zeros(Nt+1)
  t[0]=ta
#The difference:
  dx=(xb-xa)/Nx;
  dt=(tb-ta)/Nt;

#Create the different elements we will use:
  u=np.zeros((Nx+1,Nt+1))

  for i in range(Nx):
    x[i+1]=xa+dx*(i+1)
  for j in range(Nt):
    t[j+1]=ta+dt*(j+1)

#Amplitude:
    
  u[:,0]=np.sin(np.pi * x)
#Boundary conditions: 
  u[0,:]=0;
  u[Nx,:]=0;

#Use finite elements method to solve the problem:
  d=dt/(dx**2)
#d=0.1
#To make our equation stable we need that s <= 1/2

  for j in range(Nt):
    for i in range(1,Nx):
        u[i,j+1]=u[i,j]+d*(u[i+1,j]+u[i-1,j]-2*u[i,j])
        #u[j+1,i]=u[j,i]+d*(u[j,i+1]+u[j,i-1]-2*[j,i])
        #u[j+1,i] = u[j,i] + s*(T[j,i-1] - 2*T[j,i] + T[j,i+1]) 
        #print(u[i,j+1])

#Transpose our matrix to match values with analytical solution:
  ut=np.zeros((Nt+1,Nx+1))
  for j in range(Nt+1):
    for i in range(Nx+1):
#Compare with the analytical solution:
       ut[j,i]=u[i,j]

#np.save('finel_1_10_500points',ut)

#Evaluate the difference between the numerical and analytical solution: 
  X,T=np.meshgrid(x,t)
  u_ann=analytic(X,T)
  diff = np.abs(u_ann - ut)
  maxdiff[k]=np.max(diff)
  print(k,np.max(diff))
  k+=1

indxm=np.where(maxdiff == np.min(maxdiff))
print(Ntv[indxm])
