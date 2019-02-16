# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:39:21 2019

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Program to solve by the finite element method de diffusion (heat equation).

#Fisrt we define our intervals:
L=1; #Longitud of our rod (1D problem)
#Initial and final positions:
xa=0;
xb=L;
x=np.zeros(nx+1)
x[0]=xa
#Temporal interval:
T=0.001;
ta=0;
tb=T;
t=np.zeros(nt+1)
t[0]=ta
#The difference:
nx=100;
nt=100;
dx=(xb-xa)/nx;
dt=(tb-ta)/nt;

#Create the different elements we will use:
u=np.zeros((nx+1,nt+1))

for i in range(nx):
    x[i+1]=xa+dx*(i+1)
for j in range(nt):
    t[j+1]=ta+dt*(j+1)

#Amplitude:
u[:,0]=np.sin(np.pi * x)
#Boundary conditions: 
u[0,:]=0;
u[nx,:]=0;

#Use finite elements method to solve the problem:
s=dt/(dx**2)
#To make our equation stable we need that s <= 1/2

for j in range(nt):
    for i in range(1,nx):
        u[i,j+1]=u[i,j]+s*(u[i+1,j]+u[i-1,j]-2*u[i,j])
        
        #T[j+1,i] = T[j,i] + s*(T[j,i-1] - 2*T[j,i] + T[j,i+1]) 
        #print(u[i,j+1])
        
fig = plt.figure()
ax = fig.gca(projection='3d')
X,T = np.meshgrid(x,t)
#surf=ax.plot_surface(x,t,u, cmap='gist_rainbow')
surf=ax.plot_surface(X,T,u, cmap='gist_rainbow')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Temperature')


fig.colorbar(surf, shrink=0.5, aspect=5)
#ax.plot_trisurf(x, t, u, linewidth=0.2, antialiased=True)

plt.show()




