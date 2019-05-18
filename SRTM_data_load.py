# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 00:09:02 2019

@author: andrea
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('SRTM_data_Madrid.tif')

# Show the terrain
plt.figure()
plt.title('Terrain over Madrid')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x = np.arange(0,3601,1)
y = np.arange(0,3601,1)
xx,yy = np.meshgrid(x,y)

# create the x and y coordinate arrays (here we just use pixel indices)
fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, terrain1, cmap=plt.cm.gray,linewidth=2)
plt.show()