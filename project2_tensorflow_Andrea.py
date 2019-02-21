# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:34:18 2019

@author: andrea
"""

import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


tf.reset_default_graph()



Nx = 100; Nt = 100
L=1; T=1;
xv = np.linspace(0, L, Nx)
tv = np.linspace(0, T,  Nt)





X,T = np.meshgrid(xv, tv)

x = X.ravel()
t = T.ravel()
 
#reshape chagnes the shape of the given tensor (matrix) in the dimension specified in the second argument. 
#This creates vectors with just one column and the number of rows necessary to put all the elemens. 
zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

#What does tf.concat do?
# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
#tf.shape(tf.concat([t3, t4], 0))  # [4, 3] i.e 2+2 (along the first axis)
#tf.shape(tf.concat([t3, t4], 1))  # [2, 6] i.e 3+3 (along the 2n axis)

points = tf.concat([x,t],1) #this means points will have shape (_,2) that is what we want. 

num_iter = 10
num_hidden_neurons = [90]

X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)

#BUILDING NEURAL NETWORK.
#-----------------------

# tf.name_scope is used to group each step in the construction,
# just for a more organized visualization in TensorBoard
with tf.variable_scope('dnn'):
    
    #Input layer
    previous_layer = points

    #Hidden layers
    num_hidden_layers = np.size(num_hidden_neurons)
#tf.layers.dense(inputs, units, activation=None,...). units is the dimensionality of the output. 
# outputs = activation(inputs * kernel + bias) where activation is the activation function passed as the activation argument (if not None), kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True).    
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
        previous_layer = current_layer

    #Output layer
    dnn_output = tf.layers.dense(previous_layer, 1)

#CONSTRUCTING THE PDE:
#--------------------
    
def u(x):
    return tf.sin(np.pi*x)

#a. Cost function:
with tf.name_scope('cost'):
    
    g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
    #Compute the derivatives:
    g_trial_dt =  tf.gradients(g_trial,t)
    g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
# zeros: The ground truth output tensor, same dimensions as 'predictions'. predictions: The predicted outputs. 
#We write zeros because in the end we want that the difference between both is zero, since the diff eq. is g''t-g'x=0. 
#error defined as the squared of the differences.

#right_side = (3*x_tf + x_tf**2)*tf.exp(x_tf)
#err = tf.square( -d2_g_trial[0] - right_side)
#cost = tf.reduce_sum(err, name = 'cost')
#tf.losses.mean_squared_errpr adds sum of squares error to the training procedure. 
    
    error = tf.losses.mean_squared_error(zeros, g_trial_d2x[0] - g_trial_dt[0])
    
#b. Minimize the cost function with a learning rate.     
learning_rate = 0.01
with tf.name_scope('train'):
#The function in Tensor Flow is: train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traning_op = optimizer.minimize(error)

## Define a node that initializes all of the other nodes in the computational graph
# used by TensorFlow:
init = tf.global_variables_initializer()

#g(x,t)=exp(−π2t)sin(πx)
g_analytic = tf.sin(np.pi*x)*tf.exp(-(np.pi**2)*t)
g_dnn = None

## EXECUTION PHASE.
#-----------------
# Start a session where the graph defined from the construction phase can be evaluated at:
#A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. 
with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        #Evaluate the tensor training_op
        sess.run(traning_op)

        # If one desires to see how the cost function behaves during training
        #if i % 100 == 0:
        #    print(loss.eval())
#After the graph has been launched in a session, the value of the Tensor can be computed by passing it to tf.Session.run. t.eval() is a shortcut for calling tf.get_default_session().run(t).
#g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
    g_analytic = g_analytic.eval()
    g_dnn = g_trial.eval()


## COMPARE WITH THE ANALYTICAL SOLUTION. 
diff = np.abs(g_analytic - g_dnn)
print('Max absolute difference between analytical solution and TensorFlow DNN = ',np.max(diff))
G_analytic = g_analytic.reshape((Nt,Nx))
G_dnn = g_dnn.reshape((Nt,Nx))
diff = np.abs(G_analytic - G_dnn)


  
## PLOT THE RESULTS. 
X,T = np.meshgrid(xv, tv)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Solution from the deep neural network w/ %d layer'%len(num_hidden_neurons))
s = ax.plot_surface(X,T,G_dnn,linewidth=0,antialiased=False,cmap=cm.hot)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Analytical solution')
s = ax.plot_surface(X,T,G_analytic,linewidth=0,antialiased=False,cmap=cm.hot)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.set_title('Difference')
s = ax.plot_surface(X,T,diff,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('Time $t$')
ax.set_ylabel('Position $x$');

#ut1=np.load('finel_1_10.txt')

    