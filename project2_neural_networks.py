# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:07:53 2019

@author: andrea
"""

import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d



tf.reset_default_graph()
tf.set_random_seed(4155)

#-------------------------------------
# STEP 1: Definition of the variables.|
#-------------------------------------

Nx = 100
x_np = np.linspace(0,1,Nx+1)

Nt = 2000
t_np = np.linspace(0,1,Nt+1)

X,T = np.meshgrid(x_np, t_np)

x = X.ravel()
t = T.ravel()

# Conversion of the different variables into tensors

zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))
points = tf.concat([x,t],1)


#--------------------------------------------
# STEP 2: Construction of the neural network.|
#--------------------------------------------

# 1.Definition of the number of neurons within each layer 
# and total number interactions 
num_hidden_neurons = [90]
num_iter = 1000

X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)

# 2.Construction of the deep neural network with a given activation function. 
with tf.variable_scope('dnn'):
    
    num_hidden_layers = np.size(num_hidden_neurons)
    
    # a)Input layer.
    previous_layer = points

    # b)Hidden layers
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], 
                                        activation=tf.nn.sigmoid)
        previous_layer = current_layer
    
    # c)Output layer
    dnn_output = tf.layers.dense(previous_layer, 1)


#--------------------------------------
# STEP 3: Training the neural network.|
#--------------------------------------

def u(x):
    return tf.sin(np.pi*x)

# 1.Definition of the cost (loss) function. 
with tf.name_scope('loss'):
    
    # a)Trial solution wich satisfies the boundary conditions. 
    g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
    g_trial_dt =  tf.gradients(g_trial,t)
    g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
    # b)Calculation of the cost function as the mean square error. 
    loss = tf.losses.mean_squared_error(zeros, g_trial_d2x[0] - g_trial_dt[0])


# 2.Minimization of the cost function using gradient descent technique. 
learning_rate = 0.065

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

g_analytic = tf.sin(np.pi*x)*tf.exp(-(np.pi**2)*t)
g_dnn = None

 
#-----------------------------
# STEP 4: Executing the code.|
#-----------------------------

# 1.Start a session
with tf.Session() as session:
    init.run()
    for i in range(num_iter):
        session.run(traning_op)
        print(i, loss.eval())
        

    g_analytic = g_analytic.eval()
    g_dnn = g_trial.eval()
    cost = loss.eval()
    
# 2. Compare with the analytical solution. 
diff = np.abs(g_analytic - g_dnn)
print('Max absolute difference between analytical solution and TensorFlow 
      DNN = ',np.max(diff))
print('Final cost function', cost)

G_analytic = g_analytic.reshape((Nt+1,Nx+1))
G_dnn = g_dnn.reshape((Nt+1,Nx+1))

#3. Save the results. 
np.save('analytic_tensorflow',G_analytic)
np.save('dnn_tensorflow',G_dnn)


#%%
import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#-----------------------------------------------------
# ACTIVATION FUNCTION AND LEARNING RATE OPTIMIZATION |
#-----------------------------------------------------


tf.reset_default_graph()
tf.set_random_seed(4155)



Nx = 10
x_np = np.linspace(0,1,Nx+1)

Nt = 10
t_np = np.linspace(0,1,Nt+1)

X,T = np.meshgrid(x_np, t_np)

x = X.ravel()
t = T.ravel()

#Conversion of the different variables into tensors

zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

points = tf.concat([x,t],1)


#--------------------------------------------
# STEP 2: Construction of the neural network.|
#--------------------------------------------

# 1.Definition of the number of neurons within each layer 
# and total number interactions 
num_hidden_neurons = [90]
num_iter = 10

X = tf.convert_to_tensor(X)
T = tf.convert_to_tensor(T)

#2.Construction of the deep neural network with a given activation function. 
with tf.variable_scope('dnn'):
    
    num_hidden_layers = np.size(num_hidden_neurons)
    
    #a)Input layer.
    previous_layer = points

    #b)Hidden layers
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],
                                        activation=tf.nn.sigmoid)
        previous_layer = current_layer
    
    #c)Output layer
    dnn_output = tf.layers.dense(previous_layer, 1)


#-------------------------------------
#STEP 3: Training the neural network.|
#-------------------------------------

def u(x):
    return tf.sin(np.pi*x)

# 1.Definition of the cost (loss) function. 
with tf.name_scope('loss'):
    
    # a)Trial solution wich satisfies the boundary conditions. 
    g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
    g_trial_dt =  tf.gradients(g_trial,t)
    g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
    
    # b)Calculation of the cost function as the mean square error. 
    loss = tf.losses.mean_squared_error(zeros, g_trial_d2x[0] - g_trial_dt[0])


# 2.Minimization of the cost function.
    
# Study different values for the learning rate    
learning_rate = np.linspace(1e-4,1e-1,100)
cost = np.zeros(len(learning_rate))
maxdiff = np.zeros(len(learning_rate))
j = 0
for lr in learning_rate:
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(lr)
        traning_op = optimizer.minimize(loss)
 
    init = tf.global_variables_initializer()

    g_analytic = tf.sin(np.pi*x)*tf.exp(-(np.pi**2)*t)
    g_dnn = None

    with tf.Session() as session:
        init.run()
        for i in range(num_iter):
            session.run(traning_op)
     #print(i)

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()
        lossv = loss.eval() 
        
    cost[j] = lossv
    diff = np.abs(g_analytic - g_dnn)
    maxdiff[j] = np.max(diff)
    print(j,cost[j],maxdiff[j])
    j += 1

# 3.Save the results. 
np.save('maxdiff_tanh.npy',maxdiff)
np.save('cost.npy_tanh',cost)


#%%

#--------------------------------------------
# NEURAL NETWORK CONFIGURATION OPTIMIZATION |
#--------------------------------------------

# 1.Study different number of neurons for the single layer neural network 
# model

num_hidden_neurons = np.arange(50,151,1)
maxdiff = np.zeros(len(num_hidden_neurons))
cost = np.zeros(len(num_hidden_neurons))

j = 0
num_iter = 100

for neur_number in num_hidden_neurons:

# Initialize each time.     
    tf.reset_default_graph()
    tf.set_random_seed(4155)
 
    Nx = 10
    x_np = np.linspace(0,1,Nx+1)

    Nt = 10
    t_np = np.linspace(0,1,Nt+1)

    X,T = np.meshgrid(x_np, t_np)

    x = X.ravel()
    t = T.ravel()
#Conversion of the different variables into tensors

    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)),shape=(-1,1))
    x = tf.reshape(tf.convert_to_tensor(x),shape=(-1,1))
    t = tf.reshape(tf.convert_to_tensor(t),shape=(-1,1))

    points = tf.concat([x,t],1)
 
    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

#--------------------------------------------
#STEP 2: Construction of the neural network.|
#--------------------------------------------
    
#1.Construction of the deep neural network with a given activation function.
    with tf.variable_scope('dnn'):
 #with tf.variable_scope('dnn', reuse=True):
    
   
       #a)Input layer.
        previous_layer = points

       #b)Hidden layer
        for l in range(1):
            current_layer = tf.layers.dense(previous_layer, neur_number ,
                                           activation=tf.nn.sigmoid)
            previous_layer = current_layer
    
       #c)Output layer
        dnn_output = tf.layers.dense(previous_layer, 1)


#----------------------------------------
#STEP 3: Training the neural network.|
#----------------------------------------
    def u(x):
        return tf.sin(np.pi*x)
 
#1.Definition of the cost (loss) function. 
    with tf.name_scope('loss'):
    
    #a)Trial solution wich satisfies the boundary conditions. 
        g_trial = (1 - t)*u(x) + x*(1-x)*t*dnn_output
        g_trial_dt =  tf.gradients(g_trial,t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial,x),x)
    #b)Calculation of the cost function as the mean square error. 
        loss = tf.losses.mean_squared_error(zeros, g_trial_d2x[0] - 
                                            g_trial_dt[0])

    learning_rate = 0.065

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        traning_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_analytic = tf.sin(np.pi*x)*tf.exp(-(np.pi**2)*t)
    g_dnn = None

 
## The execution phase

    with tf.Session() as session:
        init.run()
        for i in range(num_iter):
            session.run(traning_op)

        # If one desires to see how the cost function behaves during training
        #if i % 100 == 0:
        #    print(loss.eval())

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()
        lossv = loss.eval()
## Comp are with the analutical solution
    diff = np.abs(g_analytic - g_dnn)
    maxdiff[j] = np.max(diff)
    cost[j] = lossv
    print(j,maxdiff[j],cost[j],neur_number)
    j+=1
 
index1 = np.where (maxdiff == np.min(maxdiff))
index2 = np.where (cost == np.min(cost))
print(index1, index2)
print(num_hidden_neurons[index1],num_hidden_neurons[index2])