import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
 
TOTAL_POINTS = 1000
 
x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)
 
k_true = 0.7
b_true = 2.0
 
y = x * k_true + b_true + noise
 


k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 5000
learning_rate = 0.02

for n in range(EPOCHS):
    with tf.GradientTape() as t:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y - f))
 
    dk, db = t.gradient(loss, [k, b])
 
    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)

y_pr = k * x + b
y_pr1=y_pr.numpy()

print(k, b, sep="\n")
# print( 'k: ', type(k)) 
# print( 'b: ', type(b)) 
# print( 'y_pr: ', type(y_pr)) 
# print( 'y_pr1: ', type(y_pr1), y_pr1.shape ) 

# plt.scatter(x, y, s=2)
# plt.scatter(x, y_pr1, c='r', s=2)
# plt.show()

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE
k = tf.Variable(0.0)
b = tf.Variable(0.0)
EPOCHS=100
for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
        x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
 
        with tf.GradientTape() as t:
            f = k * x_batch + b
            loss = tf.reduce_mean(tf.square(y_batch - f))
 
        dk, db = t.gradient(loss, [k, b])
 
        k.assign_sub(learning_rate * dk)
        b.assign_sub(learning_rate * db)

y_pr = k * x + b
y_pr1=y_pr.numpy()
print('SGD========================================================================')
print(k, b, sep="\n")
# print( 'k: ', type(k)) 
# print( 'b: ', type(b)) 
# print( 'y_pr: ', type(y_pr)) 
# print( 'y_pr1: ', type(y_pr1), y_pr1.shape ) 
