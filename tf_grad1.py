import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf
x = tf.Variable(-1.0)
y = lambda: x ** 2 - x
 
N = 100
opt = tf.optimizers.SGD(learning_rate=0.1)
for n in range(N):
  opt.minimize(y, [x])
 
print(x.numpy())
#=========================================
# x = tf.Variable(-2.0)
 
# with tf.GradientTape() as tape:
#     y = x ** 2
 
# df = tape.gradient(y, x)
# print(df)
#=========================================
# w = tf.Variable(tf.random.normal((3, 2)))
# b = tf.Variable(tf.zeros(2, dtype=tf.float32))
# x = tf.Variable([[-2.0, 1.0, 3.0]])
 
# with tf.GradientTape() as tape:
#     y = x @ w + b
#     loss = tf.reduce_mean(y ** 2)
 
# df = tape.gradient(loss, [w, b])
# print(df[0], df[1], sep="\n")