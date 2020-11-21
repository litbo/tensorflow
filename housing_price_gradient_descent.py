import tensorflow as tf
import time
import numpy as np
x = tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y = tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
learn_rate = 0.00001
iter  =  1000
step = 10
np.random.seed(612)
w = tf.Variable(np.random.randn(),dtype=tf.float64)
b = tf.Variable(np.random.randn(),dtype=tf.float64)
mse = []
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        y_ = w*x + b
        loss = 0.5*tf.reduce_mean(tf.square(y - y_))
    mse.append(loss)
    dl_dw,dl_db = tape.gradient(loss,[w,b])
    if i % step == 0:
        print('i:%i,Loss:%f,w:%f,b:%f'%(i,mse[i],w,b))
print(min(mse))

# dl_dw = np.mean(x * (w*x+b-y) )
# dl_db = np.mean(w*x)
# w = w-learn_rate*dl_dw
# b = b-learn_rate*dl_db
# pred = w*x+b
# Loss = np.mean(np.square(y - pred))/2