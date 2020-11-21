from sklearn.datasets import load_boston
import numpy as np
import tensorflow as tf
boston = load_boston()
print(boston['DIS'])
# x = tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
# y = tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
# meanX = tf.reduce_mean(x)
# meanY = tf.reduce_mean(y)
# sumXY = tf.reduce_sum((x - meanX)*(y - meanY))
# sumX = tf.reduce_sum((x - meanX)*(x - meanX))
# w = sumXY/sumX
# b = meanY-w*meanX
# print('w权值为%f'%w)
# print('b的值为%f'%b.numpy())
# x_test = np.array([128.15,45.00])
# y_pred = (w*x_test+b).numpy()
# print('预测结果为')
# print(y_pred)

