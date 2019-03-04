# -*- coding: utf-8 -*-
"""
   File Name：      test04
   Author :         mengkai
   date：           2019/2/2
   description:
"""
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 8
seed = 23455

# 随机生成32行2列的随机数据
rng = np.random.RandomState(seed=seed)
X = rng.rand(32, 2)

# 设置Y值
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
# 打印X,Y值
print("X:\n", X)
print("Y:\n", Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
# 设置权重的初始值
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 设置神经网络过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 生成会话，计算神经网络

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training steps,loss on all data is %g" % (i, total_loss))
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
