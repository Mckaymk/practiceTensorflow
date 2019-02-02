# -*- coding: utf-8 -*-
"""
   File Name：      test1
   Author :         mengkai
   date：           2019/1/27
   description:
"""
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
COST = 9
PROFIT = 1

# 随机生成数据集
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 定义神经网络，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数， 和反方向传播方法
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 生成会话，训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化网络参数
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print("after %d training steps,w1 is : " % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is :\n", sess.run(w1))
