# -*- coding: utf-8 -*-
"""
   File Name：      test3
   Author :         mengkai
   date：           2019/1/29
   description:
"""
import tensorflow as tf

# 定义待优化参数w，赋初值为5
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数loss
loss = tf.square(w+1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(80):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: w is %f, loss is %f." % (i, w_val, loss_val))
