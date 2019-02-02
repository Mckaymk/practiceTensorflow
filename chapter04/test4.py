# -*- coding: utf-8 -*-
"""
   File Name：      test4
   Author :         mengkai
   date：           2019/1/29
   description:
"""
import tensorflow as tf

LEARNING_RATE_BASE = 0.1  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 运行多少轮后，更新一次学习率，一般设为：总样本数/BATCH_SIZE

# 设置运行了几轮的计算器，初始值为0，设为不训练
global_step = tf.Variable(0, trainable=False)

# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,staircase=True)

# 定义待优化参数w，赋初值为5
w = tf.Variable(tf.constant(10, dtype=tf.float32))

# 定义损失函数loss
loss = tf.square(w + 1)

# 定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(80):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps:global_step is %f, w is %f, learning_rate is %f,loss is %f." % (i, global_step_val, w_val,
                                                                                              learning_rate_val,loss_val))
