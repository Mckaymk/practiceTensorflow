# -*- coding: utf-8 -*-
"""
   File Name：      test5
   Author :         mengkai
   date：           2019/1/29
   description:
"""
import tensorflow as tf

# 定义一个32位浮点数
w1 = tf.Variable(0, dtype=tf.float32)

# 定义一个迭代轮数值，这个值不可被优化，不参与训练
global_step = tf.Variable(0, trainable=False)

# 实例化滑动平均类，设置山间率位0.99，当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# apply后面的是更新列表，每次运行sess.run(ema_op)即，对列表中的值求一次滑动平均值
# tf.trainable_variables自动将所有参与训练的值汇总成列表
# ema_op = ema.apply([])
ema_op = ema.apply(tf.trainable_variables())

# 查看不同的取值变化
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 打印当前w1和w1的滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    # 参数w1的值赋值为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新step和w1的值，模拟出100轮后，参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)

    # 每次更新sess.run，就打印w1的滑动平均值
    for i in range(100):
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))
