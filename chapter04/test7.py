# -*- coding: utf-8 -*-
"""
   File Name：      test7
   Author :         mengkai
   date：           2019/1/30
   description:
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2


def generateds():
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)
    # 随机返回（300，2）的矩阵，表示300组坐标点作为输入值
    X = rdm.randn(300, 2)
    # 对着300行值进行判断，如果小于2给y赋值1，其余赋值为0
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 给1设置为红色，给0设置为蓝色
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集进行shape整理，第一个元素为-1，随第二个参数计算得到，第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    return X, Y_, Y_c

