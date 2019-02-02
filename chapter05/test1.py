# -*- coding: utf-8 -*-
"""
   File Name：      test1
   Author :         mengkai
   date：           2019/1/30
   description:
"""
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)

print(mnist.train.num_examples)

