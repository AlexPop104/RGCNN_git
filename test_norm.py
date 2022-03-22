import numpy as np
import torch as t
from torch.linalg import norm
import tensorflow as tf
import time

A = t.rand([32, 22, 22])

n = norm(A) ** 2

with tf.Session() as sess:
    print(tf.nn.l2_loss(A).eval())

print(n/2)
