import tensorflow as tf
sess = tf.Session()
c = tf.constant(3.0)
print(sess.run(c))
d = tf.add(c, c)
print(sess.run(d))
e = tf.multiply(c, d)
print(sess.run(e))
f = tf.multiply(e, d)
print(sess.run(f))
print(3)
import numpy as np
z = np.array([1,2])
