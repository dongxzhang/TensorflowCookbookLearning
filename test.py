import tensorflow as tf
sess = tf.Session()
c = tf.constant(3.0)
print(sess.run(c))
d = tf.add(c, c)
print(sess.run(d))
