#padding的值有两种：SAME和VALID
#SAME: 零填充（如果必要时）
#VALID: 不填充
import tensorflow as tf

a = tf.constant([[
            [[1., 17.],
             [2., 18.], 
             [3., 19.],
             [4., 20.]],
            [[5., 21.],
             [6., 22.],
             [7., 23.],
             [8., 24.]],
            [[9., 25.],
             [10., 26.],
             [11., 27.],
             [12., 28.]],
            [[13., 29.],
             [14., 30.],
             [15., 31.],
             [16., 32.]]
        ]])
pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
with tf.Session() as sess:
    print('image: ')
    print(sess.run(a))
    print('\n')
    print('result: ')
    print(sess.run(pooling))