import tensorflow as tf


sess = tf.Session()
#在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    #创建一个常量为1的v
    v= tf.get_variable('v1',[1],initializer = tf.constant_initializer(1.0))
#因为在foo空间已经创建v的变量，所以下面的代码会报错
#with tf.variable_scope("foo"）:
#   v= tf.get_variable('v',[1])
#在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable的函数将直接获取已声明的变量
#且调用with tf.variable_scope("foo"）必须是定义的foo空间，而不能是with tf.variable_scope(""）未命名或者其他空间。
with tf.variable_scope("foo",reuse =tf.AUTO_REUSE):
    v1= tf.get_variable('v1',[1], initializer = tf.constant_initializer(5.0))
    print(v1==v) #输出为True，代表v1与v是相同的变量
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(v1))
    print(sess.run(v))


with tf.variable_scope("foo1",reuse = False):
    v1= tf.get_variable('v1',[1], initializer = tf.constant_initializer(5.0))
    print(v1==v) #输出为True，代表v1与v是相同的变量
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(v1))
    print(sess.run(v))

print(foo.v1.name)

'''
#获取变量的方式主要有以下两种，实践中tf.get_variable产生的变量一定要搭配tf.variable_scope使用，不然运行脚本会报错
#v = tf.get_variable('v222',shape= [1],initializer = tf.constant_initializer(10.0))
#使用直接定义变量不会报错，可以一直调用
#vc = tf.Variable(tf.constant(1.0,shape = [1]),name = 'v')
#print(vc)
#以下使用with语法，将tf.get_variable与tf.variable_scope搭配使用,且reuse=True时，之前必须定义V
with tf.variable_scope('zdx',reuse = True):
    v = tf.get_variable('v222',shape= [1],initializer = tf.constant_initializer(100.0))
    print(v)
    v1 = tf.get_variable('v222',shape= [1],initializer = tf.constant_initializer(2.0))
    print(v1==v)
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(v1))
    print(sess.run(v))

'''