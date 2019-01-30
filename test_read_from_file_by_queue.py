#form https://blog.csdn.net/duanyajun987/article/details/82352596
# 同时打开多个文件(文件格式必须一样)，隐式示创建Queue，同时隐含了QueueRunner的创建
import tensorflow as tf
filename_queue = tf.train.string_input_producer(["iris_1.csv", "iris_2.csv", "iris_3.csv", "iris_4.csv"],
 shuffle = False, num_epochs = 2)
 
# num_epochs = 2表示文件最多被读两轮，超出两轮队列为空
#shuffle = False 顺序读取文件
#shuffle = True 把文件顺序打乱再读
reader = tf.TextLineReader()
# Tensorflow的Reader对象可以直接接受一个Queue作为输入
#  每次 read 的执行都会从文件中读取一行内容
key, value = reader.read(filename_queue)

# 如果某一列为空，指定默认值，同时指定了默认列的类型
record_defaults = [[0.0], [0.0], [0.0], [0.0], [0]]
# decode_csv 操作会解析读取的一行内容并将其转为张量列表
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = [col1, col2, col3, col4]

#获取一行数据
#row = tf.decode_csv(value, record_defaults=record_defaults)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    # 启动计算图中所有的队列线程 调用
    #  tf.train.start_queue_runners 来将文件名填充到队列，否则
    # read
    # 操作会被阻塞，直到文件名队列中有值为止。
    threads = tf.train.start_queue_runners(coord=coord)
    
    # 主线程，消费10个数据，循环次数受num_epochs影响
    for _ in range(10):
        try:
            example, label = sess.run([features, col5])
            print('Step {0} {1} {2}'.format(_,example,label))
        except tf.errors.OutOfRangeError:
            print("training stop, input queue is empty")
            break

    # 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    # 等待指定的线程结束
    coord.join(threads)