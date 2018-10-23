import tensorflow as tf

class Lenet(object):
    def __init__(self):
        self.x=tf.placeholder('float',shape=[None,784])
        self.y=tf.placeholder('float',shape=[None,10])
        self.keep_prob=tf.placeholder(tf.float32)
        self.output=None
        self.init()

    def init(self):
        with tf.Session() as sess:
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])  # 第一层：卷积层 # 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
            conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))  # 移动步长为1, 使用全0填充
            conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 激活函数Relu去线性化
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # 第二层：最大池化层 #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充

            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 第三层：卷积层

            conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(
                stddev=0.1))  # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64
            conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))  # 第四层：最大池化层 #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充

            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 第五层：全连接层
            fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=0.1))  # 7*7*64=3136把前一层的输出变成特征向量
            fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))

            pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])

            fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)  # 为了减少过拟合，加入Dropout层
            fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)

            # 第六层：全连接层
            fc2_weights = tf.get_variable("fc2_weights", [1024, 10],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))  # 神经元节点数1024, 分类节点10
            fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))
            fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

            # 第七层：输出层 #
            y_conv = tf.nn.softmax(fc2)
            self.output=y_conv


