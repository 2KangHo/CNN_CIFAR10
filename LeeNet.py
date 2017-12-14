import tensorflow as tf
from config import cfg

class LeeNet:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()
        self.summary()

    def build_net(self):
        with tf.variable_scope(self.name) as scope:
            self.keep_prob = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
            self.Y = tf.placeholder(tf.float32, [None, 10], name='y-input')

            with tf.name_scope('conv1_layer') as inner_scope:
                # L1 ImgIn shape=(?, 32, 32, 3)
                #    Conv     -> (?, 32, 32, 3*64)
                #    ReLU     -> (?, 32, 32, 3*64)
                W1 = tf.Variable(tf.random_normal([3, 3, 3, 3*64], stddev=0.01), name='weight1')
                L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            with tf.name_scope('conv2_layer') as inner_scope:
                # L2 ImgIn shape=(?, 32, 32, 3*64)
                #    Conv     -> (?, 32, 32, 3*64)
                #    ReLU     -> (?, 32, 32, 3*64)
                W2 = tf.Variable(tf.random_normal([3, 3, 3*64, 3*64], stddev=0.01), name='weight2')
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                #    Pool     -> (?, 16, 16, 3*64)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            with tf.name_scope('conv3_layer') as inner_scope:
                # L3 ImgIn shape=(?, 16, 16, 3*64)
                #    Conv     -> (?, 16, 16, 3*128)
                #    ReLU     -> (?, 16, 16, 3*128)
                W3 = tf.Variable(tf.random_normal([3, 3, 3*64, 3*128], stddev=0.01), name='weight3')
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            with tf.name_scope('conv4_layer') as inner_scope:
                # L4 ImgIn shape=(?, 16, 16, 3*128)
                #    Conv     -> (?, 16, 16, 3*128)
                #    ReLU     -> (?, 16, 16, 3*128)
                W4 = tf.Variable(tf.random_normal([3, 3, 3*128, 3*128], stddev=0.01), name='weight4')
                L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
                L4 = tf.nn.relu(L4)
                #    Pool     -> (?, 8, 8, 3*128)
                L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            with tf.name_scope('conv5_layer') as inner_scope:
                # L5 ImgIn shape=(?, 8, 8, 3*128)
                #    Conv     -> (?, 8, 8, 3*256)
                #    ReLU     -> (?, 8, 8, 3*256)
                W5 = tf.Variable(tf.random_normal([3, 3, 3*128, 3*256], stddev=0.01), name='weight5')
                L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
                L5 = tf.nn.relu(L5)
                L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

            with tf.name_scope('conv6_layer') as inner_scope:
                # L6 ImgIn shape=(?, 8, 8, 3*256)
                #    Conv     -> (?, 8, 8, 3*256)
                #    ReLU     -> (?, 8, 8, 3*256)
                W6 = tf.Variable(tf.random_normal([3, 3, 3*256, 3*256], stddev=0.01), name='weight6')
                L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
                L6 = tf.nn.relu(L6)
                #    Pool     -> (?, 4, 4, 3*256)
                L6 = tf.nn.max_pool(L6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L6 = tf.nn.dropout(L6, keep_prob=self.keep_prob)

                #    Reshape  -> (?, 4 * 4 * 3*256) # Flatten them for FC
                L6_flat = tf.reshape(L6, [-1, 4 * 4 * 3*256])

            with tf.name_scope('fc_1_layer') as inner_scope:
                # L7 FC 4x4x3x256 inputs -> 2048 outputs
                W7 = tf.get_variable(shape=[4 * 4 * 3*256, 2048],
                                     initializer=tf.contrib.layers.xavier_initializer(), name='weight7')
                b7 = tf.Variable(tf.random_normal([2048]), name='bias7')
                L7 = tf.nn.relu(tf.matmul(L6_flat, W7) + b7)
                L7 = tf.nn.dropout(L7, keep_prob=self.keep_prob)

            with tf.name_scope('fc_2_layer') as inner_scope:
                # L8 FC 2048 inputs -> 1024 outputs
                W8 = tf.get_variable(shape=[2048, 1024],
                                     initializer=tf.contrib.layers.xavier_initializer(), name='weight8')
                b8 = tf.Variable(tf.random_normal([1024]), name='bias8')
                L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
                L8 = tf.nn.dropout(L8, keep_prob=self.keep_prob)

            with tf.name_scope('final_out_layer') as inner_scope:
                # L9 Final FC 1024 inputs -> 10 outputs
                W9 = tf.get_variable(shape=[1024, 10],
                                     initializer=tf.contrib.layers.xavier_initializer(), name='weight9')
                b9 = tf.Variable(tf.random_normal([10]), name='bias9')
                self.logits = tf.matmul(L8, W9) + b9

        with tf.name_scope('cost') as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

        with tf.name_scope('optimizer') as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=cfg.l_rate).minimize(self.cost)

    def summary(self):
        with tf.name_scope('accuracy') as scope:
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cost_summ = tf.summary.scalar("cost", self.cost)
        self.accu_summ = tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summ = tf.summary.merge_all()

    def predict(self, X_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: X_test, self.keep_prob: keep_prop})

    def evaluate(self, X_sample, Y_sample, batch_size=512):
        N = X_sample.shape[0]
        correct_sample = 0

        for i in range(0, N, batch_size):
            X_batch = X_sample[i: i + batch_size]
            Y_batch = Y_sample[i: i + batch_size]
            N_batch = X_batch.shape[0]

            feed = {self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.0}
            correct_sample += self.sess.run(self.accuracy, feed_dict=feed) * N_batch

        return correct_sample / N

    def train(self, X_data, Y_data, keep_prop=0.6):
        return self.sess.run([self.optimizer, self.cost, self.accuracy, self.merged_summ],
                             feed_dict={self.X: X_data, self.Y: Y_data, self.keep_prob: keep_prop})