import numpy as np
import tensorflow as tf
from make_sin_data import make_sin_data2
from tensorflow.contrib import rnn

#global variants
INPUT_NODE_NUM = 1
HIDDEN_UNITS = 32
OUT_NODE_NUM = 1
SEQUENCE_LEN = 50
EPOCH = 200
BATCH_SIZE = 50


#make input data, target data
X_train, X_test, sin_x_train, sin_x_test = make_sin_data2()
print("X_train.shape", X_train.shape)
print("X_test.shape", X_train.shape)
print("sin_x_train.shape", sin_x_train.shape)
print("sin_x_test.shape", sin_x_train.shape)


#computation graph
x_ = tf.placeholder(tf.float32, [None, SEQUENCE_LEN, INPUT_NODE_NUM])
d_ = tf.placeholder(tf.float32, [None, 1])
# x_image = tf.reshape(x_, [-1, 28, 28, 1])#reshape for convolution

def RNN(x):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)...[timesteps, (batch_size, n_input)]
    x_unpack = tf.unstack(x, SEQUENCE_LEN, axis=1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_UNITS, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x_unpack, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return outputs[-1]

with tf.name_scope("fc1"):
    x_reshape = tf.reshape(x_, [tf.shape(x_)[0] * tf.shape(x_)[1], tf.shape(x_)[2]])
    w1 = tf.Variable(tf.random_normal([INPUT_NODE_NUM, HIDDEN_UNITS], mean=0.0, stddev=0.05), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([HIDDEN_UNITS]), dtype=tf.float32)
    fc1 = tf.matmul(x_reshape, w1) + b1

with tf.name_scope("LSTM"):
    fc1_reshape = tf.reshape(fc1, [tf.shape(x_)[0], tf.shape(x_)[1], HIDDEN_UNITS])
    # fc1_reshape = tf.reshape(fc1, [25, 50, 32])
    rnn_out = RNN(fc1_reshape)

with tf.name_scope("fc2"):
    w2 = tf.Variable(tf.random_normal([HIDDEN_UNITS, OUT_NODE_NUM], mean=0.0, stddev=0.05), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([OUT_NODE_NUM]), dtype=tf.float32)
    prob = tf.matmul(rnn_out, w2) + b2

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(prob - d_), name='loss')

def calculate_R2(y, pred):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
    R2 = tf.subtract(1.0, tf.div(unexplained_error, total_error))
    return R2

with tf.name_scope("R2"):
    r2 = calculate_R2(d_, prob)

# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('R2', r2)

merged = tf.summary.merge_all()


sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('LSTM_demo_tf_04', sess.graph)

#make minibatch
def make_minibatch(per_list, x_data, sin_x):
    X_data_batch_list = []
    d_data_batch_list = []
    for data_1 in per_list:
        X_data_batch_list.append(x_data[data_1:data_1 + SEQUENCE_LEN])
        d_data_batch_list.append(sin_x[data_1 + SEQUENCE_LEN])
    X_data_batch = np.array(X_data_batch_list)
    d_data_batch = np.array(d_data_batch_list)
    X_data_batch_reshape = X_data_batch.reshape((X_data_batch.shape[0], X_data_batch.shape[1], 1))
    d_data_batch_reshape = d_data_batch.reshape((d_data_batch.shape[0], 1))

    return X_data_batch_reshape, d_data_batch_reshape



for epoch in range(EPOCH):
    loss_train_sum = np.float32(0)
    data_num_sum = np.float32(0)
    x_random_list = np.random.permutation(len(X_train) - SEQUENCE_LEN)
    for batch_count in range(0, len(X_train) - SEQUENCE_LEN, BATCH_SIZE):

        batch_1_list = x_random_list[batch_count : (batch_count + BATCH_SIZE)]
        x_minibatch, d_minibatch = make_minibatch(batch_1_list, X_train, sin_x_train)

        sess.run(train_step, feed_dict={x_: x_minibatch, d_: d_minibatch})
        loss_train = sess.run(loss, feed_dict={x_: x_minibatch, d_: d_minibatch})
        loss_train_sum += loss_train
        data_num_sum += len(x_minibatch)

    # loss_, accu_ = sess.run([loss, accuracy], feed_dict={x_: x_train, d_: d_train})
    print('epoch =' + str(epoch) + ' ,training loss =' + str(loss_train_sum / data_num_sum))    #test phase
    if epoch % 10 == 0:
        x_test_list = np.arange(len(X_test) - SEQUENCE_LEN) # x_test_list = np.array[0, 1, 2, ...., 149]
        x_test_batch, d_test_batch = make_minibatch(x_test_list, X_test, sin_x_test)
        loss_test, merged_, r2_ = sess.run([loss, merged, r2], feed_dict={x_: x_test_batch, d_: d_test_batch})
        print('test loss = ', str(loss_test), ', r2 = ', str(r2_))

        writer.add_summary(merged_, epoch)


