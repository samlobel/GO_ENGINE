import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

in_size = 784
hid_size = 200
out_size=10
activation_function = tf.nn.relu

x = tf.placeholder(tf.float32, [None, in_size])

W_1 = tf.Variable(tf.random_uniform([in_size,hid_size],-0.05,0.05))
b_1 = tf.Variable(tf.zeros([hid_size]))

W_2 = tf.Variable(tf.random_uniform([hid_size,out_size],-0.05,0.05))
b_2 = tf.Variable(tf.zeros([out_size]))

# y = tf.nn.softmax(tf.matmul(x,W)+b)

h_inp = activation_function(tf.matmul(x,W_1) + b_1)
y = tf.nn.softmax(tf.matmul(h_inp, W_2)+b_2)


y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
mean_square = tf.reduce_mean(tf.reduce_sum((y_ - y)**2, reduction_indices=[1]))

use_err = cross_entropy

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(use_err)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x : batch_xs, y_ : batch_ys})
  if i % 25 == 0:
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))




