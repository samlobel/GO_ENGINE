import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.random_uniform([784,10],-0.05,0.05))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)


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

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x : batch_xs, y_ : batch_ys})
  if i % 25 == 0:
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


