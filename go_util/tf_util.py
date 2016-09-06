import tensorflow as tf
import numpy as np


def prefixize(suffix):
  return NAME_PREFIX + suffix


def weight_variable(shape, suffix):
  num_in = None
  num_out = None
  if len(shape) == 4:
    # conv
    num_in  = shape[0]*shape[1]*shape[2]
    num_out = shape[0]*shape[1]*shape[3]
  elif len(shape) == 2:
    num_in = shape[0]
    num_out = shape[1]
  else:
    raise Exception('shape should be either 0 or 2!')

  stddev_calc = (2.0 / (num_in + num_out))
  print("std_dev for weights is " + str(stddev_calc))

  if suffix is None or type(suffix) is not str:
    raise Exception("bad weight initialization")
  initial = tf.random_normal(shape, mean=0.0, stddev=stddev_calc)
  
  return tf.Variable(initial)

def bias_variable(shape, suffix):
  if suffix is None or type(suffix) is not str:
    raise Exception("bad bias initialization")  
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def conv2d(x,W, padding='SAME'):
  if not (padding=='SAME' or padding=='VALID'):
    print(padding)
    raise Exception("padding must be either SAME or VALID")
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding)

def conv2dResid(x,W,b, padding='SAME'):
  return (conv2d(x,W,padding) + b - x)


def softmax(softmax_input):
  flat_input = tf.reshape(softmax_input, (-1, BOARD_SIZE*BOARD_SIZE))
  input_after_softmax = tf.nn.softmax(flat_input)
  square_output = tf.reshape(input_after_softmax, (-1, BOARD_SIZE, BOARD_SIZE))
  return square_output

def softmax_with_temp(softmax_input, temperature, suffix):
  # NOTE THAT TEMP MUST BE A SCALAR.
  # if tf.less_equal(temperature, 0):
  #   print(temperature)
  #   raise Exception("temperature cannot be negative or zero! Printed above")
  if suffix is None or type(suffix) is not str:
    raise Exception("bad softmax initialization")

  exponents = tf.exp(tf.scalar_mul((1.0/temperature), softmax_input))
  sum_per_layer = tf.reduce_sum(exponents, reduction_indices=[1], keep_dims=True)
  softmax = tf.div(exponents, sum_per_layer)
  return softmax

def normalized_list_of_matrices(list_of_squares, suffix):
  to_divide = tf.reduce_sum(
    tf.reduce_sum(list_of_squares, reduction_indices=[1], keep_dims=True),
    reduction_indices=[2], keep_dims=True
  )
  # But what happens if there is a zero somewhere? That's confusing. Maybe
  # I need to make sure that's never going to happen? Since this is for softmax,
  # I should be fine. But maybe not. I really need to make sure that there's nothing 
  # where it sets everything to zero. Honestly, that's pretty tough.
  divided = tf.truediv(list_of_squares, to_divide)
  return divided

def mean_square_two_listoflists(l1, l2, suffix):
  squared_diff = tf.squared_difference(l1,l2)
  list_of_diffs_per_run = tf.reduce_sum(squared_diff, reduction_indices=[1,2])
  average_diff = tf.reduce_mean(list_of_diffs_per_run)
  return average_diff
