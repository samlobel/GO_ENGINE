from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from NNET.interface import GoBot

from go_util import util

from copy import deepcopy as copy

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# def conv2d_skip(x,W):
#   return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



x = tf.placeholder(tf.float32, [None, 81])
y_ = tf.placeholder(tf.float32, [None,1])


W_conv1 = weight_variable([3,3,1,16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x,[-1,9,9,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


W_conv2 = weight_variable([3,3,16,32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)


W_fc1 = weight_variable([9*9*32, 256])
b_fc1 = bias_variable([256])

h_conv2_flat = tf.reshape(h_conv2, [-1, 9*9*32])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# W_fc2 = weight_variable([256, 1])
W_fc2 = tf.Variable(tf.random_uniform([256,1],-0.1,0.1))
b_fc2 = bias_variable([1])

y_conv = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2) #between zero and one.

# W_fc3 = weight_variable([128, 1])
# b_fc3 = bias_variable([1])

# h_fc3

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
mean_square = tf.reduce_mean(tf.reduce_sum((y_ - y_conv)**2, reduction_indices=[1]))

error_metric = mean_square

train_step = tf.train.AdamOptimizer(1e-4).minimize(error_metric)


saver = tf.train.Saver()

sess = tf.Session()



"""
Here's how it is. If it's white's turn, what I was thinking was:
you simulate a move, and then take the move that's the highest chance of black losing.
BUT the problem with that is that now it's black's turn. That's a very different situation
than it being white's turn. So, I think what I should instead do is:
when it's white's turn, invert the board. Then, try to win with white. That's the spot
you would want to go. I think that's a much better idea.

"""


class Basic_ConvBot(GoBot):

  def __init__(self, load_path=None):
    GoBot.__init__(self)
    self.board_shape = (9,9)
    # saver = tf.train.Saver()
    # sess = tf.Session()
    self.load_path = load_path
    if load_path is None:
      init = tf.initialize_all_variables()
      sess.run(init)
      print("Initialized")
    else:
      saver.restore(sess, load_path)
      print("Loaded from path")
    
    # self.saver = saver
    # self.sess = sess

  def save_to_path(self, save_path=None):
    if save_path is None:
      raise Exception("Must save to specified path")
    full_save_path = saver.save(sess, save_path)
    print("Model saved to path: " + full_save_path)



  def evaluate_board(self, board_matrix):
    # Remember, evaluate_board is always run after a sample move. Also, it's
    # always run assuming that black just went. You need to do board*-1 if you
    # want to run it assuming white just went.
    # That makes it a little bit confusing, but it also is the only way that
    # makes sense. Plus, as long as I'm consistent, I think I'm fine.
    flattened = board_matrix.reshape((1,81))
    results = sess.run(y_conv, feed_dict={
        x : flattened
    })
    return results[0]

  # def train_on_random_board(self, num_moves):


  def get_on_policy_result_for_random_board(self, num_moves):
    """
    I really don't want to make a binary error here, that would make
    everything wrong.
    y_conv outputs the chance that, if black just went, it is going to
    win. Because we don't really use this on current board (instead we
    use it on the next), we usually are looking at a board one step
    in the future. And black is the default.
    So, if we're given a board, and it says it's black's turn:

    If it's WHITE's turn, the value of the board should represent
    the chance that BLACK wins from this position, if WHITE goes next.

    If it's BLACK's turn, the value of the board should represent
    the chance that WHITE wins from this position, if BLACK goes next.

    So, we always want the evaluator to be figuring out what BLACK's
    chances of winning are. That means that if it's BLACK's turn and we're
    evaluating a board (this means the AI is playing white):
    We should invert the board, and say that it's white's turn. What that's saying
    is that if we're trying to evaluate as a WHITE AI, act like the colors switch.
    This will make us always be 'playing' BLACK, according to the simulator.

    Then, you'll get a value for the board, which is the chance that "BLACK" wins,
    which is actually the chance that YOU win (white or black)

    So, you should similarly simulate on the flipped board, starting with the flipped
    starter. So, it should always be that BLACK just went, and WHITE is about to go.

    And then you'll get the expected chance of whoever you are winning, as well as the
    'on-policy' person who wins (because it outputs 1 when 'black' wins).

    So, 
    if next == 1:
      flipped = flip_board(original)
      valued = evaluate_board_with_first_move(flipped, -1)
    elif next == -1: (as it should be for our simulator)
      valued = evaluate_board_with_first_move(original, -1)
    else:
      raise Exception("Should never get here!")



    the move you would want is 
    then we want to multiply the board by -1
    """

    print("training on board with " + str(num_moves) + " random-spots")

    random_board, next_turn = util.generate_random_board((9,9), num_moves)
    if random_board is None:
      print("Looks like we got stuck somewhere, returning")
      return None, None

    simulator_input_board = None
    simulator_next_turn = None
    if next_turn == 1:
      # This means next is black, which is all wrong.
      # That means the initial guy was white. We're training as
      # if the last move was from black.
      # Because we're asking, is it a good idea for BLACK to get here?
      # So, the person going should be WHITE.
      simulator_input_board = -1 * random_board
      simulator_next_turn = -1
    elif next_turn == -1:
      simulator_input_board = copy(random_board)
      simulator_next_turn = -1
    else:
      raise Exception("Should never get here!")

    flattened_board = simulator_input_board.reshape((1,81))
    simulated_board_conclusion = self.get_results_of_board(
      simulator_input_board, np.zeros((9,9)), -1, []
    )
    if simulated_board_conclusion is None:
      print("Lasted more than 200.")
      return None, None
    winner = util.determine_winner(simulated_board_conclusion)


    # winner_np = np.asarray([[winner]])
    # print(winner_np)

    
    # self.sess.run(train_step, feed_dict={
    #   x : flattened_board,
    #   y_ : winner_np
    # })
    # print("trained")
    print("board and conclusion generated")
    return simulator_input_board.reshape(81), [winner]





    # # random_board, next_turn = generate_random_board((9,9), num_moves)
    # # board_to_score = next_turn * random_board #so it seems black.

    # flattened_board = board_to_score.reshape((1,81))

    # # estimate_board_value = self.evaluate_board(board_to_evaluate)
    # boards_natural_conclusion = self.get_results_of_board(copy(random_board),
    #                                           np.zeros((9,9)), next_turn, [])
    # true_winner = util.determine_winner(boards_natural_conclusion)
    # winner_np = np.asarray([[true_winner]])

    # self.sess.run(train_step, feed_dict={
    #   x : flattened_board, 
    #   y_ : winner_np
    # })
    # print("trained on a board.")


  def gather_input_matrices_from_num_moves_array(self, num_move_array):
    final_in = []
    final_out = []

    for num in num_move_array:
      board, result = self.get_on_policy_result_for_random_board(num)
      if (board is None) and (result is None):
        continue
      elif (board is None) or (result is None):
        raise Exception("If one is None, the other should be as well.")
      else:
        final_in.append(board)
        final_out.append(result)

    final_in = np.asarray(final_in)
    final_out = np.asarray(final_out)

    return final_in, final_out


  def train_from_num_moves_array(self, num_move_array):
    inputs, target_outputs = self.gather_input_matrices_from_num_moves_array(num_move_array)
    print(inputs)
    print(target_outputs)

    print("About to train")
    sess.run(train_step, feed_dict={
      x : inputs, 
      y_ : target_outputs
    })
    print("Trained.")
    return










  def evaluate_boards(self, board_matrices):
    results = sess.run(y_conv, feed_dict={
        x : board_matrices
      })
    return results

  def get_best_move(self, board_matrix, previous_board, current_turn):
    # print("Simulating turn...")
    valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
    
    if len(valid_moves) == 0:
      return None

    new_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_moves]
    valid_moves.append(None)
    new_boards.append(copy(board_matrix))



    if current_turn == -1:
      new_boards = [(-1 * board) for board in new_boards]

    value_of_new_boards = [self.evaluate_board(b) for b in new_boards]

    value_move_pairs = zip(value_of_new_boards, valid_moves)
    # print(value_move_pairs)
    # value_move_pairs = sorted(value_move_pairs)
    best_value_move_pair = max(value_move_pairs)
    # print(best_value_move_pair)

    # print("\n\n\ngetting best move from convnet")

    return best_value_move_pair[1]







  def get_results_of_board(self, board_matrix, previous_board, current_turn, move_list):

    # print("moves: " + str(len(move_list)))

    # if (len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None):
    #   return copy(board_matrix)

    # print("Simulating turn...")

    move_list=[]

    while not ((len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None)):
      # print ("move: " + str(len(move_list)))
      if len(move_list) > 200:
        return None
      best_move = self.get_best_move(board_matrix, previous_board, current_turn)
      if best_move is None:
        best_new_board = copy(board_matrix)
      else:
        best_new_board = util.update_board_from_move(board_matrix, best_move, current_turn)
      move_list.append(best_move)
      previous_board = board_matrix
      board_matrix = best_new_board
      current_turn *= -1

    return copy(board_matrix)


    # best_move = self.get_best_move(board_matrix, previous_board, current_turn)

    # if best_move is None:
    #   best_new_board = copy(board_matrix)
    # else:
    #   best_new_board = util.update_board_from_move(board_matrix, best_move, current_turn)

    # move_list.append(best_move)

    # return self.get_results_of_board(best_new_board, board_matrix, current_turn*-1, move_list)



def train_convbot_on_randoms(load_path=None):
  b_c = Basic_ConvBot(load_path=load_path)
  # for i in xrange(10):
  #   for num_moves in xrange(50, 75):
  #     b_c.train_on_random_board(num_moves)
  #   b_c.save_to_path(save_path="./saved_models/trained_on_" + str(i+1) + "_epochs.ckpt")
  num_array = range(50,65)
  b_c.train_from_num_moves_array(num_array)
  b_c.save_to_path('./saved_models/basic_convnet/trained_on_7_batch.ckpt')

  # print("Done with training it seems")






if __name__ == '__main__':
  # b_c = Basic_ConvBot()
  print("training!")
  train_convbot_on_randoms('./saved_models/basic_convnet/trained_on_6_batch.ckpt')
  # train_convbot_on_randoms()
  print("trained!")

  # starting_boards = []
  # for i in xrange(81):
  #   j = [0 for k in xrange(81)]
  #   j[i]=1
  #   starting_boards.append(j)

  # print(b_c.evaluate_boards(starting_boards))
  # # print(b_c.evaluate_boards(starting_boards))
  # starting_board = np.zeros((9,9))
  # result = b_c.get_results_of_board(starting_board, starting_board, 1, [])

  # print("\n\n\n results:\n")
  # print(result)




