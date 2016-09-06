from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from NNET.interface import GoBot
from go_util import util, tf_util
from copy import deepcopy as copy
import random
import json



"""
This would be a good time to implement the other input layers.
Namely, the constant zeros, the constant ones, and the current-color.
Especially current-color.

You know, honestly I think that it makes more sense to filter based on
how much of the board has stuff on it, because what I'm trying to
do is make the distributions tighter, and that's the determination of
the distributions...
On the other hand, that makes me lose the part of it that easily trains
the last layer first, and then goes backwards...

I don't know. This is tough. Maybe I should just do both? That's a good idea.

I'll start with the turn-based one.

Also, just the value network. Who cares about policy!

I would like to put somet thought into how to make saving clean.
I used to do it because I had a self-play training thing, but that's out
the window. So, back in business baby.
Maybe use the prefixize as the folder_name. But the reason I didn't want to
do that is that I sometimes change things as I go and want to save the old ones.
But that's stupid, because then they don't load. 

ACTUALLY, I DON'T NEED A FOLDER NAME AT ALL!

"""


NAME_PREFIX='fivebot_clean_'
BOARD_SIZE = 5
# NUM_FEATURES = 14
NUM_FEATURES = 14
MAX_TO_KEEP = 2
KEEP_CHECKPOINT_EVERY_N_HOURS = 0.1
NUM_NETWORKS = 3

NONLIN = tf.nn.elu



if NUM_NETWORKS != 3:
  print('REMEMBER TO CHANGE EVERYTHING!')
  raise Exception('^^')


def prefixize(suffix):
  return NAME_PREFIX + suffix


def make_value_network(i):
  # I is just for unique naming. Not sure if I need it, but I probably do.
  print('creating network')
  x_ = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE, NUM_FEATURES], name=prefixize('x_policy'))
  y_ = tf.placeholder(tf.float32, [None, 1], name=prefixize("y_"))

  WC_1 = tf_util.weight_variable([3,3,NUM_FEATURES,20], suffix="WC_1")
  bc_1 = tf_util.bias_variable([20], suffix="bc_1")
  hc_1 = NONLIN(tf_util.conv2d(x_,WC_1) + bc_1) # not residual bc changes size

  WC_2 = tf_util.weight_variable([3,3,20,20], suffix="WC_2")
  bc_2 = tf_util.bias_variable([20], suffix="bc_2")
  hc_2 = NONLIN(tf_util.conv2dResid(hc_1,WC_2,bc_2))

  WC_3 = tf_util.weight_variable([3,3,20,20], suffix="WC_3")
  bc_3 = tf_util.bias_variable([20], suffix="bc_3")
  hc_3 = NONLIN(tf_util.conv2dResid(hc_2,WC_3,bc_3))

  FINAL_FEATS = 5

  WC_4 = tf_util.weight_variable([3,3,20,FINAL_FEATS], suffix="WC_4")
  bc_4 = tf_util.bias_variable([FINAL_FEATS], suffix="bc_4")
  hc_4 = NONLIN(tf_util.conv2d(hc_3,WC_4) + bc_4)

  hf_4 = tf.reshape(hc_4, [-1, BOARD_SIZE*BOARD_SIZE*FINAL_FEATS])

  Wfc_5 = tf_util.weight_variable([BOARD_SIZE*BOARD_SIZE*FINAL_FEATS,1], suffix="Wfc_5")
  bfc_5 = tf_util.bias_variable([1], suffix="bfc_5")

  output = tf.tanh(tf.matmul(hf_4, Wfc_5) + bfc_5)

  mse = tf.reduce_mean(tf.squared_difference(output, y_), name=prefixize('mean_square_value'))

  MomentumOptimizer = tf.train.MomentumOptimizer(0.01, 0.9, name=prefixize('momentum_optimize_value'))
  train_step = MomentumOptimizer.minimize(mse)

  network_dict = {}
  network_dict['x_'] = x_
  network_dict['y_'] = y_

  network_dict['WC_1'] = WC_1
  network_dict['bc_1'] = bc_1
  network_dict['hc_1'] = hc_1
  network_dict['WC_2'] = WC_2
  network_dict['bc_2'] = bc_2
  network_dict['hc_2'] = hc_2
  network_dict['WC_3'] = WC_3
  network_dict['bc_3'] = bc_3
  network_dict['hc_3'] = hc_3
  network_dict['WC_4'] = WC_4
  network_dict['bc_4'] = bc_4
  network_dict['hc_4'] = hc_4

  network_dict['hf_4'] = hf_4  
  network_dict['Wfc_5'] = Wfc_5
  network_dict['bfc_5'] = bfc_5
  network_dict['output'] = output
  network_dict['mse'] = mse
  network_dict['train_step'] = train_step

  print('network created')
  return network_dict



class Convbot_Clean(GoBot):
  def __init__(self,  batch_num=0):
    GoBot.__init__(self)
    self.set_up_bot_network()
    self.board_shape = (BOARD_SIZE,BOARD_SIZE)
    self.sess = tf.Session()
    tf_nodes = self.get_tf_nodes()
    self.var_list = tf_nodes
    self.saver = tf.train.Saver(var_list=tf_nodes, max_to_keep=MAX_TO_KEEP,
      keep_checkpoint_every_n_hours = KEEP_CHECKPOINT_EVERY_N_HOURS,
      name=prefixize("saver"))

    if batch_num == 0:
      init = tf.initialize_variables(self.var_list, name=prefixize("init"))
      self.sess.run(init)
      print("Initialized randomly")
      self.folder_name = NAME_PREFIX
      self.batch_num = 0
    else:
      self.folder_name = NAME_PREFIX
      self.batch_num = batch_num
      load_path = make_path_from_folder_and_batch_num(batch_num)
      print("initializing from path: " + str(load_path))
      saver.restore(self.sess, load_path)

  def get_tf_nodes(self):
    vn_arr = self.vn_arr
    end = [[subarr[key] for key in subarr] for subarr in vn_arr]
    flat = util.flatten_list(end)
    filtered = [v for v in flat if type(v) == tf.Variable]
    print(filtered)
    return filtered






  def save_in_next_slot(self):
    if self.folder_name is None:
      raise Exception("can't save if we don't know the folder name!")
    load_batch = self.batch_num
    save_batch = load_batch + 1
    save_path = make_path_from_folder_and_batch_num(self.folder_name, save_batch)
    print(save_path)
    saver.save(self.sess, save_path)
    # print("Model saved to path: " + str(save_path))
    return self.folder_name, save_batch


  def set_up_bot_network(self):
    vn_arr = [make_value_network(i) for i in range(NUM_NETWORKS)]
    self.vn_arr = vn_arr

  def turn_number_to_network(self, turn_number):
    # Hard coding is okay here.
    if type(turn_number) is not int:
      raise Exception('turn number should be an int!')
    if turn_number <= 8:
      return self.vn_arr[0]
    if turn_number <= 15:
      return self.vn_arr[1]
    return self.vn_arr[2]

  def get_best_move(self, board_matrix, previous_board, current_turn, turn_number):
    if (board_matrix is None) or not (current_turn in (-1,1)) or (turn_number is None):
      raise Exception("Invalid inputs to get_best_move.")
    # Not sure if I need these, but...
    previous_board = copy(previous_board)
    board_matrix = copy(board_matrix)
    valid_sensible_moves = list(util.output_all_valid_sensible_moves(board_matrix, previous_board, current_turn))

    inputs_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_sensible_moves]
    # Remember, the new previous is the original now!
    inputs = np.asarray([board_to_input_transform_value(board, board_matrix, current_turn * -1) for board in inputs_boards], dtype=np.float32)
    network = self.turn_number_to_network(turn_number)
    outputs = self.sess.run(network['output'], feed_dict={
      network['x_'] : inputs
    })
    zipped = zip(outputs, valid_sensible_moves)
    print(zipped)
    best_move = max(zipped)[1]
    return best_move



def board_to_input_transform_value(board_matrix, previous_board, current_turn):
  """
  Outputs the input to the value network.
  Remmeber, it's important that a zero doesn't have two meanings...
  """
  print('transforming board')
  if current_turn not in (-1,1):
    raise Exception('current turn must be -1 or 1. Instead it is: ' + str(current_turn))

  legal_sensible_move_map = util.output_valid_sensible_moves_boardmap(board_matrix, previous_board, current_turn)
  liberty_map = util.output_liberty_map(board_matrix)
  
  split_board_arr = util.split_board(board_matrix, current_turn)
  split_liberties_arr = util.split_liberties(liberty_map, current_turn)

  ones_layer = np.ones_like(board_matrix)
  zeros_layer = np.zeros_like(board_matrix)

  black_turn_layer = np.ones_like(board_matrix) if current_turn == 1 else np.zeros_like(board_matrix)
  white_turn_layer = np.ones_like(board_matrix) if current_turn == -1 else np.zeros_like(board_matrix)


  feature_list = []
  feature_list.extend(split_board_arr)
  feature_list.extend(split_liberties_arr)
  feature_list.append(black_turn_layer)
  feature_list.append(white_turn_layer)
  feature_list.append(legal_sensible_move_map)
  feature_list.append(ones_layer)
  feature_list.append(zeros_layer)
  

  feature_array = np.asarray(feature_list, dtype=np.float32)

  feature_array = feature_array.T
  print('shape of feature array: ' + str(feature_array.shape) + ' should be [N,N,?]')
  return feature_array


def make_path_from_folder_and_batch_num(batch_num):
  save_path = os.path.join(this_dir,NAME_PREFIX)
  file_name = 'v' + str(batch_num) + '.ckpt'
  save_path = os.path.join(save_path, file_name)
  return save_path

def create_random_starter():
  if largest_batch_file_exists(folder_name):
    print('already exists')
    return
  print("creating random starter in folder")
  new_cb = Convbot_Clean(batch_num=0)
  new_cb.save_in_next_slot()
  print("random starter created and saved")
  set_largest_batch_in_folder(folder_name, 1)
  new_cb.sess.close()


def set_largest_batch_in_folder(batch_num):
  folder = os.path.join(this_dir,NAME_PREFIX)
  filename = os.path.join(folder, 'largest.txt')
  f = open(filename, 'w')
  f.write(str(batch_num))
  f.close()

def largest_batch_file_exists(f_name):
  folder = os.path.join(this_dir, 'saved_models', 'convnet_pol_val_good')
  filename = os.path.join(folder, f_name,'largest.txt')
  return os.path.isfile(filename)









  # # legal_moves_map = util.output_valid_moves_boardmap(board_matrix, all_previous_boards, current_turn)


  # # You know, I really don't like this whole reshaping nonsense. I'm going to skip it I think.
  

  # legal_sensible_move_map = util.output_valid_sensible_moves_boardmap(board_matrix, all_previous_boards, current_turn)
  # # print('input and legal moves: ')
  # # print(board_matrix)
  # # print(all_previous_boards)
  # # print(legal_sensible_move_map)

  # liberty_map = util.output_liberty_map(board_matrix)


  # # This is about to become a DOOZY
  # board_this_player, board_blank, board_other_player = split_board(board_matrix, current_turn)
  # where_me_gt_two, where_me_two, where_me_one,\
  #     where_they_one, where_they_two, where_they_gt_two =\
  #                       split_liberties(liberty_map, current_turn)

  # ones_layer = np.ones_like(board_matrix)
  # zeros_layer = np.zeros_like(board_matrix)

  # feature_array = np.asarray([
  #   board_this_player,
  #   board_blank, 
  #   board_other_player,
  #   where_me_gt_two,
  #   where_me_two,
  #   where_me_one,
  #   where_they_one,
  #   where_they_two,
  #   where_they_gt_two,
  #   legal_sensible_move_map,
  #   ones_layer,
  #   zeros_layer
  # ], dtype=np.float32)

  # feature_array = feature_array.T
  # print('shape of feature array: ' + str(feature_array.shape) + ' should be [N,N,?]')
  # return feature_array

  # # feature_array = None
  # # if current_turn == 1:
  # #   feature_array = np.asarray([board_matrix, legal_sensible_move_map, liberty_map])
  # # else:
  # #   feature_array = np.asarray([-1 *board_matrix, legal_sensible_move_map, -1 * liberty_map])

  # # SHIT. This isn't exactly right. This has shape (3, 5, 5) I need something that is 
  # # of the form: (5,5,3). Looks like Transpose somehow does EXACTLY what I want.
  # feature_array = feature_array.T

  # flattened_array = feature_array.flatten()
  # for i in range(len(flattened_array)):
  #   if flattened_array[i] < 0:
  #     print("Flattened array element less than zero!")
  #     # print(feature_array)
  #     print(flattened_array[i])
  #     raise Exception('no element can be less than zero!')
  # # print('checks out')
  # return feature_array

  # # feature_array = feature_array.T
  # # flattened_input = feature_array.reshape((1, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES))
  
  # # return flattened_input

