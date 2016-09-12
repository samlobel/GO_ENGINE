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

import time

this_dir = os.path.dirname(os.path.realpath(__file__))


"""
CHECKS OUT, NOT FOR WHO WINS, BUT FOR WHAT VALUE IT SHOULD BE.


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
  x_ = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE, NUM_FEATURES], name=prefixize('x_policy'+str(i)))
  y_ = tf.placeholder(tf.float32, [None, 1], name=prefixize("y_"+str(i)))

  WC_1 = tf_util.weight_variable([3,3,NUM_FEATURES,20], suffix="WC_1"+str(i))
  bc_1 = tf_util.bias_variable([20], suffix="bc_1"+str(i))
  hc_1 = NONLIN(tf_util.conv2d(x_,WC_1) + bc_1) # not residual bc changes size

  WC_2 = tf_util.weight_variable([3,3,20,20], suffix="WC_2"+str(i))
  bc_2 = tf_util.bias_variable([20], suffix="bc_2"+str(i))
  hc_2 = NONLIN(tf_util.conv2dResid(hc_1,WC_2,bc_2))

  WC_3 = tf_util.weight_variable([3,3,20,20], suffix="WC_3"+str(i))
  bc_3 = tf_util.bias_variable([20], suffix="bc_3"+str(i))
  hc_3 = NONLIN(tf_util.conv2dResid(hc_2,WC_3,bc_3))

  FINAL_FEATS = 5

  WC_4 = tf_util.weight_variable([3,3,20,FINAL_FEATS], suffix="WC_4"+str(i))
  bc_4 = tf_util.bias_variable([FINAL_FEATS], suffix="bc_4"+str(i))
  hc_4 = NONLIN(tf_util.conv2d(hc_3,WC_4) + bc_4)

  hf_4 = tf.reshape(hc_4, [-1, BOARD_SIZE*BOARD_SIZE*FINAL_FEATS])

  Wfc_5 = tf_util.weight_variable([BOARD_SIZE*BOARD_SIZE*FINAL_FEATS,1], suffix="Wfc_5"+str(i))
  bfc_5 = tf_util.bias_variable([1], suffix="bfc_5"+str(i))

  output = tf.tanh(tf.matmul(hf_4, Wfc_5) + bfc_5)

  mse = tf.reduce_mean(tf.squared_difference(output, y_), name=prefixize('mean_square_value'+str(i)))

  # MomentumOptimizer = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_optimize_value'+str(i)))
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer(0.01, name=prefixize('gd_optimize_value'+str(i)))
  train_step = GradientDescentOptimizer.minimize(mse)

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
  network_dict['GradientDescentOptimizer'] = GradientDescentOptimizer
  network_dict['train_step'] = train_step

  print('network created')
  return network_dict



class Convbot_Clean(GoBot):
  """
  I need a way to use random instead of a not-well-trained model.
  """
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
      # init = tf.initialize_variables(self.var_list, name=prefixize("init"))
      init = tf.initialize_all_variables()
      self.sess.run(init)
      print("Initialized randomly")
      self.folder_name = NAME_PREFIX
      self.batch_num = 0
      self.well_trained = [False for i in self.vn_arr]
      self.write_config(self.well_trained)
    else:
      self.folder_name = NAME_PREFIX
      self.batch_num = batch_num
      load_path = make_path_from_folder_and_batch_num(batch_num)
      print("initializing from path: " + str(load_path))
      self.saver.restore(self.sess, load_path)
      self.well_trained = self.load_config()

  def get_tf_nodes(self):
    vn_arr = self.vn_arr
    end = [[subarr[key] for key in subarr] for subarr in vn_arr]
    flat = util.flatten_list(end)
    filtered = [v for v in flat if type(v) == tf.Variable]
    # momentum_nodes = [subarr['MomentumOptimizer'] for subarr in vn_arr]
    # for m in momentum_nodes:
    #   names = m.get_slot_names()
    #   variables = [m.]
    # momentum_node_names = [(subarr['MomentumOptimizer'], subarr['MomentumOptimizer'].get_slot_names()) for subarr in vn_arr]

    # print(filtered)
    return filtered

  def write_config(self, to_write):
    if len(to_write) != len(self.vn_arr):
      raise Exception('Should be same length as vn_arr')
    for i in to_write:
      if i != True and i != False:
        raise Exception('should all be true or false!')
    string = json.dumps(to_write)
    with open(os.path.join(this_dir, 'saved_models', NAME_PREFIX, 'config.json'), 'w') as f:
      f.write(string)
    print ('config written: ' + string)
    return


  def load_config(self):
    try:
      with open(os.path.join(this_dir, 'saved_models', NAME_PREFIX, 'config.json'), 'r') as f:
        contents = f.read()
        if not contents:
          print('NADA')
          return None
        print('LOADING CONTENTS')
        obj = json.loads(contents)
        return obj
    except Exception:
      print ("No Config to load!")
      obj = [False for f in self.vn_arr]
      return obj

  def save_in_next_slot(self):
    if self.folder_name is None:
      raise Exception("can't save if we don't know the folder name!")
    load_batch = self.batch_num
    save_batch = load_batch + 1
    save_path = make_path_from_folder_and_batch_num(save_batch)
    print("SAVE PATH: " + str(save_path))
    print(save_path)
    set_largest_batch_in_folder(save_batch)
    self.saver.save(self.sess, save_path)
    self.batch_num = save_batch
    # print("Model saved to path: " + str(save_path))

    return self.folder_name, save_batch


  def set_up_bot_network(self):
    vn_arr = [make_value_network(i) for i in range(NUM_NETWORKS)]
    self.vn_arr = vn_arr

  def turn_number_to_network_index(self, turn_number):
    # Hard coding is okay here.
    if type(turn_number) is not int:
      raise Exception('turn number should be an int!')
    if turn_number < 10:
      return 0
      # return self.vn_arr[0]
    if turn_number < 20:
      return 1
      # return self.vn_arr[1]
    return 2
    # return self.vn_arr[2]
  def random_number_in_next_slot(self, turn_number):
    if type(turn_number) is not int:
      raise Exception('turn number should be an int!')
    if self.turn_number_to_network_index(turn_number) == 2:
      return None
    if self.turn_number_to_network_index(turn_number) == 1:
      return int(20 + (10*random.random()))
    if self.turn_number_to_network_index(turn_number) == 0:
      return int(10 + (10*random.random()))
    raise Exception('Should not get here in random_number_in_next_slot')


    pass

  def get_best_move(self, board_matrix, previous_board, current_turn, turn_number):
    if (board_matrix is None) or not (current_turn in (-1,1)) or (turn_number is None):
      raise Exception("Invalid inputs to get_best_move.")
    # Not sure if I need these, but...
    previous_board = copy(previous_board)
    board_matrix = copy(board_matrix)
    valid_sensible_moves = list(util.output_all_valid_sensible_moves(board_matrix, previous_board, current_turn))
    if len(valid_sensible_moves) == 0:
      print('no more valid sensible moves for color: ' + str(current_turn))
      return None

    inputs_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_sensible_moves]
    # Remember, the new previous is the original now!
    inputs = np.asarray([board_to_input_transform_value(board, board_matrix, current_turn * -1) for board in inputs_boards], dtype=np.float32)
    network_num = self.turn_number_to_network_index(turn_number)
    network = self.vn_arr[network_num]
    outputs = self.sess.run(network['output'], feed_dict={
      network['x_'] : inputs
    })
    zipped = zip(outputs, valid_sensible_moves)
    print(zipped)
    best_move = max(zipped)[1]
    return best_move

  def get_training_move(self, board_matrix, previous_board, current_turn, turn_number, softmax_temp=0.5):
    # Should I have softmax here? I feel like maybe I should. But it needs to be a 
    # lot colder than I previously had it. Maybe like temperature of 0.5
    # That would mean that a move with value 0 versus value 1 would be 1.6 times 
    # more likely. That's not very much.
    # Oh, it's energy/temp. So 0.5 is times two. So ten times as good. That's
    # seriously not that much, but good enough I guess.
    # e^0 / (e^0+e^1) --> 1/3.7 vs 2.7/3.7 --> 2.7 times more likely.
    # e^0 / (e^0+e^0.5) --> 1/3.7 vs 2.7/3.7 --> 2.7 times more likely.
    if (board_matrix is None) or not (current_turn in (-1,1)) or (turn_number is None):
      raise Exception("Invalid inputs to get_best_move.")
    # Not sure if I need these, but...
    previous_board = copy(previous_board)
    board_matrix = copy(board_matrix)
    network_num = self.turn_number_to_network_index(turn_number)
    network_is_trained = self.well_trained[network_num]
    if not network_is_trained:
      # print('using random move.')
      training_move = util.output_one_valid_sensible_move(board_matrix, previous_board, current_turn)
      return training_move
    # Calculate all of them, put through softmax. 
    valid_sensible_moves = list(util.output_all_valid_sensible_moves(board_matrix, previous_board, current_turn))
    if len(valid_sensible_moves) == 0:
      print ("No moves, in softmax.")
      return None    
    inputs_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_sensible_moves]
    # Remember, the new previous is the original now!
    inputs = np.asarray([board_to_input_transform_value(board, board_matrix, current_turn * -1) for board in inputs_boards], dtype=np.float32)
    network = self.vn_arr[network_num]
    outputs = self.sess.run(network['output'], feed_dict={
      network['x_'] : inputs
    })
    index_from_softmax = util.get_softmax_index(outputs, softmax_temp)
    training_move = valid_sensible_moves[index_from_softmax]
    return training_move

  def learn_from_list_of_games(self, list_of_learn_objects, vn_number):
    # print(list_of_learn_objects)
    start_time = time.time()
    # print('extracting boards')
    np_boards = [np.asarray(x['board'], dtype=np.float32) for x in list_of_learn_objects]
    # print('extracting turns')
    turns = [x['turn'] for x in list_of_learn_objects]
    inputs = np.asarray([board_to_input_transform_value(np_boards[i],None,turns[i]) for i in range(len(list_of_learn_objects))])
    vals = np.asarray([[x['val']] for x in list_of_learn_objects])
    vn = self.vn_arr[vn_number]
    _mse, _nothing = self.sess.run([vn['mse'], vn['train_step']], feed_dict={
      vn['x_'] : inputs,
      vn['y_'] : vals
    })
    print("MSE: " + str(_mse))
    end_time = time.time()
    # print('Time to learn from one sublist: ' + str(end_time - start_time))
    return _mse



def board_to_input_transform_value(board_matrix, previous_board, current_turn):
  """
  Outputs the input to the value network.
  Remmeber, it's important that a zero doesn't have two meanings...
  """
  # print('transforming board')
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
  # print('shape of feature array: ' + str(feature_array.shape) + ' should be [N,N,?]')
  return feature_array


def make_path_from_folder_and_batch_num(batch_num):
  save_path = os.path.join(this_dir,'saved_models', NAME_PREFIX)
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


def get_largest_batch_in_folder():
  folder = os.path.join(this_dir, 'saved_models', NAME_PREFIX)
  filename = os.path.join(folder, 'largest.txt')
  f = open(filename, 'r')
  content = f.read()
  content = content.strip()
  latest = int(content)
  f.close()
  return latest



def set_largest_batch_in_folder(batch_num):
  folder = os.path.join(this_dir,'saved_models', NAME_PREFIX)
  filename = os.path.join(folder, 'largest.txt')
  f = open(filename, 'w')
  f.write(str(batch_num))
  f.close()

def largest_batch_file_exists():
  folder = os.path.join(this_dir,'saved_models', NAME_PREFIX)
  filename = os.path.join(folder, 'largest.txt')
  return os.path.isfile(filename)

def get_best_bot():
  largest_exists = largest_batch_file_exists()
  if not largest_exists:
    print('initializing from scratch')
    return Convbot_Clean()
  largest = get_largest_batch_in_folder()
  print('initializing largest from checkpoint, num: ' + str(largest))
  return Convbot_Clean(batch_num=largest)









def get_results_of_self_play(BOT, board_matrix, previous_board, current_turn, turn_number):
  """
  I should write two, one that plays all the way to the end, and one that plays until the
  next trained convnet. Or, why even do the first? Who knows!
  As long as I train it backwards, it doesn't really matter.
  I should get a few squares deep into the next turn, otherwise I'm only looking
  at how it performs on the border, which isn't the place where most of the 
  sampling comes into play.
  So, maybe I should do something along the lines of 
  Maybe I should write something that gives a random index in the next section,
  or None if it's last. That's a good enough idea for me.
  """
  shape = board_matrix.shape
  max_moves = shape[0]*shape[1]*4 #for 5x5, that's 100.
  all_moves = []
  # current_vn_index = BOT.turn_number_to_network_index(turn_number)
  current_board = np.copy(board_matrix)
  max_turn_number = BOT.random_number_in_next_slot(turn_number)
  while True:
    if max_turn_number != None and turn_number >= max_turn_number:
      print('game hit max_turn_number, truncating value. At ' + str(max_turn_number))
      """Remember, this gives the value for the person before this turn.
      So if -1 is about to go, that means this value is the chance 1 is going to
      win. Which is the same as determine_winner. In other words, to get the winner,
      you do (-1*current_turn)*value. The first part gives you who the value is for.
      If it's for 1, then a 1 val means 1 wins and a -1 val means 1 loses.
      If it's for -1, then a 1 val means 1 loses, and a -1 val means 1 wins.
      """
      vn_input = board_to_input_transform_value(current_board, previous_board, current_turn)
      # print('after board_to_input_transform_value')
      vn_input = np.asarray([vn_input]) #Because there is only one.... ANNOYING BUT OH WELL
      # print('after vn_input to array')
      current_vn = BOT.turn_number_to_network_index(turn_number)
      # print("VN NUMBER THAT WE'RE GETTING OUTPUT FROM: " + str(current_vn))
      current_vn = BOT.vn_arr[current_vn]
      # print('after BOT.turn_number_to_network_index')
      output_val = BOT.sess.run(current_vn['output'], feed_dict={
        current_vn['x_'] : vn_input
      })
      # print("after sess.run current_vn['output'']")

      # Again, if current_turn is -1, and we get a 1, that means 1 wins. Confirmed.
      winner_val = (current_turn*-1)*output_val[0][0]
      return winner_val # The value of the game, not to the bot who's going necessarily.

    if len(all_moves) > max_moves:
      return None
    if (len(all_moves) >= 2) and all_moves[-1] is None and all_moves[-2] is None:
      # print("Game is over!")
      break
    this_move = BOT.get_training_move(current_board, previous_board, current_turn, turn_number, softmax_temp=0.5)
    new_board = util.update_board_from_move(current_board, this_move, current_turn)
    
    all_moves.append(this_move)
    previous_board = current_board
    current_board = new_board
    current_turn *= -1
    turn_number+= 1
    # debug_print('move length now is: ' + str(len(all_moves)))

  # debug_print("Game completed, lasted " + str(len(all_moves)) + " more turns")
  winner = util.determine_winner(current_board)
  return winner








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

