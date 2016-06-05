from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from NNET.interface import GoBot

from go_util import util

from copy import deepcopy as copy
import random
import json


this_dir = os.path.dirname(os.path.realpath(__file__))




TRAIN_OR_TEST = "TRAIN"
# TRAIN_OR_TEST = "TEST"


NAME_PREFIX='fivebot_policy_value_better_'

BOARD_SIZE = 5

# NUM_FEATURES = 3
NUM_FEATURES = 12


MAX_TO_KEEP = 2
KEEP_CHECKPOINT_EVERY_N_HOURS = 0.1

GLOBAL_TEMPERATURE = 1.0 #Since values range from -1 to 1,
            # passing this through softmax with temp "2" will give the
            # difference between best and worst as 2.7 times as likely.
            # That's a pretty damn high temperature. But, it's good,
            # because at the beginning that means it will lead to spots
            # that have a lot of good moves following them. Nice.
            # temp 1 gives 9.7 times likelihood.
            # temp 1.5 gives like 5 times.

NUM_POLICY_GAMES_TO_SIMULATE_PER_BOARD = 5

REGULARIZER_CONSTANT = 0.0001


"""
Let's make a list of features that will make this the most
powerful neural network in town!

First is the raw input representation, with 0 for empty, 1 for your stone,
and -1 for their stone.

Next, there's a feature layer that will be GREAT for policy network: legal
moves. It's 0 if something is an illegal move, and a 1 if it is legal
for the current player. Heres a question: for Value network, is it legal
moves for the turn you're looking at or the one before? I think its the
turn you're looking at.

Then, there are the liberty layers. I think that a good idea would be 
positive liberties if its your group, and negative liberties if its their
group. And zero if its nobodys group.

Interesting. I think the reason that they break it up the way they do is because
linear rectifiers throw away negative numbers. That's not a bad idea.
But, it does seeme like its silly to break the board into three features that say
the same thing. Can't you just do that by having more hidden features? And
wont weights be able to send it down if it needs to?

NOPE! That was my key mistake. They break it down that way so that the filter
can tell the difference bewteen an edge and a zero-padded spot. Before,
it thought that an edge had room, which is why it always stuck to them.
Now it's game time.


FEATURE 1: RAW INPUT
FEATURE 2: VALID MOVES
FEATURE 3: LIBERTY MAP

NOT NO MORE!

"""


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
  
  return tf.Variable(initial, name=prefixize(suffix))

def bias_variable(shape, suffix):
  if suffix is None or type(suffix) is not str:
    raise Exception("bad bias initialization")  
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial, name=prefixize(suffix))


def conv2d(x,W, padding='VALID'):
  if not (padding=='SAME' or padding=='VALID'):
    print(padding)
    raise Exception("padding must be either SAME or VALID")
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding)

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
  softmax = tf.div(exponents, sum_per_layer, name=prefixize(suffix))
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
  divided = tf.truediv(list_of_squares, to_divide, name=prefixize(suffix))
  return divided

def mean_square_two_listoflists(l1, l2, suffix):
  squared_diff = tf.squared_difference(l1,l2)
  list_of_diffs_per_run = tf.reduce_sum(squared_diff, reduction_indices=[1,2])
  average_diff = tf.reduce_mean(list_of_diffs_per_run, name=prefixize(suffix))
  return average_diff







"""
INPUTS POLICY: 
x_policy : These are board inputs. The inputs are used for training and testing. They are
a list of boards, where each square of the board has many features.

legal_sensible_moves_map_policy: we use this as a way to output a valid move. It is 1 wherever a
move is valid, and 0 wherever a move is invalid. Ideally the network would 
predict a probability of 0 for each of these spots that is 0.

The y_ output is used for training just the value network
the computed_values_for_moves goes into a softmax to create the target
output for the policy network.

softmax_output_goal_policy: This is the goal of what you want to acheive. You should 
pass in 1 for wherever you actually want to go, and 0 elsewhere. If you win, you should
minimize MSE, if you lose, you should maximize it.
"""

# x_value = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES], name=prefixize('x_value'))

x_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE, NUM_FEATURES], name=prefixize('x_policy'))

legal_sensible_moves_map_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE], name=prefixize('legal_sensible_moves_map_policy'))
softmax_output_goal_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE], name=prefixize('softmax_output_goal_policy'))


"""
POLICY NETWORK VARIABLES
All variables used for the policy network, including outputs.
Note that the optimizers don't have names, for either.

These should all end in the word 'policy'.
"""


# x_image_policy = tf.reshape(x_policy, (-1,BOARD_SIZE,BOARD_SIZE,NUM_FEATURES), name=prefixize("x_image_policy"))

W_conv1_policy = weight_variable([3,3,NUM_FEATURES,20], suffix="W_conv1_policy")
b_conv1_policy = bias_variable([20], suffix="b_conv1_policy")
h_conv1_policy = tf.nn.elu(conv2d(x_policy, W_conv1_policy, padding="SAME") + b_conv1_policy, name=prefixize("h_conv1_policy"))

W_conv2_policy = weight_variable([3,3,20,20], suffix="W_conv2_policy")
b_conv2_policy = bias_variable([20], suffix="b_conv2_policy")
h_conv2_policy = tf.nn.elu(conv2d(h_conv1_policy, W_conv2_policy, padding="SAME") + b_conv2_policy, name=prefixize("h_conv2_policy"))

W_conv3_policy = weight_variable([3,3,20,20], suffix="W_conv3_policy")
b_conv3_policy = bias_variable([20], suffix="b_conv3_policy")
h_conv3_policy = tf.nn.elu(conv2d(h_conv2_policy, W_conv3_policy, padding="SAME") + b_conv3_policy, name=prefixize("h_conv3_policy"))

W_conv4_policy = weight_variable([3,3,20,1], suffix="W_conv4_policy")

convolved_4_policy = conv2d(h_conv3_policy, W_conv4_policy, padding="SAME")
convolved_4_reshaped_policy = tf.reshape(convolved_4_policy,(-1, BOARD_SIZE, BOARD_SIZE))
bias_4_policy = bias_variable([BOARD_SIZE,BOARD_SIZE], suffix="square_bias_policy")
input_to_softmax_policy = convolved_4_reshaped_policy + bias_4_policy

softmax_output_policy = softmax(input_to_softmax_policy)

legal_softmax_output_policy = tf.mul(softmax_output_policy, legal_sensible_moves_map_policy)

normalized_legal_softmax_output_policy = normalized_list_of_matrices(legal_softmax_output_policy, suffix="normalized_output_policy")

# mean_square_policy = tf.reduce_mean(tf.squared_difference(l1,l2), name=prefixize(suffix))
mean_square_policy = mean_square_two_listoflists(softmax_output_policy, softmax_output_goal_policy, suffix="mean_square_policy")

l2_loss_layer_1_policy = tf.nn.l2_loss(W_conv1_policy, name=prefixize('l2_layer1_policy'))
l2_loss_layer_2_policy = tf.nn.l2_loss(W_conv2_policy, name=prefixize('l2_layer2_policy'))
l2_loss_layer_3_policy = tf.nn.l2_loss(W_conv3_policy, name=prefixize('l2_layer3_policy'))
l2_loss_layer_4_policy = tf.nn.l2_loss(W_conv4_policy, name=prefixize('l2_layer4_policy'))

l2_error_total_policy = REGULARIZER_CONSTANT * (l2_loss_layer_1_policy + l2_loss_layer_2_policy + 
                    l2_loss_layer_3_policy + l2_loss_layer_4_policy)

negative_mean_square_policy = -1.0 * mean_square_policy

AdamOptimizer_policy = tf.train.AdamOptimizer(1e-4)
# tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_policy'))
train_step_winning_policy = AdamOptimizer_policy.minimize(mean_square_policy)

train_step_losing_policy = AdamOptimizer_policy.minimize(negative_mean_square_policy)

# MomentumOptimizer_forlosing_policy = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_policy'))

MomentumOptimizer_learnmoves_policy = tf.train.MomentumOptimizer(0.001, 0.1, name=prefixize('momentum_learnmoves_policy'))
train_step_learnmoves_policy = MomentumOptimizer_learnmoves_policy.minimize(mean_square_policy)


GradientOptimizer_l2_policy = tf.train.GradientDescentOptimizer(0.0005, name=prefixize('l2_optimizer_policy'))
train_step_l2_reg_policy = GradientOptimizer_l2_policy.minimize(l2_error_total)



# total_error = mean_square_policy + l2_error_total

# I think I should separate the L2 and the mean_square errors.


# MomentumOptimizer_policy = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_policy'))
# train_step_winning_policy = MomentumOptimizer_policy.minimize(mean_square_policy)

# train_step_losing_policy = MomentumOptimizer_policy.minimize(negative_mean_square_policy)

# # MomentumOptimizer_forlosing_policy = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_policy'))

# MomentumOptimizer_learnmoves_policy = tf.train.MomentumOptimizer(0.0005, 0.1, name=prefixize('momentum_learnmoves_policy'))
# train_step_learnmoves_policy = MomentumOptimizer_policy.minimize(mean_square_policy)



"""
Inputs VALUE
"""

x_value = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE, NUM_FEATURES], name=prefixize('x_value'))
legal_sensible_moves_map_value = tf.placeholder(tf.float32, [None, BOARD_SIZE,BOARD_SIZE], name=prefixize('legal_sensible_moves_map_value'))
target_value = tf.placeholder(tf.float32, [None, 1], name=prefixize("target_value"))


"""
VALUE NETWORK VARIABLES
"""

# x_image_value = tf.reshape(x_value, (-1,BOARD_SIZE,BOARD_SIZE,NUM_FEATURES), name=prefixize("x_image_value"))

W_conv1_value = weight_variable([3,3,NUM_FEATURES,20], suffix="W_conv1_value")
b_conv1_value = bias_variable([20], suffix="b_conv1_value")
h_conv1_value = tf.nn.elu(conv2d(x_value, W_conv1_value, padding="SAME") + b_conv1_value, name=prefixize("h_conv1_value"))

W_conv2_value = weight_variable([3,3,20,20], suffix="W_conv2_value")
b_conv2_value = bias_variable([20], suffix="b_conv2_value")
h_conv2_value = tf.nn.elu(conv2d(h_conv1_value, W_conv2_value, padding="SAME") + b_conv2_value, name=prefixize("h_conv2_value"))

W_conv3_value = weight_variable([3,3,20,20], suffix="W_conv3_value")
b_conv3_value = bias_variable([20], suffix="b_conv3_value")
h_conv3_value = tf.nn.elu(conv2d(h_conv2_value, W_conv3_value, padding="SAME") + b_conv3_value, name=prefixize("h_conv3_value"))

W_conv4_value = weight_variable([3,3,20,1], suffix="W_conv4_value")
b_conv4_value = bias_variable([BOARD_SIZE,BOARD_SIZE,1], suffix="b_conv4_value")
h_conv4_value = tf.nn.elu(conv2d(h_conv3_value, W_conv4_value, padding="SAME" + b_conv4_value, name=prefixize('h_conv4_value')))

# Reshape into a flattened board. That's so small! I want another layer, but I'm not
# sure I can handle it or need it. For now, NO THANKS.
h_conv4_flattened_value = tf.reshape([-1, BOARD_SIZE*BOARD_SIZE])
W_fc1_value = weight_variable([25,1], suffix="W_fc1_value")
b_fc1_value = bias_variable([1], suffix="b_fc1_value")
output_value = tf.tanh(tf.matmul(h_conv4_flattened_value, W_fc1_value) + b_fc1_value)

mean_square_value = tf.reduce_mean(tf.squared_difference(output_value, target_value), name=prefixize('mean_square_value'))

MomentumOptimizer_value = tf.train.MomentumOptimizer(0.01, 0.9, name=prefixize('momentum_optimize_value'))
train_step_value = MomentumOptimizer_value.minimize(mean_square_value)




l2_loss_layer_1_value = tf.nn.l2_loss(W_conv1_value, name=prefixize('l2_layer1_value'))
l2_loss_layer_2_value = tf.nn.l2_loss(W_conv2_value, name=prefixize('l2_layer2_value'))
l2_loss_layer_3_value = tf.nn.l2_loss(W_conv3_value, name=prefixize('l2_layer3_value'))
l2_loss_layer_4_value = tf.nn.l2_loss(W_conv4_value, name=prefixize('l2_layer4_value'))
l2_loss_layer_5_value = tf.nn.l2_loss(W_fc1_value, name=prefixize('l2_layer5_value'))

l2_error_total_value = REGULARIZER_CONSTANT * (l2_loss_layer_1_value + l2_loss_layer_2_value + 
                    l2_loss_layer_3_value + l2_loss_layer_4_value + l2_loss_layer_5_value)


GradientOptimizer_l2_value = tf.train.GradientDescentOptimizer(0.0005, name=prefixize('l2_optimizer_value'))
train_step_l2_reg_value = GradientOptimizer_l2_value.minimize(l2_error_total_value)


"""
The rest of everything
"""



all_variables = tf.all_variables()
if TRAIN_OR_TEST == "TRAIN":
  relavent_variables = all_variables
elif TRAIN_OR_TEST == "TEST":
  relavent_variables = [v for v in all_variables if (type(v.name) is unicode) and (v.name.startswith(NAME_PREFIX))]
else:
  raise Exception("TRAIN_OR_TEST must be TRAIN or TEST. Duh.")



saver = tf.train.Saver(var_list=relavent_variables, max_to_keep=MAX_TO_KEEP,
   keep_checkpoint_every_n_hours = KEEP_CHECKPOINT_EVERY_N_HOURS,
   name=prefixize("saver"))





"""
Here's how it is. If it's white's turn, what I was thinking was:
you simulate a move, and then take the move that's the highest chance of black losing.
BUT the problem with that is that now it's black's turn. That's a very different situation
than it being white's turn. So, I think what I should instead do is:
when it's white's turn, invert the board. Then, try to win with white. That's the spot
you would want to go. I think that's a much better idea.


Is this a good idea? I really don't know how to do these optimizers at the same time.

But, if they have internal variables, then the values are determined by the session,
not the graph. The graph is just a series of computations. So, I think I'm good.

The only time I might not be good is when I'm initializing from the first time. No,
I should be good then too.

"""


class Convbot_FIVE_POLICY_VALUE_NEWEST(GoBot):
  def __init__(self, folder_name=None, batch_num=0):
    GoBot.__init__(self)
    self.board_shape = (BOARD_SIZE,BOARD_SIZE)
    self.sess = tf.Session()
    # saver = tf.train.Saver()
    # sess = tf.Session()
    if folder_name is None or batch_num == 0:
      init = tf.initialize_variables(relavent_variables, name=prefixize("init"))
      self.sess.run(init)
      print("Initialized randomly")
      self.folder_name = folder_name
      self.batch_num = 0
    else:
      self.folder_name = folder_name
      self.batch_num = batch_num
      load_path = make_path_from_folder_and_batch_num(folder_name, batch_num)
      # print("initializing from path: " + str(load_path))
      saver.restore(self.sess, load_path)



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






  def save_to_path(self, save_path=None):
    if save_path is None:
      raise Exception("Must save to specified path")
    full_save_path = saver.save(self.sess, save_path)
    print("Model saved to path: " + full_save_path)



  def get_best_move(self, board_matrix, all_previous_boards, current_turn):

    if (board_matrix is None) or not (current_turn in (-1,1)):
      raise Exception("Invalid inputs to get_best_move.")

    valid_sensible_boardmap = util.output_valid_sensible_moves_boardmap(board_matrix, all_previous_boards, current_turn)
    zero_board = np.zeros((BOARD_SIZE,BOARD_SIZE))
    if zero_board.shape != valid_sensible_boardmap.shape:
      raise Exception('Sam, rethink something, because these shapes should be equal.')
    if np.array_equal(valid_sensible_boardmap, np.zeros((BOARD_SIZE,BOARD_SIZE))):
      print("This means that there are no valid moves left. Returning None.")
      return None

    valid_sensible_boardmap = np.asarray([valid_sensible_boardmap], dtype=np.float32)


    # valid_moves_mask = util.output_valid_moves_mask(board_matrix, all_previous_boards, current_turn)
    # print(valid_moves_mask)
    # print()
    # valid_moves_mask = valid_moves_mask.reshape([1, BOARD_SIZE*BOARD_SIZE +1])
    # valid_sensible_boardmap = valid_sensible_boardmap.reshape([1, BOARD_SIZE, BOARD_SIZE])


    # print(valid_moves_mask)

    board_input = board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)
    board_input = np.asarray([board_input], dtype=np.float32)

    legal_move_output_probs, output_policy = self.sess.run([normalized_legal_softmax_output_policy, softmax_output_policy], feed_dict={
      x_policy : board_input,
      legal_sensible_moves_map_policy : valid_sensible_boardmap
    })

    legal_move_output_probs = legal_move_output_probs[0]
    print(legal_move_output_probs)
    print(output_policy)
    
    best_move_index = np.argmax(legal_move_output_probs)
    best_move_tuple = (best_move_index // BOARD_SIZE, best_move_index % BOARD_SIZE)
    return best_move_tuple
    # Now for some funky business, because its an array.

    # return from_index_to_move_tuple(best_move_index)



  def from_board_to_on_policy_move(self, board_matrix, all_previous_boards, current_turn):
    """
    This is the thing I describe below.
    As always, we want to always be looking at the board from BLACK's
    perspective. Right?
    """
    if (board_matrix is None) or not (current_turn in (-1,1)):
      raise Exception("Invalid inputs to from_board_to_on_policy_move.")

    # valid_moves_mask = util.output_valid_moves_mask(board_matrix, all_previous_boards, current_turn)
    # valid_moves_mask = valid_moves_mask.reshape([1, BOARD_SIZE*BOARD_SIZE+1])    

    valid_sensible_moves_mask = util.output_valid_sensible_moves_boardmap(board_matrix, all_previous_boards, current_turn)
    zero_board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    
    if np.array_equal(valid_sensible_moves_mask, zero_board):
      return None
    # print(valid_moves_mask)

    valid_sensible_moves_mask = np.asarray([valid_sensible_moves_mask], dtype=np.float32)

    board_input = board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)
    board_input = np.asarray([board_input], dtype=np.float32)
    # print(board_input)

    # 
    
    # toPrint = [softmax_output_policy, legal_softmax_output_policy, sum_of_legal_probs_policy, normalized_legal_softmax_output_policy, h_fc1_policy, h_conv5_flat_policy]
    # # toPrint = [h_conv1_policy, h_conv2_policy, h_conv3_policy, h_conv4_policy]
    # print('printing along the way: ')
    # print(
    #   self.sess.run(toPrint, feed_dict={
    #     x_policy : board_input,
    #     softmax_temperature_policy : GLOBAL_TEMPERATURE,
    #     legal_sensible_moves_map_policy : valid_moves_mask
    #   })
    # )
    # print('printed')

    legal_move_output_probs = self.sess.run(normalized_legal_softmax_output_policy, feed_dict={
      x_policy : board_input,
      # softmax_temperature_policy : GLOBAL_TEMPERATURE,
      legal_sensible_moves_map_policy : valid_sensible_moves_mask
    })



    legal_move_output_probs = legal_move_output_probs[0]

    # print('legal_move_output_probs.')
    # print(legal_move_output_probs)

    random_number = random.random()
    prob_sum = 0.0
    desired_tuple = None

    should_break = False

    for i in xrange(BOARD_SIZE):
      for j in xrange(BOARD_SIZE):
        prob_sum += legal_move_output_probs[i][j]
        if prob_sum >= random_number:
          desired_tuple = (i,j)
          should_break = True
        if should_break:
          break
      if should_break:
        break

    if desired_tuple is None:
      print("didn't find a match for tuple. no problem though. It's just None.")

    return desired_tuple

    # for i in xrange(len(legal_move_output_probs)):
    #   prob_sum += legal_move_output_probs[i]
    #   if prob_sum >= random_number:
    #     desired_index = i
    #     break

    # if desired_index == -1:
    #   print(legal_move_output_probs)
    #   print(prob_sum)
    #   print("for some reason, prob_sum did not catch random_number")
    #   desired_index = BOARD_SIZE*BOARD_SIZE #Just put it at none, because that will
    #   # always be legal.
    #   # raise Exception("for some reason, prob_sum did not catch random_number")

    # tup = from_index_to_move_tuple(desired_index)
    # return tup


  def mask_and_renormalize_outputs(outputs, valid_sensible_board_mask):

    pass


  def create_inputs_to_learn_from_for_results_of_game(self, all_boards_list, all_moves_list, this_player, player_who_won):
    """
    This is a complicated one. First of all, I should
    find a way to do it in batch so I don't have to shoot myself in the head.

    But maybe I can try and do it not-batched to start. Yeah, I'll do that.
    
    So what's the gameplan? From a board, I get all the legal moves. I also get
    the move that actually happened. I also get the input that the person saw.

    I make a mask by starting everything as a zero, and making it "1" for the
    move that you did, plus all ILLEGAL moves.

    I make a goal by starting everything as a zero. If you won, you make the spot you
    went into a 1. If you lost, then everything remains a zero.

    Then, you input the things into the model, and update! That's really it.

    I've gotta do the whole game at once, there's no way around it.

    The only problem with the 'new' way of making the desired output is,
    I need to have the output probability for each step of the game.

    """
    
    """
    SO, NEW PLAN:
    Go through, calculate all outputs that the guy made (when it was his turn).
    Then, if it was a loss:
      set all of the spots to zero that are illegal, and set to zero the spot
      that made you lose. Then, renormalize. That's your goal.
    If it was a win, you just set everything but the move you made to 0.0, 
    and that to 1.0. That's easy, and you don't even have to calculate the output probs.


    HERE, what do I have to do? I need to change it so that the goal doesn't take into
    account the legal moves, until the end.

    """


    if len(all_boards_list) != len(all_moves_list):
      raise Exception("Should pass in one board per move.")

    current_turn = 1
    all_inputs = []
    players_moves = []
    legal_movemap_list = []

    for i in xrange(len(all_boards_list)):
      if current_turn != this_player:
        current_turn *= -1
        continue

      # Something here about passing on "move is None".

      board = all_boards_list[i]
      move = all_moves_list[i]

      if move is None:
        print("None, should not learn from this decision. The man had no choice.")
        continue

      all_previous_boards_from_this_turn = all_boards_list[0:i] #It shouldn't include the current board.

      board_input = board_to_input_transform_policy(
            board, all_previous_boards_from_this_turn, current_turn)
      

      players_moves.append(move)
      all_inputs.append(board_input)

      valid_sensible_boardmap = util.output_valid_sensible_moves_boardmap(board, 
                  all_previous_boards_from_this_turn, current_turn)
      legal_movemap_list.append(valid_sensible_boardmap)

    all_inputs = np.asarray(all_inputs, dtype=np.float32)
    # print(all_inputs)
    # .reshape((-1,BOARD_SIZE*BOARD_SIZE,NUM_FEATURES))
    
    legal_movemap_list = np.asarray(legal_movemap_list, dtype=np.float32)

    if this_player != player_who_won:
      if len(all_inputs) != len(legal_movemap_list):
        raise Exception('should be same number of moves!')


    all_output_goals = []


    for move in players_moves:
      if move is None:
        raise Exception("move should not be none. You can't learn from that, it should be caught.")
      output_goal = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
      output_goal[move[0]][move[1]] = 1.0
      # output_goal = np.zeros(BOARD_SIZE*BOARD_SIZE+1, dtype=np.float32)
      # index = from_move_tuple_to_index(move)
      # output_goal[index] = 1.0
      all_output_goals.append(output_goal)

    all_output_goals = np.asarray(all_output_goals, dtype=np.float32)
    print('desired outputs created')
    # It's odd, re-normalizing could make it so that some things are greater than 1.
    # It's actually not THAT unlikely. Anyway, it is what it is. Maybe I can go through
    # and MAX_CAP things. That's a good idea, it un-normalizes, but that's a better option.

    return all_inputs, all_output_goals, None




    # only_correct_moves = self.sess.run(softmax_output_policy, feed_dict={
    #   x_policy : all_inputs
    #   # softmax_temperature_policy : GLOBAL_TEMPERATURE
    # })
    # # Note: will be edited for move-legality.

    # for valid_output, legal_sensible_move in zip(only_correct_moves, legal_movemap_list):
    #   if np.count_nonzero(legal_sensible_move) == 0:
    #     print("no legal moves left, there's nothing much you can do. ")
    #     continue
    #   copy_valid = np.copy(valid_output)
    #   copy_valid *= legal_sensible_move
    #   if np.count_nonzero(copy_valid) == 0:
    #     print("Valid output is zero for some reason after the mult.")
    #     continue
    #   legal_sum = np.sum(copy_valid)
    #   if legal_sum < 0.01:
    #     print("For some reason, the sum is very small. Only small ones left?")
    #     print(legal_sum)
    #     print(copy_valid)
    #     continue
    #   valid_output /= legal_sum
    #   valid_output = np.clip(valid_output, 0.0, 1.0, out=valid_output)

    #   continue
    #   # valid_output *= legal_sensible_move
    #   # legal_sum = np.sum(valid_output)
    #   # print(legal_sum)
    #   # if legal_sum < 0.01:
    #   #   print("for some reason, only the very small ones are left. You dont want to mess with that")
    #   #   print("legal sum is 0!")
    #   #   continue
    #   #   print(valid_output)
    #   #   print(legal_sensible_move)
    #   #   raise Exception("boom bam")
    #   # valid_output /= legal_sum
    #   # valid_output = np.clip(valid_output, 0.0, 1.0, out=valid_output)

    # # return all_inputs, all_output_goals, only_correct_moves
    # return all_inputs, all_output_goals, only_correct_moves

    # # raise Exception("should return before here.")





    # if this_player == player_who_won:
    #   print('won!')
    #   for move in players_moves:
    #     if move is None:
    #       raise Exception("move should not be none. You can't learn from that, it should be caught.")
    #     output_goal = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    #     output_goal[move[0]][move[1]] = 1.0
    #     # output_goal = np.zeros(BOARD_SIZE*BOARD_SIZE+1, dtype=np.float32)
    #     # index = from_move_tuple_to_index(move)
    #     # output_goal[index] = 1.0
    #     all_output_goals.append(output_goal)
    #   all_output_goals = np.asarray(all_output_goals, dtype=np.float32)
    # else:
    #   print('lost!')
    #   print('need to compute legal moves here again...')
    #   # all_outputs_from_inputs = self.sess.run(softmax_output_policy, feed_dict={
    #   #   x_policy : all_inputs,
    #   #   # softmax_temperature_policy : GLOBAL_TEMPERATURE
    #   # })

    #   if all_outputs_from_inputs.shape != legal_movemap_list.shape:
    #     raise Exception('shapes should be same for moves and outputs.')

    #   for output, move in zip(valid_outputs, players_moves):
    #     old_val = output[move[0]][move[1]]
    #     output[move[0]][move[1]] = 0.0
    #     for_norm = np.sum(output)
    #     if abs(for_norm) <= 0.001:
    #       print('caught a zero-bug. Reverting.')
    #       output[index] = old_val
    #     else:
    #       output /= for_norm
    #   all_output_goals = np.asarray(valid_outputs, dtype=np.float32)

    # print('desired outputs created')
    # # It's odd, re-normalizing could make it so that some things are greater than 1.
    # # It's actually not THAT unlikely. Anyway, it is what it is. Maybe I can go through
    # # and MAX_CAP things. That's a good idea, it un-normalizes, but that's a better option.
    # for valid_output, legal_sensible_move in zip(valid_move_outputs, legal_movemap_list):
    #   valid_output *= legal_sensible_move
    #   legal_sum = np.sum(valid_output)
    #   valid_output /= legal_sum
    #   valid_output = np.clip(valid_output, 0.0, 1.0, out=valid_output)

    # return all_inputs, all_output_goals, valid_move_outputs

      
    #   # valid_outputs = all_outputs_from_inputs * legal_movemap_list
      
    #   # for output, move in zip(all_outputs_from_inputs, legal_movemap_list):

    #   # I think it will edit in place if I loop through.
    #   for output, move in zip(valid_outputs, players_moves):
    #     index = from_move_tuple_to_index(move)
    #     old_val = output[index]
    #     output[index] = 0.0 #we don't want to do this move!
    #     # But, there's a chance that there are no legal moves besides the
    #     # one you did, in which case we'll want to catch it.
    #     for_norm = np.sum(output)

    #     if abs(for_norm) <= 0.001:
    #       print('caught a zero-bug. Reverting.')
    #       output[index] = old_val
    #     else:
    #       output /= for_norm
    #   all_output_goals = np.asarray(valid_outputs, dtype=np.float32)
    #     # At this point, it should be normalized, and what we want for output.
    #     # Boy that was tough.
    # print('desired outputs created')
    # # valid_output_goals = np.asarray(valid_outputs, dtype=np.float32).reshape((-1, BOARD_SIZE*BOARD_SIZE+1))

    # return all_inputs, all_output_goals



    # # all_output_goals = []

    # return
    # raise Exception("everything below here is old!")




  def learn_from_for_results_of_game(self, all_boards_list, all_moves_list, this_player, player_who_won, num_times_to_train=1):
    all_inputs, all_output_goals, valid_moves_output_goals = self.create_inputs_to_learn_from_for_results_of_game(all_boards_list, all_moves_list, this_player, player_who_won)

    if this_player == player_who_won:
      print("Won!")
      for i in xrange(num_times_to_train):
        mse, who_cares = self.sess.run([mean_square_policy, train_step_winning_policy], feed_dict={
          x_policy : all_inputs,
          softmax_output_goal_policy : all_output_goals
        })
        print("MSE=" + str(mse))
    else:
      print("Lost!")
      for i in xrange(num_times_to_train):
        mse, who_cares = self.sess.run([mean_square_policy, train_step_losing_policy], feed_dict={
          x_policy : all_inputs,
          softmax_output_goal_policy : all_output_goals
        })
        print("MSE=" + str(mse))
    print("training L2_REG")
    self.sess.run([train_step_l2_reg_policy], feed_dict={
      # x_policy: all_inputs,
      # softmax_output_goal_policy: valid_moves_output_goals
    })
    print("properly trained!")
    return


    # print("training on game:")
    # print('all inputs shape:')
    # print(all_inputs.shape)
    # print('all training masks:')
    # print(all_training_masks)
    # print('all_training_masks shape:')
    # print(all_training_masks.shape)
    # print('all_output_goals shape')
    # print(all_output_goals.shape)
    # softmax_output_policy_result = self.sess.run(softmax_output_policy, feed_dict={
    #   x_policy : all_inputs,
    #   softmax_temperature_policy : GLOBAL_TEMPERATURE,
    #   training_mask_policy : all_training_masks,
    #   softmax_output_goal_policy : all_output_goals
    # })
    # print('softmax_output_policy shape: ')
    # print(softmax_output_policy_result.shape)

    """
    I really wish I knew more about the whether just minimizing the negative
    of the error is what we want to do. Hmmm. I think it really is. I really can't
    see a reason it wouldn't be. And it's nice because it's literally the opposite.

    # """

    # for i in xrange(num_times_to_train):
    #   self.sess.run(train_step_policy, feed_dict={
    #     x_policy : all_inputs,
    #     # softmax_temperature_policy : GLOBAL_TEMPERATURE,
    #     softmax_output_goal_policy : all_output_goals
    #   })
    # print("trained!")
    # print("error on this one: ")
    # print(
    #   self.sess.run(total_error, feed_dict={
    #     x_policy : all_inputs,
    #     # softmax_temperature_policy : GLOBAL_TEMPERATURE,
    #     softmax_output_goal_policy : all_output_goals
    #   })
    # )

def split_liberties(liberty_map, current_turn):
  """
  It's interesting. I'm going to separate by your liberties
  and their liberties.
  I don't really like this whole thing of passing in THIS MANY features.
  It seems like it should be able to learn this stuff on its own.

  They don't have all that stuff about their liberties because they have
  the feature that tells them whether they'd be captured or not, or something.

  But it would be nice to have a -2, -1, 1, 2 maybe? Am I optimizing too early?
  No, that's fine. I'll settle for six.

  """

  shape = liberty_map.shape

  where_me_one = np.zeros(shape, dtype=np.float32)
  where_me_two = np.zeros(shape, dtype=np.float32)
  where_me_gt_two = np.zeros(shape, dtype=np.float32)

  where_they_one = np.zeros(shape, dtype=np.float32)
  where_they_two = np.zeros(shape, dtype=np.float32)
  where_they_gt_two = np.zeros(shape, dtype=np.float32)

  where_zero = np.zeros(shape, dtype=np.float32) #Not returning this one, who cares.

  val_map = {
    -3  : where_they_gt_two,
    -2  : where_they_two,
    -1  : where_they_one,
     0  : where_zero,
     1  : where_me_one,
     2 : where_me_two,
     3 : where_me_gt_two
  }

  for i in xrange(shape[0]):
    for j in xrange(shape[1]):
      val = liberty_map[i][j]
      clipped = max(min(val,3),-3)
      clipped = clipped*current_turn
      if clipped not in val_map:
        print("clipped")
        print(clipped)
        print("How could it not be in these?")
        raise Exception('clipped should have been in this')
      val_map[clipped] = 1.0

  return where_me_gt_two, where_me_two, where_me_one,\
          where_they_one, where_they_two, where_they_gt_two


def split_board(board_matrix, current_turn):

  # I guess I can actually do it with yours vs theirs pretty easily.
  # All you have to do is multiply it by current_turn. Because if you're
  # negative one and they're one, that works out.

  # all_player = np.ones(board_matrix.shape).fill(current_turn)
  # all_negative_ones = -1 * np.ones(board_matrix.shape).fill(-1*current_turn)
  # pass
  shape = board_matrix.shape
  where_this_player = np.zeros(shape, dtype=np.float32)
  where_other_player = np.zeros(shape, dtype=np.float32)
  where_blank = np.zeros(shape, dtype=np.float32)

  val_map = {
    1   : where_this_player,
    0   : where_blank,
    -1  : where_other_player
  }

  for i in xrange(shape[0]):
    for j in xrange(shape[1]):
      val = board_matrix[i][j]
      val = val * current_turn #This should make yours positive.
      if val not in val_map:
        print(val)
        print('how is that not in val map?')
        raise Exception('should only have -1,0,1 on board')
      val_map[val] = 1.0

  return where_this_player, where_blank, where_other_player


  # where_one = np.zeros(shape, dtype=np.float32)
  # where_negative = np.zeros(shape, dtype=np.float32)
  # where_zero = np.zeros(shape, dtype=np.float32)

  # val_map = {
  #   -1 : where_negative,
  #   0 : where_zero,
  #   1: where_one
  # }

  # for i in xrange(shape[0]):
  #   for j in xrange(shape[1]):
  #     val = board_matrix[i][j]
  #     if val not in val_map:
  #       print(val)
  #       print('how is that not in val map?')
  #       raise Exception('should only have -1,0,1 on board')
  #     val_map[val] = 1.0
  # return where_one, where_zero, where_negative




  
def board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn):
  """
  I should get this ready for features, but I really don't want to.
  Remember, this is assuming that it's white's turn? No. For policy,
  it's black's turn. For value, it's white's turn. I think that means 
  I should have two of these functions.
  How can I still not be sure if I have a *-1 error somewhere? That's
  ridiculous.

  ASSUMES THIS PLAYER IS BLACK. IMPORTANT FOR LEGAL_MOVE MAP.

  NOPE, IT DOESN'T. BUT IT TRANSFORMS IT SO THAT THE OUTPUT THINKS IT IS.

  Why in the world is this a class method when it doesn't use anything from the
  class? I'm going to make it not-class.

  """

  # legal_moves_map = util.output_valid_moves_boardmap(board_matrix, all_previous_boards, current_turn)


  # You know, I really don't like this whole reshaping nonsense. I'm going to skip it I think.
  legal_sensible_move_map = util.output_valid_sensible_moves_boardmap(board_matrix, all_previous_boards, current_turn)
  liberty_map = util.output_liberty_map(board_matrix)


  # This is about to become a DOOZY
  board_this_player, board_blank, board_other_player = split_board(board_matrix, current_turn)
  where_me_gt_two, where_me_two, where_me_one,\
      where_they_one, where_they_two, where_they_gt_two =\
                        split_liberties(liberty_map, current_turn)

  ones_layer = np.ones_like(board_matrix)
  zeros_layer = np.zeros_like(board_matrix)

  feature_array = np.asarray([
    board_this_player,
    board_blank, 
    board_other_player,
    where_me_gt_two,
    where_me_two,
    where_me_one,
    where_they_one,
    where_they_two,
    where_they_gt_two,
    legal_sensible_move_map,
    ones_layer,
    zeros_layer
  ], dtype=np.float32)

  # feature_array = None
  # if current_turn == 1:
  #   feature_array = np.asarray([board_matrix, legal_sensible_move_map, liberty_map])
  # else:
  #   feature_array = np.asarray([-1 *board_matrix, legal_sensible_move_map, -1 * liberty_map])

  # SHIT. This isn't exactly right. This has shape (3, 5, 5) I need something that is 
  # of the form: (5,5,3). Looks like Transpose somehow does EXACTLY what I want.
  feature_array = feature_array.T

  flattened_array = feature_array.flatten()
  for i in range(len(flattened_array)):
    if flattened_array[i] < 0:
      print("Flattened array element less than zero!")
      # print(feature_array)
      print(flattened_array[i])
      raise Exception('no element can be less than zero!')
  # print('checks out')
  return feature_array

  # feature_array = feature_array.T
  # flattened_input = feature_array.reshape((1, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES))
  
  # return flattened_input


def board_to_input_transform_value(board_matrix, all_previous_boards, current_turn):
  """
  Same thing as above, right? Should I have one that says which color you are,
  actually? Yes, because of the handicap thing.

  I should get this ready for features, but I really don't want to.
  Remember, this is assuming that it's white's turn? No. For policy,
  it's black's turn. For value, it's white's turn. I think that means 
  I should have two of these functions.
  How can I still not be sure if I have a *-1 error somewhere? That's
  ridiculous.

  ASSUMES THIS PLAYER IS BLACK. IMPORTANT FOR LEGAL_MOVE MAP.

  NOPE, IT DOESN'T. BUT IT TRANSFORMS IT SO THAT THE OUTPUT THINKS IT IS.

  Why in the world is this a class method when it doesn't use anything from the
  class? I'm going to make it not-class.

  """

  # legal_moves_map = util.output_valid_moves_boardmap(board_matrix, all_previous_boards, current_turn)


  # You know, I really don't like this whole reshaping nonsense. I'm going to skip it I think.
  legal_sensible_move_map = util.output_valid_sensible_moves_boardmap(board_matrix, all_previous_boards, current_turn)
  liberty_map = util.output_liberty_map(board_matrix)


  # This is about to become a DOOZY
  board_this_player, board_blank, board_other_player = split_board(board_matrix, current_turn)
  where_me_gt_two, where_me_two, where_me_one,\
      where_they_one, where_they_two, where_they_gt_two =\
                        split_liberties(liberty_map, current_turn)

  ones_layer = np.ones_like(board_matrix)
  zeros_layer = np.zeros_like(board_matrix)

  feature_array = np.asarray([
    board_this_player,
    board_blank, 
    board_other_player,
    where_me_gt_two,
    where_me_two,
    where_me_one,
    where_they_one,
    where_they_two,
    where_they_gt_two,
    legal_sensible_move_map,
    ones_layer,
    zeros_layer
  ], dtype=np.float32)

  # feature_array = None
  # if current_turn == 1:
  #   feature_array = np.asarray([board_matrix, legal_sensible_move_map, liberty_map])
  # else:
  #   feature_array = np.asarray([-1 *board_matrix, legal_sensible_move_map, -1 * liberty_map])

  # SHIT. This isn't exactly right. This has shape (3, 5, 5) I need something that is 
  # of the form: (5,5,3). Looks like Transpose somehow does EXACTLY what I want.
  feature_array = feature_array.T

  flattened_array = feature_array.flatten()
  for i in range(len(flattened_array)):
    if flattened_array[i] < 0:
      print("Flattened array element less than zero!")
      # print(feature_array)
      print(flattened_array[i])
      raise Exception('no element can be less than zero!')
  # print('checks out')
  return feature_array

  # feature_array = feature_array.T
  # flattened_input = feature_array.reshape((1, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES))
  
  # return flattened_input









def make_path_from_folder_and_batch_num(folder_name, batch_num):
  save_path = os.path.join(this_dir + '/saved_models/convnet_pol_val_good', str(folder_name))
  file_name = 'trained_on_batch_' + str(batch_num) + '.ckpt'
  save_path = os.path.join(save_path, file_name)
  return save_path
  # save_path = './saved_models/convnet_feat_pol/trained_on_' + str(1) + '_batch.ckpt'

def play_game(load_data_1, load_data_2):
  """
  This is pretty involved. First, I'm only going to train the first one.
  I think that its a good idea to keep one stationary, so you're able
  to actually make a little bit of progress. I'm not sure though.

  We're going to simulate a game between the two of them, storing
  all of the boards and all of the moves that occur. 
  Then, we'll update the GO board using the function I wrote for that purpose.

  An update I think will REALLY help: By far the most time-consuming part of this
  whole thing is playing the game. So, after I get the results of the game,
  I'll train may times on them.

  """

  folder_1, batch_num_1 = load_data_1 
  folder_2, batch_num_2 = load_data_2
  # load_path_1 os.path.join('./saved_models/convnet_pol_val_good',str(folder_1),)
  # load_path_1 = make_path_from_folder_and_batch_num(folder_1, batch_num_1)
  # load_path_2 = make_path_from_folder_and_batch_num(folder_2, batch_num_2)

  convbot_one = Convbot_FIVE_SPLIT_FEATURES(folder_name=folder_1, batch_num=batch_num_1)
  convbot_two = Convbot_FIVE_SPLIT_FEATURES(folder_name=folder_2, batch_num=batch_num_2)

  "assign randomly who is who"
  p1 = (int(random.random() > 0.5) * 2) - 1 #That should generate 0/1, transform to 0/2, shift to -1,1
  p2 = -1 * p1
  print("training player playing as " + str(p1))

  player_dict = {
    p1 : convbot_one,
    p2 : convbot_two
  }

  all_moves = []
  all_previous_boards = []
  current_board = np.zeros((BOARD_SIZE, BOARD_SIZE))
  current_turn = 1
  while True:
    if (len(all_moves) >= 2) and all_moves[-1] is None and all_moves[-2] is None:
      print("Game is over!")
      break
    bot_up = player_dict[current_turn]
    # print(current_board)
    # print(all_previous_boards)
    # print(current_turn)
    this_move = bot_up.from_board_to_on_policy_move(current_board, all_previous_boards, current_turn)
    new_board = util.update_board_from_move(current_board, this_move, current_turn)

    all_moves.append(this_move)
    all_previous_boards.append(current_board)
    current_board = new_board
    current_turn *= -1
  print("Game lasted for " + str(len(all_moves)) + " turns.")
  winner = util.determine_winner(current_board)
  if winner == p1:
    print("p1 won!")
  else:
    print("p2 won!")

  convbot_one.learn_from_for_results_of_game(all_previous_boards, all_moves, p1, winner, num_times_to_train=1)
  # print("convbot_one has been taught")
  convbot_one.save_in_next_slot()
  print("convbot_one has been saved")
  convbot_one.sess.close()
  convbot_two.sess.close()



















def create_random_starters_for_folder(folder_name):
  print("creating random starter in folder: " + str(folder_name))
  new_cb = Convbot_FIVE_SPLIT_FEATURES(folder_name=folder_name, batch_num=0)
  new_cb.save_in_next_slot()
  print("random starter created and saved")
  set_largest_batch_in_folder(folder_name, 1)
  new_cb.sess.close()

def create_random_starters_for_folders(folder_name_list):
  for fn in folder_name_list:
    full_folder = os.path.join(this_dir, 'saved_models', 'convnet_pol_val_good', fn)
    print(full_folder)
    try:
      os.makedirs(full_folder)
    except Exception:
      print("looks like folder with name: " + str(fn) + " already exists")
  for fn in folder_name_list:
    create_random_starters_for_folder(fn)
  print('all created')





def get_largest_batch_in_folder(f_name):
  folder = os.path.join(this_dir, 'saved_models', 'convnet_pol_val_good')
  filename = os.path.join(folder, f_name,'largest.txt')
  f = open(filename, 'r')
  content = f.read()
  content = content.strip()
  latest = int(content)
  f.close()
  return latest


def set_largest_batch_in_folder(f_name, batch_num):
  folder = os.path.join(this_dir, 'saved_models', 'convnet_pol_val_good')
  filename = os.path.join(folder, f_name,'largest.txt')
  f = open(filename, 'w')
  f.write(str(batch_num))
  f.close()





def continuously_train(folders=['1','2','3']):
  global saver
  games_per_folder = 15
  # folders = ['1','2','3','4','5']
  # folders = ['1','2','3']
  while True:
    for f_name in folders:
      print('re-initializing saver')
      saver = tf.train.Saver(#var_list=relavent_variables, 
        max_to_keep=MAX_TO_KEEP,
        keep_checkpoint_every_n_hours = KEEP_CHECKPOINT_EVERY_N_HOURS,
        name=prefixize("saver"))
      print("training folder: " + f_name)
      largest_batch = get_largest_batch_in_folder(f_name)
      other_folders = set([f for f in folders if f != f_name])
      count = 0
      for other_f in other_folders:
        print("folder " + f_name + " playing folder " + other_f)
        largest_other_batch = get_largest_batch_in_folder(other_f)
        for game in xrange(games_per_folder):
          b1 = largest_batch + count
          play_game((f_name, b1), (other_f, largest_other_batch))
          count += 1
      set_largest_batch_in_folder(f_name, (largest_batch + count))
      print("Moving on to the next folder.")



def random_board_iterator():
  with open('./random_boards.txt') as f:
    while True:
      to_yield = [f.readline() for i in xrange(100)]
      if to_yield[-1] == '':
        print('done with iteration, hit end of file.')
        # print(to_yield)
        break
      to_yield = [json.loads(x.strip()) for x in to_yield]
      yield to_yield
  print("exit iterator")


def random_board_results_iterator():
  with open('./random_boards.txt') as f:
    while True:
      to_yield = [f.readline() for i in xrange(100)]
      if to_yield[-1] == '':
        print('done with iteration, hit end of file.')
        # print(to_yield)
        break
      to_yield = [json.loads(x.strip()) for x in to_yield]
      yield to_yield
  print("exit iterator")


def random_board_results_input_output_iterator():
  for results_obj in random_board_results_iterator():
    to_yield = []
    for result in results_obj:
      board = result['board']
      # input_black = 
      board_matrix = np.asarray(board, dtype=np.float32)
      # input_black = board_to
      # target_black_next
    continue






# def make_mirror_images(list_of_boards):
#   # Makes an assumption that it's Nx9x9
#   to_return = []
#   for board in list_of_boards:
#     inverted = [[-1.0 * val for val in row] for row in board]
#     to_return.append(inverted)
#   return to_return

def get_inputs_from_boards(boards):
  to_return = []
  for board in boards:
    board_matrix = np.asarray(board, dtype=np.float32)
    board_input_black = board_to_input_transform_policy(board_matrix, [], 1)
    board_input_white = board_to_input_transform_policy(board_matrix, [], -1)
    to_return.append(board_input_black)
    to_return.append(board_input_white)
  # There's this weird thing because I make the inputs wrapped, so I need
  # to take out at part.
  # to_return = [elem[0] for elem in to_return]
  to_return = np.asarray(to_return, dtype=np.float32)
  return to_return

def get_output_goals_for_boards(boards):
  # should be 0 for illegal moves, and 1/len(valid_moves) for legal moves.
  # I can set it to 0 and 1 everywhere, and then normalize.
  to_return = []
  for board in boards:
    board_matrix = np.asarray(board, dtype=np.float32)
    valid_move_goal_black = util.output_valid_sensible_moves_boardmap(board_matrix, [], 1)
    valid_move_goal_white = util.output_valid_sensible_moves_boardmap(board_matrix, [], -1)
    # black_sum = np.sum(valid_move_goal_black, keep_dims=True)
    # white_sum = np.sum(valid_move_goal_white, keep_dims=True)
    # zeros = np.zeros(valid_move_goal_black.shape)
    # if np.array_equal(valid_move_goal_black, zeros):
    #   pass
    # if np.array_equal(valid_move_goal_black, np.zeros(valid_move_goal_black.shape)):
    #   pass

    valid_black_sum = valid_move_goal_black.sum()
    if valid_black_sum < 0.01:
      valid_black_sum = 1.0
    valid_white_sum = valid_move_goal_white.sum()
    if valid_white_sum < 0.01:
      valid_white_sum = 1.0

    # if valid_black_sum < 0.001;
    #   to_return.
    # if valid_black_sum = 
    normalized_black_goal = valid_move_goal_black / valid_black_sum
    normalized_white_goal = valid_move_goal_white / valid_white_sum
    # print("normalized moves, to make sure its not all zeros or something crazy.")
    # print(normalized_black_goal)
    to_return.append(normalized_black_goal)
    to_return.append(normalized_white_goal)
  return np.asarray(to_return, dtype=np.float32)


# def train_on_all_random_boards(f_name):
#   largest = get_largest_batch_in_folder(f_name)
#   convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)
#   error_list = []
#   l2_error_list = []
#   for board_list in random_board_iterator():
#     inputs = get_inputs_from_boards(board_list)
#     outputs = get_output_goals_for_boards(board_list)
#     # training_mask = np.ones_like(outputs)
#     # The training mask is unneccesarry here, but what it does is say, everything
#     # is in play.
#     print("starting training calculation")
#     l2_err, tot_err, who_cares = convbot.sess.run([l2_error_total, total_error, train_step_winning_policy], feed_dict={
#       x_policy : inputs,
#       softmax_output_goal_policy : outputs
#     })
#     print("ending training calculation")
#     error_list.append(tot_err)
#     l2_error_list.append(l2_err)
#     print("done with " + str(len(error_list)) + " batches")
#     # print(error_list)
#     # print(l2_error_list)
#     if len(error_list) % 10 == 0:
#       print('error:')
#       print(error_list)
#       same_fname, new_largest = convbot.save_in_next_slot()
#       set_largest_batch_in_folder(same_fname, new_largest)
#       convbot.sess.close()
#       largest = new_largest
#       convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)

  
#   print(error_list)
#   print(l2_error_list)


def train_on_each_batch_lots(f_name):
  """
  Obviously this isn't perfect, but the reason I do it is because most
  of the time spent training is calculating the inputs and outputs, in lieu of
  storing them, I do this.

  What I COULD do if I wanna be fancy is prioritized experience replay.
  """
  largest = get_largest_batch_in_folder(f_name)
  convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)
  error_list = []
  l2_error_list = []
  num = 0
  for board_list in random_board_iterator():
    inputs = get_inputs_from_boards(board_list)
    outputs = get_output_goals_for_boards(board_list)
    if len(inputs) != len(outputs):
      raise Exception("lengths should be the same!")
    new_inputs = []
    new_outputs = []
    for ins, outs in zip(inputs, outputs):
      if np.array_equal(outs, np.zeros(outs.shape)):
        print('caught a zero matrix! Aha!')
        continue
      new_inputs.append(ins)
      new_outputs.append(outs)
    inputs = np.asarray(new_inputs, dtype=np.float32)
    outputs = np.asarray(new_outputs, dtype=np.float32)

    for i in range(50):
      num += 1
      # print("starting training calculation")
      err, who_cares = convbot.sess.run([mean_square_policy, train_step_winning_policy], feed_dict={
        x_policy : inputs,
        softmax_output_goal_policy : outputs
      })

      l2_err, who_cares = convbot.sess.run([l2_error_total, train_step_l2_reg_policy], feed_dict={

      })
      # shouldn't need a dict?




      # print("ending training calculation")
      # error_list.append(tot_err)
      # l2_error_list.append(l2_err)
      
      # print(error_list)
      # print(l2_error_list)
      if num % 10 == 0:
        print("done with " + str(num) + " batches")
        error_list.append(err)
        l2_error_list.append(l2_err)
        print('error:')
        print(error_list)
        # print('l2_error: ')
        # print(l2_error_list)
        same_fname, new_largest = convbot.save_in_next_slot()
        set_largest_batch_in_folder(same_fname, new_largest)
        convbot.sess.close()
        largest = new_largest
        print("one that just printed is " + str(largest - 1))
        convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)
      if len(error_list) > 200:
        print('shortening error lists')
        error_list = error_list[0::2]
        l2_error_list = l2_error_list[0::2]

  print(error_list)



def train_on_random_board_results(f_name):
  largest = get_largest_batch_in_folder(f_name)
  convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)
  error_list = []
  l2_error_list = []
  num = 0
  for result_obj_list in random_board_iterator():

    inputs = get_inputs_from_boards(board_list)
    outputs = get_output_goals_for_boards(board_list)
    if len(inputs) != len(outputs):
      raise Exception("lengths should be the same!")
    new_inputs = []
    new_outputs = []
    for ins, outs in zip(inputs, outputs):
      if np.array_equal(outs, np.zeros(outs.shape)):
        print('caught a zero matrix! Aha!')
        continue
      new_inputs.append(ins)
      new_outputs.append(outs)
    inputs = np.asarray(new_inputs, dtype=np.float32)
    outputs = np.asarray(new_outputs, dtype=np.float32)

    for i in range(50):
      num += 1
      # print("starting training calculation")
      err, who_cares = convbot.sess.run([mean_square_policy, train_step_winning_policy], feed_dict={
        x_policy : inputs,
        softmax_output_goal_policy : outputs
      })

      l2_err, who_cares = convbot.sess.run([l2_error_total, train_step_l2_reg_policy], feed_dict={

      })
      if num % 10 == 0:
        print("done with " + str(num) + " batches")
        error_list.append(err)
        l2_error_list.append(l2_err)
        print('error:')
        print(error_list)
        same_fname, new_largest = convbot.save_in_next_slot()
        set_largest_batch_in_folder(same_fname, new_largest)
        convbot.sess.close()
        largest = new_largest
        print("one that just printed is " + str(largest - 1))
        convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=largest)
      if len(error_list) > 200:
        print('shortening error lists')
        error_list = error_list[0::2]
        l2_error_list = l2_error_list[0::2]

  print(error_list)

def test_move_accuracy(f_name, batch=None):
  if batch is None:
    batch = get_largest_batch_in_folder(f_name)
  # largest = get_largest_batch_in_folder(f_name)
  convbot = Convbot_FIVE_SPLIT_FEATURES(folder_name=f_name, batch_num=batch)

  for board_list in random_board_iterator():
    inputs = get_inputs_from_boards(board_list)
    print(inputs[80])
    outputs = get_output_goals_for_boards(board_list)
    # training_mask = np.ones_like(outputs)
    # The training mask is unneccesarry here, but what it does is say, everything
    # is in play.
    print("starting training calculation")
    softmax_output, l2_err, mse = convbot.sess.run([softmax_output_policy, l2_error_total, mean_square_policy], feed_dict={
      x_policy : inputs,
      softmax_output_goal_policy : outputs,
      # softmax_temperature_policy : GLOBAL_TEMPERATURE
    })
    print('mean square error: ')
    print(mse)
    # print('first 10 outputs')
    # print(outputs[0:10])
    # print('first 10 produced outputs')
    # print(softmax_output[0:10])
    outputs_zipped = zip(outputs[80:90],softmax_output[80:90])
    print('printing zipped:')
    print(outputs_zipped)
































if __name__ == '__main__':
  continuously_train()


  # folder_name = "test_lotsonone"
  # dirpath = os.path.join(this_dir + '/saved_models/convnet_pol_val_good', folder_name)
  # if not os.path.isdir(dirpath):
  #   os.makedirs(dirpath)
  #   create_random_starters_for_folder(folder_name)
  # # train_on_all_random_boards(folder_name)
  # train_on_each_batch_lots(folder_name)


  # test_move_accuracy("test_lotsonone", batch=601)

  # num = 0
  # for rbs in random_board_iterator():
  #   num += 1
  #   # print("rbs")
  #   # print(len(rbs))
  #   # print(len(rbs[0]))
  #   # print(len(rbs[0][0]))
  # print(num)
  

  # print(len(list(random_board_iterator())))

  # create_random_starters_for_folder("test")
  # create_random_starters_for_folders(['1','2','3'])
  # train_on_all_random_boards("test")

  # train_on_each_batch_lots("test")
  # test_move_accuracy("test", 553)

  
  # continuously_train()
  # print "nothing here for now. Stopped training because."





  # times = 100
  # lower_folder = 0
  # round_num = 0
  # while True:
  #   f1 = (lower_folder + 1) % 5
  #   f2 = (lower_folder + 2) % 5



  # for i in 
  # for i in range()
  # f1 = "1"
  # f2 = "1"
  # b1 = 704
  # b2 = 461
  # for i in range(500):
  #   print("playing game " + str(i))
  #   play_game((f1,b1),(f2,b2))
  #   b1 += 1
  
  # print("training!")
  # automate_testing(0)
  # # start = 0
  # # finish = 100
  # # for i in range(start, finish):
  # #   # train_and_save_from_n_move_board((i * 7) % 20, batch_num=i)
  # #   # train backwards. I like this, it goes smoothly, but doesn't stick
  # #   # on one color.
  # #   n = int(20 - ((i - start)*20 / (finish - start))) + (i % 2)
  # #   train_and_save_from_n_move_board(n, batch_num=i)

  # print("trained!")



# I've got a flippity dippity error!!!!!


# I need to train, decreasing the GLOBAL_TEMPERATURE, over and over.
# I could put this on an EC2 instance. And then I could write an automated testing
# script



  
