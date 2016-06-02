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


NAME_PREFIX='ninebot_policy_movetrained_'

BOARD_SIZE = 9

NUM_FEATURES = 3


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


FEATURE 1: RAW INPUT
FEATURE 2: VALID MOVES
FEATURE 3: LIBERTY MAP

"""


"""
Like a god damn idiot, I deleted that whole blurb on how I want to do this.
And I'm too tired to think straight. But gotta keep chugging until my head
hits the keyboard. The enlightened way.

"""



def prefixize(suffix):
  return NAME_PREFIX + suffix


def from_index_to_move_tuple(index):
  if index < 0:
    raise Exception("index must be in bounds of board")
  elif index > BOARD_SIZE*BOARD_SIZE:
    raise Exception("index must be in bounds of board")
  elif index == BOARD_SIZE*BOARD_SIZE:
    return None
  else:
    tup = (index // BOARD_SIZE, index % BOARD_SIZE)
    return tup

def from_move_tuple_to_index(tup):
  if tup is None:
    return BOARD_SIZE*BOARD_SIZE
  else:
    r,c = tup
    if (r >= BOARD_SIZE) or (c >= BOARD_SIZE):
      raise Exception("rows and columns must both be present on board")
    elif (r < 0) or (c < 0):
      raise Exception("rows and columns must both be present on board")
    else:
      index = BOARD_SIZE*r + c
      return index


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




"""
INPUTS/OUTPUTS: 
x : These are board inputs. The inputs are used for training and testing,
for both the policy network and the value network.
The y_ output is used for training just the value network
the computed_values_for_moves goes into a softmax to create the target
output for the policy network.

What each of these does is confusing.

x_policy: is the transformed board input. Right now it's three features long, and
soon it will be four, but its your standard input.

softmax_temperature_policy: is the temperature of the softmax layer.

legal_moves_mask_policy: we use this as a way to output a valid move. It is 1 wherever a
move is valid, and 0 wherever a move is invalid.

training_mask_policy : its sort of the opposite of legal_moves_mask_policy. It is
one wherever a move is ILLEGAL, and also 1 wherever you actually went.
The reason for this is, where this is 1, the optimizer will have access to the 
underlying things, and therefore will optimizer parameters. Where it is zero,
the optimizer can ignore. It's very nice that these things are so smart.
NOTE:::: I'm changing my thoughts on this, because it should already know the
rules, more or less. I should train it on everything, always, and then set the
softmax_output_goal_policy to be : if it WON, 1 at the move it did and 0 elsewhere.
if it lost, I think I should just smooth it out a little. Send it to the average
over all legal moves. It would be better if I could send it to the average over
all legal moves except the one it chose, but that's probably unneccesarry.
The only problem I see with this is, half the time, you're training it
to not recognize anything, and that's pretty shitty, cause it's already
REALLY good at that. Shouldn't matter in the long-run, but it won't be good in
the short-run. If I lost, I would just want to go in the opposite direction of
winning. Maybe, that means that I need to minimize the negative of the old error.
That would definitely do what I want it to. It does seem a little shitty though that
it's going to smooth everything out, because it will push error to illegal moves.
So, even though it seems like a great idea, I'm not a fan.

Maybe, what I should do is (if I lose), get the output, set the index to 0
that we want to minimize, set everything to zero that is illegal, renormalize,
and then make that the target. Nice, that pushes the probability from the move
you took to everywhere that is legal except it. It would be nice if I didn't
have to have two separate training procedures, but whatever whatever.



softmax_output_goal_policy: This is the goal of what you want to acheive. It
really only matters at locations that the training_mask_policy is 1. You should
pass in all zeros if you're trying to dissuade from a losing move, and all 
zeros EXCEPT for the move-spot you're trying to persuade more of a winning move.
"""

# x_value = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES], name=prefixize('x_value'))

x_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES], name=prefixize('x_policy'))
softmax_temperature_policy = tf.placeholder(tf.float32, [], name=prefixize('softmax_temperature_policy'))

legal_moves_mask_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('legal_moves_mask_policy'))

# training_mask_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('training_mask_policy'))
softmax_output_goal_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('softmax_output_goal_policy'))


# y_ = tf.placeholder(tf.float32, [None, 1], name=prefixize("y_"))

# computed_values_for_moves = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('computed_values_for_moves'))


"""
POLICY NETWORK VARIABLES
All variables used for the policy network, including outputs.
Note that the optimizers don't have names, for either.

These should all end in the word 'policy'.

The problem is you don't want too much compression, because then you lose out
on the ability to know what's happening where.

I'll do this one with padding=same


It's tricky that you can't compress the input too much, because you don't
just need one number any more, but 26 (or 82, etc.) numbers. So, maybe
I should have padding='SAME' for the parts of the policy network that are 
not the first layer. OR/ALSO, I can just have a more complex fully-connected
part.
"""


x_image_policy = tf.reshape(x_policy, (-1,BOARD_SIZE,BOARD_SIZE,NUM_FEATURES), name=prefixize("x_image_policy"))

W_conv1_policy = weight_variable([5,5,NUM_FEATURES,20], suffix="W_conv1_policy")
b_conv1_policy = bias_variable([20], suffix="b_conv1_policy")
h_conv1_policy = tf.nn.elu(conv2d(x_image_policy, W_conv1_policy, padding="SAME") + b_conv1_policy, name=prefixize("h_conv1_policy"))

W_conv2_policy = weight_variable([3,3,20,20], suffix="W_conv2_policy")
b_conv2_policy = bias_variable([20], suffix="b_conv2_policy")
h_conv2_policy = tf.nn.elu(conv2d(h_conv1_policy, W_conv2_policy, padding="SAME") + b_conv2_policy, name=prefixize("h_conv2_policy"))

W_conv3_policy = weight_variable([3,3,20,20], suffix="W_conv3_policy")
b_conv3_policy = bias_variable([20], suffix="b_conv3_policy")
h_conv3_policy = tf.nn.elu(conv2d(h_conv2_policy, W_conv3_policy, padding="SAME") + b_conv3_policy, name=prefixize("h_conv3_policy"))

W_conv4_policy = weight_variable([3,3,20,20], suffix="W_conv4_policy")
b_conv4_policy = bias_variable([20], suffix="b_conv4_policy")
h_conv4_policy = tf.nn.elu(conv2d(h_conv3_policy, W_conv4_policy, padding="SAME") + b_conv4_policy, name=prefixize("h_conv4_policy"))

W_conv5_policy = weight_variable([3,3,20,20], suffix="W_conv5_policy")
b_conv5_policy = bias_variable([20], suffix="b_conv5_policy")
h_conv5_policy = tf.nn.elu(conv2d(h_conv4_policy, W_conv5_policy, padding="SAME") + b_conv5_policy, name=prefixize("h_conv5_policy"))

h_conv5_flat_policy = tf.reshape(h_conv5_policy, [-1, BOARD_SIZE*BOARD_SIZE*20], name=prefixize("h_conv2_flat_policy"))

W_fc1_policy = weight_variable([BOARD_SIZE*BOARD_SIZE*20, BOARD_SIZE*BOARD_SIZE + 1], suffix="W_fc1_policy")
b_fc1_policy = bias_variable([BOARD_SIZE*BOARD_SIZE + 1], suffix="b_fc1_policy")
h_fc1_policy = tf.nn.elu(tf.matmul(h_conv5_flat_policy, W_fc1_policy) + b_fc1_policy, name="h_fc1_policy")


softmax_output_policy = softmax_with_temp(h_fc1_policy, softmax_temperature_policy, suffix="softmax_output_policy")

legal_softmax_output_policy = tf.mul(softmax_output_policy, legal_moves_mask_policy)
sum_of_legal_probs_policy = tf.reduce_sum(legal_softmax_output_policy, reduction_indices=[1], keep_dims=True)
normalized_legal_softmax_output_policy = legal_softmax_output_policy / sum_of_legal_probs_policy
# The normalized_legal_softmax_output_policy is the thing you want to
# return for get_best_move, or for choosing your next move in a game.

# masked_softmax_output_policy = tf.mul(softmax_output_policy, training_mask_policy)
# 

mean_square_policy = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(softmax_output_policy, softmax_output_goal_policy), reduction_indices=[1]), name=prefixize("mean_square_policy"))
l2_loss_layer_1 = tf.nn.l2_loss(W_conv1_policy, name=prefixize('l2_layer1'))
l2_loss_layer_2 = tf.nn.l2_loss(W_conv2_policy, name=prefixize('l2_layer2'))
l2_loss_layer_3 = tf.nn.l2_loss(W_conv3_policy, name=prefixize('l2_layer3'))
l2_loss_layer_4 = tf.nn.l2_loss(W_conv4_policy, name=prefixize('l2_layer4'))
l2_loss_layer_5 = tf.nn.l2_loss(W_conv5_policy, name=prefixize('l2_layer5'))


# tf.contrib.layers.l2_regularizer(REGULARIZER_CONSTANT)#, name=prefixize("r1_policy"))
# l2_loss_layer_2 = tf.contrib.layers.l2_regularizer(REGULARIZER_CONSTANT)#, name=prefixize("r1_policy"))
# l2_loss_layer_3 = tf.contrib.layers.l2_regularizer(REGULARIZER_CONSTANT)#, name=prefixize("r1_policy"))
# l2_loss_layer_4 = tf.contrib.layers.l2_regularizer(REGULARIZER_CONSTANT)#, name=prefixize("r1_policy"))
# l2_loss_layer_5 = tf.contrib.layers.l2_regularizer(REGULARIZER_CONSTANT)#, name=prefixize("r1_policy"))

l2_error_total = REGULARIZER_CONSTANT * (l2_loss_layer_1 + l2_loss_layer_2 + 
                l2_loss_layer_3 + l2_loss_layer_4 + l2_loss_layer_5)

total_error = mean_square_policy + l2_error_total


# AdamOptimizer_policy = tf.train.AdamOptimizer(1e-4)
# _policy(learning_rate=0.01, momentum=0.9)


MomentumOptimizer_policy = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_policy'))
train_step_policy = MomentumOptimizer_policy.minimize(total_error)

# AdamOptimizer = tf.train.AdamOptimizer(1e-4)
# train_step_policy = AdamOptimizer.minimize(total_error)

MomentumOptimizer_learnmoves = tf.train.MomentumOptimizer(0.01, 0.1, name=prefixize('momentum_learnmoves_policy'))
train_step_learnmoves = MomentumOptimizer_policy.minimize(total_error)





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


class Convbot_NINE_POLICY_MOVETRAINED(GoBot):

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

    valid_moves_mask = util.output_valid_moves_mask(board_matrix, all_previous_boards, current_turn)
    # print(valid_moves_mask)
    # print()
    valid_moves_mask = valid_moves_mask.reshape([1, BOARD_SIZE*BOARD_SIZE +1])    
    print(valid_moves_mask)

    board_input = board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)

    legal_move_output_probs, output_policy = self.sess.run([normalized_legal_softmax_output_policy, softmax_output_policy], feed_dict={
      x_policy : board_input,
      softmax_temperature_policy : GLOBAL_TEMPERATURE,
      legal_moves_mask_policy : valid_moves_mask
    })

    legal_move_output_probs = legal_move_output_probs[0]
    print(legal_move_output_probs)
    print(output_policy)
    best_move_index = np.argmax(legal_move_output_probs)
    return from_index_to_move_tuple(best_move_index)



  def from_board_to_on_policy_move(self, board_matrix, all_previous_boards, current_turn):
    """
    This is the thing I describe below.
    As always, we want to always be looking at the board from BLACK's
    perspective. Right?
    """
    if (board_matrix is None) or not (current_turn in (-1,1)):
      raise Exception("Invalid inputs to from_board_to_on_policy_move.")

    valid_moves_mask = util.output_valid_moves_mask(board_matrix, all_previous_boards, current_turn)
    valid_moves_mask = valid_moves_mask.reshape([1, BOARD_SIZE*BOARD_SIZE+1])    
    # print(valid_moves_mask)

    board_input = board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)
    # print(board_input)

    # 
    
    # toPrint = [softmax_output_policy, legal_softmax_output_policy, sum_of_legal_probs_policy, normalized_legal_softmax_output_policy, h_fc1_policy, h_conv5_flat_policy]
    # # toPrint = [h_conv1_policy, h_conv2_policy, h_conv3_policy, h_conv4_policy]
    # print('printing along the way: ')
    # print(
    #   self.sess.run(toPrint, feed_dict={
    #     x_policy : board_input,
    #     softmax_temperature_policy : GLOBAL_TEMPERATURE,
    #     legal_moves_mask_policy : valid_moves_mask
    #   })
    # )
    # print('printed')

    legal_move_output_probs = self.sess.run(normalized_legal_softmax_output_policy, feed_dict={
      x_policy : board_input,
      softmax_temperature_policy : GLOBAL_TEMPERATURE,
      legal_moves_mask_policy : valid_moves_mask
    })



    legal_move_output_probs = legal_move_output_probs[0]

    random_number = random.random()
    prob_sum = 0.0
    desired_index = -1

    for i in xrange(len(legal_move_output_probs)):
      prob_sum += legal_move_output_probs[i]
      if prob_sum >= random_number:
        desired_index = i
        break

    if desired_index == -1:
      print(legal_move_output_probs)
      print(prob_sum)
      print("for some reason, prob_sum did not catch random_number")
      desired_index = BOARD_SIZE*BOARD_SIZE #Just put it at none, because that will
      # always be legal.
      # raise Exception("for some reason, prob_sum did not catch random_number")

    tup = from_index_to_move_tuple(desired_index)
    return tup


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

    """


    if len(all_boards_list) != len(all_moves_list):
      raise Exception("Should pass in one board per move.")

    current_turn = 1
    all_inputs = []
    players_moves = []
    if this_player != player_who_won:
      legal_movemap_list = []

    for i in xrange(len(all_boards_list)):
      if current_turn != this_player:
        current_turn *= -1
        continue

      board = all_boards_list[i]
      move = all_moves_list[i]
      all_previous_boards_from_this_turn = all_boards_list[0:i] #It shouldn't include the current board.

      board_input = board_to_input_transform_policy(
            board, all_previous_boards_from_this_turn, current_turn)

      all_inputs.append(board_input)
      players_moves.append(move)
      if this_player != player_who_won:
        valid_movemask = util.output_valid_moves_mask(board, 
                    all_previous_boards_from_this_turn, current_turn)
        legal_movemap_list.append(valid_movemask)

    all_inputs = np.asarray(all_inputs, dtype=np.float32).reshape((-1,BOARD_SIZE*BOARD_SIZE,NUM_FEATURES))
    
    if this_player != player_who_won:
      legal_movemap_list = np.asarray(legal_movemap_list, dtype=np.float32)

    if len(all_inputs) != len(players_moves):
      raise Exception('should be same number of moves!')

    if this_player != player_who_won:
      if len(all_inputs) != len(legal_movemap_list):
        raise Exception('should be same number of moves!')



    all_output_goals = []

    if this_player == player_who_won:
      print('won!')
      for move in players_moves:
        output_goal = np.zeros(BOARD_SIZE*BOARD_SIZE+1, dtype=np.float32)
        index = from_move_tuple_to_index(move)
        output_goal[index] = 1.0
        all_output_goals.append(output_goal)
      all_output_goals = np.asarray(all_output_goals, dtype=np.float32)
    else:
      print('lost!')
      print('need to compute legal moves here again...')
      all_outputs_from_inputs = self.sess.run(softmax_output_policy, feed_dict={
        x_policy : all_inputs,
        softmax_temperature_policy : GLOBAL_TEMPERATURE
      })
      if all_outputs_from_inputs.shape != legal_movemap_list.shape:
        raise Exception('shapes should be same for moves and outputs.')
      valid_outputs = all_outputs_from_inputs * legal_movemap_list
      # for output, move in zip(all_outputs_from_inputs, legal_movemap_list):

      # I think it will edit in place if I loop through.
      for output, move in zip(valid_outputs, players_moves):
        index = from_move_tuple_to_index(move)
        old_val = output[index]
        output[index] = 0.0 #we don't want to do this move!
        # But, there's a chance that there are no legal moves besides the
        # one you did, in which case we'll want to catch it.
        for_norm = np.sum(output)

        if abs(for_norm) <= 0.001:
          print('caught a zero-bug. Reverting.')
          output[index] = old_val
        else:
          output /= for_norm
      all_output_goals = np.asarray(valid_outputs, dtype=np.float32)
        # At this point, it should be normalized, and what we want for output.
        # Boy that was tough.
    print('desired outputs created')
    # valid_output_goals = np.asarray(valid_outputs, dtype=np.float32).reshape((-1, BOARD_SIZE*BOARD_SIZE+1))

    return all_inputs, all_output_goals



    # all_output_goals = []

    return
    raise Exception("everything below here is old!")




  def learn_from_for_results_of_game(self, all_boards_list, all_moves_list, this_player, player_who_won, num_times_to_train=25):
    all_inputs, all_output_goals = self.create_inputs_to_learn_from_for_results_of_game(all_boards_list, all_moves_list, this_player, player_who_won)
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
    for i in xrange(num_times_to_train):
      self.sess.run(train_step_policy, feed_dict={
        x_policy : all_inputs,
        softmax_temperature_policy : GLOBAL_TEMPERATURE,
        softmax_output_goal_policy : all_output_goals
      })
    print("trained!")
    print("error on this one: ")
    print(
      self.sess.run(total_error, feed_dict={
        x_policy : all_inputs,
        softmax_temperature_policy : GLOBAL_TEMPERATURE,
        softmax_output_goal_policy : all_output_goals
      })
    )



  
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

  # current_turn = 1

  legal_moves_map = util.output_valid_moves_boardmap(board_matrix, all_previous_boards, current_turn)
  liberty_map = util.output_liberty_map(board_matrix)

  feature_array = None
  if current_turn == 1:
    feature_array = np.asarray([board_matrix, legal_moves_map, liberty_map])
  else:
    feature_array = np.asarray([-1 *board_matrix, legal_moves_map, -1 * liberty_map])

  feature_array = feature_array.T
  flattened_input = feature_array.reshape((1, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES))
  
  return flattened_input








def make_path_from_folder_and_batch_num(folder_name, batch_num):
  save_path = os.path.join(this_dir + '/saved_models/only_policy_convnet', str(folder_name))
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
  # load_path_1 os.path.join('./saved_models/only_policy_convnet',str(folder_1),)
  # load_path_1 = make_path_from_folder_and_batch_num(folder_1, batch_num_1)
  # load_path_2 = make_path_from_folder_and_batch_num(folder_2, batch_num_2)

  convbot_one = Convbot_NINE_POLICY_MOVETRAINED(folder_name=folder_1, batch_num=batch_num_1)
  convbot_two = Convbot_NINE_POLICY_MOVETRAINED(folder_name=folder_2, batch_num=batch_num_2)

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

  convbot_one.learn_from_for_results_of_game(all_previous_boards, all_moves, p1, winner, num_times_to_train=10)
  # print("convbot_one has been taught")
  convbot_one.save_in_next_slot()
  print("convbot_one has been saved")
  convbot_one.sess.close()
  convbot_two.sess.close()



















def create_random_starters_for_folder(folder_name):
  print("creating random starter in folder: " + str(folder_name))
  new_cb = Convbot_NINE_POLICY_MOVETRAINED(folder_name=folder_name, batch_num=0)
  new_cb.save_in_next_slot()
  print("random starter created and saved")
  set_largest_batch_in_folder(folder_name, 1)
  new_cb.sess.close()

def create_random_starters_for_folders(folder_name_list):
  for fn in folder_name_list:
    full_folder = os.path.join(this_dir, 'saved_models', 'only_policy_convnet', fn)
    print(full_folder)
    try:
      os.makedirs(full_folder)
    except Exception:
      print("looks like folder with name: " + str(fn) + " already exists")
  for fn in folder_name_list:
    create_random_starters_for_folder(fn)
  print('all created')





def get_largest_batch_in_folder(f_name):
  folder = os.path.join(this_dir, 'saved_models', 'only_policy_convnet')
  filename = os.path.join(folder, f_name,'largest.txt')
  f = open(filename, 'r')
  content = f.read()
  content = content.strip()
  latest = int(content)
  f.close()
  return latest


def set_largest_batch_in_folder(f_name, batch_num):
  folder = os.path.join(this_dir, 'saved_models', 'only_policy_convnet')
  filename = os.path.join(folder, f_name,'largest.txt')
  f = open(filename, 'w')
  f.write(str(batch_num))
  f.close()





def continuously_train():
  global saver
  games_per_folder = 15
  folders = ['1','2','3','4','5']
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
  to_return = [elem[0] for elem in to_return]
  to_return = np.asarray(to_return, dtype=np.float32)
  return to_return

def get_output_goals_for_boards(boards):
  # should be 0 for illegal moves, and 1/len(valid_moves) for legal moves.
  # I can set it to 0 and 1 everywhere, and then normalize.
  to_return = []
  for board in boards:
    board_matrix = np.asarray(board, dtype=np.float32)
    valid_move_goal_black = util.output_valid_moves_mask(board_matrix, [], 1)
    valid_move_goal_white = util.output_valid_moves_mask(board_matrix, [], -1)
    # black_sum = np.sum(valid_move_goal_black, keep_dims=True)
    # white_sum = np.sum(valid_move_goal_white, keep_dims=True)
    normalized_black_goal = valid_move_goal_black / valid_move_goal_black.sum()
    normalized_white_goal = valid_move_goal_white / valid_move_goal_white.sum()
    # print("normalized moves, to make sure its not all zeros or something crazy.")
    # print(normalized_black_goal)
    to_return.append(normalized_black_goal)
    to_return.append(normalized_white_goal)
  return np.asarray(to_return, dtype=np.float32)


# def train_on_all_random_boards(f_name):
#   largest = get_largest_batch_in_folder(f_name)
#   convbot = Convbot_NINE_POLICY_MOVETRAINED(folder_name=f_name, batch_num=largest)
#   error_list = []
#   l2_error_list = []
#   for board_list in random_board_iterator():
#     inputs = get_inputs_from_boards(board_list)
#     outputs = get_output_goals_for_boards(board_list)
#     training_mask = np.ones_like(outputs)
#     # The training mask is unneccesarry here, but what it does is say, everything
#     # is in play.
#     print("starting training calculation")
#     l2_err, tot_err, who_cares = convbot.sess.run([l2_error_total, total_error, train_step_policy], feed_dict={
#       x_policy : inputs,
#       softmax_output_goal_policy : outputs,
#       softmax_temperature_policy : GLOBAL_TEMPERATURE,
#       training_mask_policy : training_mask
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
#       convbot = Convbot_NINE_POLICY_MOVETRAINED(folder_name=f_name, batch_num=largest)

  
#   print(error_list)


def train_on_each_batch_lots(f_name):
  """
  Obviously this isn't perfect, but the reason I do it is because most
  of the time spent training is calculating the inputs and outputs, in lieu of
  storing them, I do this.

  What I COULD do if I wanna be fancy is prioritized experience replay.
  """
  largest = get_largest_batch_in_folder(f_name)
  convbot = Convbot_NINE_POLICY_MOVETRAINED(folder_name=f_name, batch_num=largest)
  error_list = []
  l2_error_list = []
  num = 0
  for board_list in random_board_iterator():
    inputs = get_inputs_from_boards(board_list)
    outputs = get_output_goals_for_boards(board_list)
    # training_mask = np.ones_like(outputs)
    # The training mask is unneccesarry here, but what it does is say, everything
    # is in play.
    for i in range(50):
      num += 1
      # print("starting training calculation")
      l2_err, tot_err, who_cares = convbot.sess.run([l2_error_total, total_error, train_step_policy], feed_dict={
        x_policy : inputs,
        softmax_output_goal_policy : outputs,
        softmax_temperature_policy : GLOBAL_TEMPERATURE,
      })
      # print("ending training calculation")
      # error_list.append(tot_err)
      # l2_error_list.append(l2_err)
      
      # print(error_list)
      # print(l2_error_list)
      if num % 10 == 0:
        print("done with " + str(num) + " batches")
        error_list.append(tot_err)
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
        convbot = Convbot_NINE_POLICY_MOVETRAINED(folder_name=f_name, batch_num=largest)
      if len(error_list) > 200:
        print('shortening error lists')
        error_list = error_list[0::2]
        l2_error_list = l2_error_list[0::2]

  print(error_list)


def test_move_accuracy(f_name, batch=None):
  if batch is None:
    batch = get_largest_batch_in_folder(f_name)
  # largest = get_largest_batch_in_folder(f_name)
  convbot = Convbot_NINE_POLICY_MOVETRAINED(folder_name=f_name, batch_num=batch)

  for board_list in random_board_iterator():
    inputs = get_inputs_from_boards(board_list)
    print(inputs[80])
    outputs = get_output_goals_for_boards(board_list)
    # training_mask = np.ones_like(outputs)
    # The training mask is unneccesarry here, but what it does is say, everything
    # is in play.
    print("starting training calculation")
    softmax_output, l2_err, tot_err = convbot.sess.run([softmax_output_policy, l2_error_total, total_error], feed_dict={
      x_policy : inputs,
      softmax_output_goal_policy : outputs,
      softmax_temperature_policy : GLOBAL_TEMPERATURE
    })
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
  # dirpath = os.path.join(this_dir + '/saved_models/only_policy_convnet', folder_name)
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
  # train_on_all_random_boards("test")

  
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



  
