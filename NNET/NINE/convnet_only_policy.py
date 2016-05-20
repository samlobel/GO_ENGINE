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


this_dir = os.path.dirname(os.path.realpath(__file__))




TRAIN_OR_TEST = "TRAIN"
# TRAIN_OR_TEST = "TEST"


NAME_PREFIX='ninebot_only_policy_'

BOARD_SIZE = 9

NUM_FEATURES = 3


MAX_TO_KEEP = 2
KEEP_CHECKPOINT_EVERY_N_HOURS = 0.1

GLOBAL_TEMPERATURE = 1.5 #Since values range from -1 to 1,
            # passing this through softmax with temp "2" will give the
            # difference between best and worst as 2.7 times as likely.
            # That's a pretty damn high temperature. But, it's good,
            # because at the beginning that means it will lead to spots
            # that have a lot of good moves following them. Nice.
            # temp 1 gives 9.7 times likelihood.
            # temp 1.5 gives like 5 times.

NUM_POLICY_GAMES_TO_SIMULATE_PER_BOARD = 5


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
  if suffix is None or type(suffix) is not str:
    raise Exception("bad weight initialization")
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=prefixize(suffix))

def bias_variable(shape, suffix):
  if suffix is None or type(suffix) is not str:
    raise Exception("bad bias initialization")  
  initial = tf.constant(0.1, shape=shape)
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

softmax_output_goal_policy: This is the goal of what you want to acheive. It
really only matters at locations that the training_mask_policy is 1. You should
pass in all zeros if you're trying to dissuade from a losing move, and all 
zeros EXCEPT for the move-spot you're trying to persuade more of a winning move.
"""

# x_value = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES], name=prefixize('x_value'))

x_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES], name=prefixize('x_policy'))
softmax_temperature_policy = tf.placeholder(tf.float32, [], name=prefixize('softmax_temperature_policy'))

legal_moves_mask_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('legal_moves_mask_policy'))

training_mask_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('training_mask_policy'))
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

masked_softmax_output_policy = tf.mul(softmax_output_policy, training_mask_policy)
# 

mean_square_policy = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(masked_softmax_output_policy, softmax_output_goal_policy), reduction_indices=[1]), name=prefixize("mean_square_policy"))


# AdamOptimizer_policy = tf.train.AdamOptimizer(1e-4)
# _policy(learning_rate=0.01, momentum=0.9)


# MomentumOptimizer_policy = tf.train.MomentumOptimizer(0.01, 0.9)
# train_step_policy = MomentumOptimizer_policy.minimize(mean_square_policy)

GDOptimizer_policy = tf.train.GradientDescentOptimizer(0.01)
train_step_policy = GDOptimizer_policy.minimize(mean_square_policy)





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


class Convbot_NINE_PURE_POLICY(GoBot):

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

    board_input = self.board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)

    legal_move_output_probs = self.sess.run(normalized_legal_softmax_output_policy, feed_dict={
      x_policy : board_input,
      softmax_temperature_policy : GLOBAL_TEMPERATURE,
      legal_moves_mask_policy : valid_moves_mask
    })

    legal_move_output_probs = legal_move_output_probs[0]
    print(legal_move_output_probs)
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

    board_input = self.board_to_input_transform_policy(board_matrix, all_previous_boards, current_turn)
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

    """
    set_target_to = None
    if this_player == player_who_won:
      set_target_to = 1.0
    else:
      set_target_to = 0.0



    if len(all_boards_list) != len(all_moves_list):
      raise Exception("Should pass in one board per move.")
    current_turn = 1
    all_inputs = []
    all_training_masks = []
    all_output_goals = []
    for i in xrange(len(all_boards_list)):
      if current_turn != this_player:
        current_turn *= -1
        continue

      board = all_boards_list[i]
      move = all_moves_list[i]

      all_previous_boards_from_this_turn = all_boards_list[0:i] #It shouldn't include the current board.
      # On the first time, it should be empty, on the last time, it should be all but the last.

      valid_moves = util.output_all_valid_moves(board, 
            all_previous_boards_from_this_turn, current_turn)

      board_input = self.board_to_input_transform_policy(
            board, all_previous_boards_from_this_turn, current_turn)

      """
      Training mask: I want the invalid moves to show through, as well
      as the single move that we care about. I don't care what it does
      to the other valid moves, I have no info on them
      """ 

      training_mask_policy = np.ones(BOARD_SIZE*BOARD_SIZE+1, dtype=np.float32)
      for valid_move in valid_moves:
        index_of_valid_move = from_move_tuple_to_index(move)
        training_mask_policy[index_of_valid_move] = 0.0 #We don't care about valid moves.
      # And now for the move we care about. Set it to 1, because we care about it.
      index_of_move = from_move_tuple_to_index(move)
      training_mask_policy[index_of_move] = 1.0

      # And finally, the goal. All zeros if you lost, all but one zero if you won.
      output_goal = np.zeros(BOARD_SIZE*BOARD_SIZE+1, dtype=np.float32)
      output_goal[index_of_move] = set_target_to

      all_inputs.append(board_input)
      all_training_masks.append(training_mask_policy)
      all_output_goals.append(output_goal)

      # This is important!
      current_turn *= -1

    all_inputs = np.asarray(all_inputs, dtype=np.float32).reshape((-1,BOARD_SIZE*BOARD_SIZE,NUM_FEATURES))
    all_training_masks = np.asarray(all_training_masks, dtype=np.float32).reshape((-1, BOARD_SIZE*BOARD_SIZE+1))
    all_output_goals = np.asarray(all_output_goals, dtype=np.float32).reshape((-1, BOARD_SIZE*BOARD_SIZE+1))

    # print("Finally, created all of the inputs. Who knows if they are \
    #   right though. Will print in the beginning to make sure.")

    # print("inputs: ")
    # print(all_inputs)
    # print('training masks: ')
    # print(all_training_masks)
    # print('output_goals')
    # print(all_output_goals)
    # print("screw it, I'm going to learn from them here too.")

    return all_inputs, all_training_masks, all_output_goals


  def learn_from_for_results_of_game(self, all_boards_list, all_moves_list, this_player, player_who_won):
    all_inputs, all_training_masks, all_output_goals = self.create_inputs_to_learn_from_for_results_of_game(all_boards_list, all_moves_list, this_player, player_who_won)
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
    self.sess.run(train_step_policy, feed_dict={
      x_policy : all_inputs,
      softmax_temperature_policy : GLOBAL_TEMPERATURE,
      training_mask_policy : all_training_masks,
      softmax_output_goal_policy : all_output_goals
    })
    print("trained!")
    print("error on this one: ")
    print(
      self.sess.run(mean_square_policy, feed_dict={
        x_policy : all_inputs,
        softmax_temperature_policy : GLOBAL_TEMPERATURE,
        training_mask_policy : all_training_masks,
        softmax_output_goal_policy : all_output_goals
      })
    )



  


  def board_to_input_transform_value(self, board_matrix, all_previous_boards, current_turn):
    """
    I should get this ready for features, but I really don't want to.
    Remember, this is assuming that it's white's turn? No. For policy,
    it's black's turn. For value, it's white's turn. I think that means 
    I should have two of these functions.
    How can I still not be sure if I have a *-1 error somewhere? That's
    ridiculous.

    ASSUMES THIS PLAYER IS WHITE. IMPORTANT FOR LEGAL_MOVE MAP.

    ACTUALLY, DOESN'T ASSUME THIS. BUT IT DOES TRANSFORM IT SO THAT ITS TRUE.

    """
    if current_turn not in (-1,1):
      raise Exception("current turn must be -1 or 1. instead it is " + str(current_turn))
    legal_moves_map = util.output_valid_moves_boardmap(board_matrix, all_previous_boards, current_turn)
    liberty_map = util.output_liberty_map(board_matrix)

    feature_array = None
    if current_turn == -1:
      feature_array = np.asarray([board_matrix, legal_moves_map, liberty_map])
    else:
      feature_array = np.asarray([-1 *board_matrix, legal_moves_map, -1 * liberty_map])

    feature_array = feature_array.T
    flattened_input = feature_array.reshape((1, BOARD_SIZE*BOARD_SIZE, NUM_FEATURES))
    
    return flattened_input

  def board_to_input_transform_policy(self, board_matrix, all_previous_boards, current_turn):
    """
    I should get this ready for features, but I really don't want to.
    Remember, this is assuming that it's white's turn? No. For policy,
    it's black's turn. For value, it's white's turn. I think that means 
    I should have two of these functions.
    How can I still not be sure if I have a *-1 error somewhere? That's
    ridiculous.

    ASSUMES THIS PLAYER IS BLACK. IMPORTANT FOR LEGAL_MOVE MAP.

    NOPE, IT DOESN'T. BUT IT TRANSFORMS IT SO THAT THE OUTPUT THINKS IT IS.
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
  """

  folder_1, batch_num_1 = load_data_1 
  folder_2, batch_num_2 = load_data_2
  # load_path_1 os.path.join('./saved_models/only_policy_convnet',str(folder_1),)
  # load_path_1 = make_path_from_folder_and_batch_num(folder_1, batch_num_1)
  # load_path_2 = make_path_from_folder_and_batch_num(folder_2, batch_num_2)

  convbot_one = Convbot_NINE_PURE_POLICY(folder_name=folder_1, batch_num=batch_num_1)
  convbot_two = Convbot_NINE_PURE_POLICY(folder_name=folder_2, batch_num=batch_num_2)

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

  convbot_one.learn_from_for_results_of_game(all_previous_boards, all_moves, p1, winner)
  # print("convbot_one has been taught")
  convbot_one.save_in_next_slot()
  print("convbot_one has been saved")
  convbot_one.sess.close()
  convbot_two.sess.close()



















def create_random_starters_for_folder(folder_name):
  print("creating random starter in folder: " + str(folder_name))
  new_cb = Convbot_NINE_PURE_POLICY(folder_name=folder_name, batch_num=0)
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
  games_per_folder = 25
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






if __name__ == '__main__':
  # for i in range(1,6):
  # create_random_starters_for_folder("1")
  continuously_train()





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



  
