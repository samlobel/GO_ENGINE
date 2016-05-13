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


"""
This is going to be a lot different than my other ones. First of all, it's going to
be convolutional the whole way through, which should make it MUCH better. Second 
of all, the whole thing about passing, I'm going to just code in. There's an easy
way to see if a spot is sensible, which is that you never want to make a group
of yours have less than two liberties. So, if you would make the move, and it ends up
with your group in a situation that only has one liberty, you shouldn't do that move.
Not such a hard thing to code in.

Third, there's NO VALUE FUNCTION! If I want one, I can code it in later, and teach
it based on the policy. But for now, I'm going to have an adversarial training
ground. Here's how it's going to work:

I'm going to instantiate a bunch of random networks using elu units. Let's say 5
of them. Then, I'm going to train each one of them against a random one of the other
guys. The way to do that is a little complicated. 

To decide what move to do, you pass in a board to the policy network. You get 
activations based on that, and for each activation, you get move-probabilities.
You mask the illegal moves, and then re-normalize the probabilities, and then
choose a move. You do this back and forth.

Finally, at the end of the game, you see who won. If you win, you strenghten the
moves you did throughout the game. If you lose, you weaken them. And the whole
time, you're decreasing the strength of illegal/insensible moves. The way you
do this is:

Let's say you won. That means you want to strengthen the probability you
made the move you did, but you don't want to directly optimize the other legal 
moves. You take the output of the network, 
output_probs = output_from_policy_network
masker = (ZERO for all LEGAL moves that you didn't do, ONE otherwise)
masked_output = output_probs * masker
GOAL = (zeros everywhere except for the good move (if it was a loss, then zeros EVERYWHERE))
Error = mean square difference.
Minimize error using momentum.

What's nice about that is, the masker makes it so it doesn't try and edit the masked
probabilities. But it'll drive down the unmasked things where the goal is zero, and
drive up the unmasked things where the goal is one.


ANOTHER FEATURE I SHOULD ADD IS SCORE! IF A SPOT IS BLACKS AFTER SCORING, OR WHITES,
OR WHATEVER. THATS IMPORTANT. 


"""



# TRAIN_OR_TEST = "TEST"
TRAIN_OR_TEST = "TRAIN"


NAME_PREFIX='ninebot_policy_'

BOARD_SIZE = 9

NUM_FEATURES = 3


MAX_TO_KEEP = 5
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
      index = 5*r + c
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

def last_row_bias(shape, suffix):
  if suffix is None or type(suffix) is not str:
    raise Exception("bad last-row bias initialization")  
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
  sum_per_layer = tf.reduce_sum(exponents, reduction_indices=[1])
  softmax = tf.div(exponents, sum_per_layer, name=prefixize(suffix))
  return softmax




"""
INPUTS/OUTPUTS: 
x : These are board inputs. The inputs are used for training and testing,
for both the policy network and the value network.
The y_ output is used for training just the value network
the computed_values_for_moves goes into a softmax to create the target
output for the policy network.
"""

x_value = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE], name=prefixize('x_value'))

x_policy = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE], name=prefixize('x_policy'))
softmax_temperature_policy = tf.placeholder(tf.float32, [], name=prefixize('softmax_temperature_policy'))

y_ = tf.placeholder(tf.float32, [None, 1], name=prefixize("y_"))

computed_values_for_moves = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE+1], name=prefixize('computed_values_for_moves'))


"""
VALUE NETWORK VARIABLES
All variables used for the value network, including outputs.
Note that the optimizers don't have names, for either.

These should all end in the word 'value'.
"""



W_conv1_value = weight_variable([3,3,1,20], suffix="W_conv1_value")
b_conv1_value = bias_variable([20], suffix="b_conv1_value")

x_image_value = tf.reshape(x_value, [-1,BOARD_SIZE,BOARD_SIZE,1], name=prefixize("x_image_value"))

h_conv1_value = tf.nn.relu(conv2d(x_image_value, W_conv1_value, padding="VALID") + b_conv1_value, name=prefixize("h_conv1_value"))

W_conv2_value = weight_variable([2,2,20,20], suffix="W_conv2_value")
b_conv2_value = bias_variable([20], suffix="b_conv2_value")
h_conv2_value = tf.nn.relu(conv2d(h_conv1_value, W_conv2_value, padding="VALID") + b_conv2_value, name=prefixize("h_conv2_value"))

W_fc1_value = tf.Variable(tf.random_uniform([2*2*20,1],-0.1,0.1), name=prefixize("W_fc1_value"))
b_fc1_value = last_row_bias([1], suffix="b_fc1_value")

h_conv2_flat_value = tf.reshape(h_conv2_value, [-1, 2*2*20], name=prefixize("h_conv2_flat_value"))
y_conv_value = tf.nn.tanh(tf.matmul(h_conv2_flat_value, W_fc1_value) + b_fc1_value, name=prefixize("y_conv_value"))

cross_entropy_value = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv_value), reduction_indices=[1]), name=prefixize("cross_entropy_value"))
# mean_square_value = tf.reduce_mean(tf.reduce_sum((y_ - y_conv_value)**2, reduction_indices=[1]), name=prefixize("mean_square_value"))
mean_square_value = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, y_conv_value), reduction_indices=[1]), name=prefixize("mean_square_value"))

error_metric_value = mean_square_value

AdamOptimizer_value = tf.train.AdamOptimizer(1e-4)
train_step_value = AdamOptimizer_value.minimize(error_metric_value)


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



W_conv1_policy = weight_variable([3,3,1,20], suffix="W_conv1_policy")
b_conv1_policy = bias_variable([20], suffix="b_conv1_policy")

x_image_policy = tf.reshape(x_policy, [-1,BOARD_SIZE,BOARD_SIZE,1], name=prefixize("x_image_policy"))

h_conv1_policy = tf.nn.relu(conv2d(x_image_policy, W_conv1_policy, padding="SAME") + b_conv1_policy, name=prefixize("h_conv1_policy"))

W_conv2_policy = weight_variable([3,3,20,20], suffix="W_conv2_policy")
b_conv2_policy = bias_variable([20], suffix="b_conv2_policy")
h_conv2_policy = tf.nn.relu(conv2d(h_conv1_policy, W_conv2_policy, padding="SAME") + b_conv2_policy, name=prefixize("h_conv2_policy"))



W_fc1_policy = tf.Variable(tf.random_uniform([5*5*20, 128],-0.1,0.1), name=prefixize("W_fc1_policy"))
b_fc1_policy = last_row_bias([128], suffix="b_fc1_policy")

h_conv2_flat_policy = tf.reshape(h_conv2_policy, [-1, 5*5*20], name=prefixize("h_conv2_flat_policy"))
h_fc1_policy = tf.nn.relu(tf.matmul(h_conv2_flat_policy, W_fc1_policy) + b_fc1_policy, name="h_fc1_policy")

W_fc2_policy = tf.Variable(tf.random_uniform([128, (BOARD_SIZE*BOARD_SIZE + 1)],-0.1,0.1), name=prefixize("W_fc2_policy"))
b_fc2_policy = last_row_bias([BOARD_SIZE*BOARD_SIZE + 1], suffix="b_fc1_policy")

softmax_input_policy = tf.matmul(h_fc1_policy,W_fc2_policy) + b_fc2_policy
softmax_output_policy = softmax_with_temp(softmax_input_policy, softmax_temperature_policy, suffix="softmax_output_policy")


softmax_of_target_policy = softmax_with_temp(computed_values_for_moves, softmax_temperature_policy, suffix="softmax_of_target_policy")



cross_entropy_policy = tf.reduce_mean(-tf.reduce_sum(softmax_of_target_policy * tf.log(softmax_output_policy), reduction_indices=[1]), name=prefixize("cross_entropy_policy"))

error_metric_policy = cross_entropy_policy

AdamOptimizer_policy = tf.train.AdamOptimizer(1e-4)
train_step_policy = AdamOptimizer_policy.minimize(error_metric_policy)



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

sess = tf.Session()



"""
Here's how it is. If it's white's turn, what I was thinking was:
you simulate a move, and then take the move that's the highest chance of black losing.
BUT the problem with that is that now it's black's turn. That's a very different situation
than it being white's turn. So, I think what I should instead do is:
when it's white's turn, invert the board. Then, try to win with white. That's the spot
you would want to go. I think that's a much better idea.

"""


class Convbot_FIVE_POLICY(GoBot):

  def __init__(self, load_path=None):
    GoBot.__init__(self)
    self.board_shape = (BOARD_SIZE,BOARD_SIZE)
    # saver = tf.train.Saver()
    # sess = tf.Session()
    self.load_path = load_path
    if load_path is None:
      init = tf.initialize_variables(relavent_variables, name=prefixize("init"))
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
    flattened = board_matrix.reshape((1,BOARD_SIZE*BOARD_SIZE))
    results = sess.run(y_conv_value, feed_dict={
        x_value : flattened
    })
    return results[0]

  def evaluate_boards(self, board_matrices):
    results = sess.run(y_conv_value, feed_dict={
        x_value : board_matrices
      })
    return results





  def get_best_move(self, board_matrix, previous_board, current_turn):
    """
    As I said before: If current_turn==1, then you simulate one move ahead,
    and output the move with the highest score.
    If current_turn==-1, then you flip the board, simulate one move ahead as
    if you are black, and output the move with the highest score (e.g. the

    Similar to the old one, but the flipping happens at a more easy-to-reason
    place.
    """
    if current_turn == -1:
      board_matrix = -1 * board_matrix
    if (current_turn == -1) and (previous_board is not None):
      previous_board = -1 * previous_board

    valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, 1))
    if len(valid_moves) is None:
      return None

    new_boards = [util.update_board_from_move(board_matrix, move, 1) for move in valid_moves]
    
    value_of_new_boards = [self.evaluate_board(b) for b in new_boards]
    value_move_pairs = zip(value_of_new_boards, valid_moves)
    best_value_move_pair = max(value_move_pairs)

    return best_value_move_pair[1]


  def from_board_to_on_policy_move(self, board_matrix, temperature, previous_board, current_turn):
    """
    This is the thing I describe below.
    As always, we want to always be looking at the board from BLACK's
    perspective. Right?
    """
    if (board_matrix is None) or (temperature is None) or not (current_turn in (-1,1)):
      raise Exception("Invalid inputs to from_board_to_on_policy_move.")
    
    if current_turn == -1:
      board_matrix = -1 * board_matrix
    if (current_turn == -1) and (previous_board is not None):
      previous_board = -1 * previous_board

    board_input = self.board_to_input_transform(board_matrix)
    
    output_probs = sess.run(softmax_output_policy, feed_dict={
      x_policy : board_input,
      softmax_temperature_policy : temperature
    })

    output_probs = output_probs[0]

    # print("output probs: ")
    # print(output_probs)

    legal_moves = np.zeros(BOARD_SIZE*BOARD_SIZE + 1, dtype=np.float32)
    for i in xrange(BOARD_SIZE*BOARD_SIZE+1):
      move = from_index_to_move_tuple(i) #This should include None.
      if util.move_is_valid(board_matrix, move, 1, previous_board):
        legal_moves[i] = 1.0
      else:
        continue
    # print(legal_moves)
    only_legal_probs = output_probs * legal_moves
    # print(only_legal_probs)
    normalized_legal_probs = only_legal_probs / np.sum(only_legal_probs)
    # print(normalized_legal_probs)
    random_number = random.random()
    prob_sum = 0.0
    desired_index = -1
    for i in xrange(len(normalized_legal_probs)):
      prob_sum += normalized_legal_probs[i]
      if prob_sum >= random_number:
        desired_index = i
        break

    if desired_index == -1:
      print(normalized_legal_probs)
      print(prob_sum)
      raise Exception("for some reason, prob_sum did not catch random_number")

    tup = from_index_to_move_tuple(desired_index)
    return tup
    

  def generate_board_from_n_on_policy_moves(self, n):
    """
    Returns: A board, and the person who should move next.
    Retuns None, None if the game reached its conclusion while simulating.

    The tricky thing here is always picking legal moves. I think that the
    legal moves should be an input feature to the policy. Maybe to the value
    network as well, I'm not sure. Why not I guess.
    So you use it as an input to the policy network.
    And then you get all of the probabilities from the temperature-softmax. 
    Then you multiply that by a mask of legal moves. And then you re-normalize
    by dividing by the new sum. And then, you generate a random number between
    0 and 1, and iterate through the array, keeping a sum, until you get a 
    probability larger than your random number. Then, you go from that index
    to a move, by doing (i mod BOARD_SIZE, i // BOARD_SIZE) (or the inverse), or
    if it's the last one, NONE.
    Then you make the move.
    But remember, the policy is ALWAYS asking about a move that is BLACK's turn.
    """
    
    current_turn = 1
    previous_board = np.zeros((BOARD_SIZE,BOARD_SIZE))
    current_board = np.zeros((BOARD_SIZE,BOARD_SIZE))

    moves = []
    for i in xrange(n):
      if (len(moves) >=2) and (moves[-1] is None) and (moves[-2] is None):
        # The board hit the end of the game!
        return None, None

      probabilistic_next_move = self.from_board_to_on_policy_move(current_board, GLOBAL_TEMPERATURE, previous_board, current_turn)

      moves.append(probabilistic_next_move)
      next_board = util.update_board_from_move(current_board, probabilistic_next_move, current_turn)
      previous_board = current_board
      current_board = next_board
      current_turn = -1 * current_turn


    return current_board, current_turn


  def board_to_input_transform(self, board_matrix):
    """
    I should get this ready for features, but I really don't want to.
    """
    flattened = board_matrix.reshape((1, BOARD_SIZE*BOARD_SIZE))
    return flattened


  def multiple_boards_to_input_transform(self, board_matrices):
    mapped = map(self.board_to_input_transform, board_matrices)
    return np.asarray(mapped)




  def get_results_of_board_on_policy(self, board_matrix, previous_board, current_turn, move_list=None):
    if move_list is None and util.boards_are_equal(board_matrix, previous_board):
      move_list = [None]
    if move_list is None:
      move_list=[]
    # print(len(move_list))
    move_list = list(move_list) #Before, it was passing by reference, which is terrible.
    # I guess why it was taking so short before was because it was just giving up on most of them,
    # because it was throwing them out.

    while not ((len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None)):
      # print ("move: " + str(len(move_list)))
      if len(move_list) > 100:
        print("simulation lasted more than 100 moves")
        return None
      # best_move = self.get_best_move(board_matrix, previous_board, current_turn)
      on_policy_move = self.from_board_to_on_policy_move(board_matrix, GLOBAL_TEMPERATURE, previous_board, current_turn)
      if on_policy_move is None:
        new_board = copy(board_matrix)
      else:
        new_board = util.update_board_from_move(board_matrix, on_policy_move, current_turn)
      move_list.append(on_policy_move)
      previous_board = board_matrix
      board_matrix = new_board
      current_turn *= -1

    return copy(board_matrix)

  def get_outcome_of_board_on_policy(self, board_matrix, previous_board, current_turn, move_list=None):
    """
    How am I going to use this? 
    I'm going to simulate out to some point, then I'm going to try every move, then
    I'm going to ask how it turns out a bunch of times, then I'm going to average those
    results together, then I'm going to update the value function, then I'm going to
    use the value function to update the policy function.

    So, if you ask me about a board, it's going to be assuming the current turn is BLACK.
    But after you take each move from BLACK, you'll be asking this when it's white's
    turn. So, you should just return the honest answer, because that's what it's looking for.
    Did White, or Black, win?

    HOW MANY SHOULD I AVERAGE TOGETHER?!?!?! I SHOULD DEFINE A CONSTANT.

    """
    # print(current_turn)
    result_of_board = self.get_results_of_board_on_policy(board_matrix, previous_board, current_turn, move_list)
    if result_of_board is None:
      return None
    winner_of_game = util.determine_winner(result_of_board, handicap=0.0)
    return winner_of_game ##Winner of game is -1 or 1, or MAYBE 0.

  def get_average_result_of_board_on_policy(self, board_matrix, previous_board, current_turn, move_list=None):

    """
    Uses the above function, calls it NUM_POLICY_GAMES_TO_SIMULATE_PER_BOARD 
    times, and spits out average.
    """

    results_array = [self.get_outcome_of_board_on_policy(board_matrix, previous_board, current_turn, move_list)
                      for i in xrange(NUM_POLICY_GAMES_TO_SIMULATE_PER_BOARD)]
    results_array = [r for r in results_array if r is not None]
    
    if len(results_array) == 0:
      return 0.0
    average_result = (sum(results_array) + 0.0) / len(results_array)
    # print('average_result:')
    # print(average_result)
    return average_result






  def gather_all_possible_results(self, board_input, previous_board, current_turn):
    print('gathering results')
    if board_input is None:
      print("passed None to gather_all_possible_results")
      return None
    if current_turn == -1:
      print(board_input)
      board_input = -1 * board_input
    if (current_turn == -1) and (previous_board is not None):
      previous_board = -1 * previous_board
    if current_turn == -1:
      print(board_input)

    valid_moves = list(util.output_all_valid_moves(board_input, previous_board, 1))
    value_board_move_list = []
    for move in valid_moves:
      resulting_board = util.update_board_from_move(board_input, move, 1)
      board_value = self.get_average_result_of_board_on_policy(resulting_board, board_input, -1, [move])
      # This should be using -1 because we're playing from the other guy now.
      value_board_move_list.append((board_value, resulting_board, move))

    return value_board_move_list

  def from_value_board_move_list_to_value_list(self, value_board_move_list):
    possible_moves_length = BOARD_SIZE*BOARD_SIZE+1
    goal_array = np.full((possible_moves_length, ), -10000.0, np.float32)

    for (value, board, move) in value_board_move_list:
      # input_board = board_to_input_transform(board)
      computed_board_value = self.evaluate_board(board)
      move_index = from_move_tuple_to_index(move)
      goal_array[move_index] = computed_board_value

    return goal_array

    




  def train_policy_and_value_from_input(self, board_input, current_turn):
    if current_turn == -1:
      board_input = -1 * board_input
      current_turn = 1

    value_board_move_list = self.gather_all_possible_results(board_input, None, 1)
    if value_board_move_list is None:
      print("Somehow we were passed a dead board.")
      return
    
    y_goal = np.asarray([value for (value, board, move) in value_board_move_list])
    boards = np.asarray([board for (value, board, move) in value_board_move_list])
    
    print("Result of on-policy simulation: ")
    print(y_goal)

    y_goal = y_goal.reshape((-1,1))
    boards = boards.reshape((-1, BOARD_SIZE*BOARD_SIZE))

    sess.run(train_step_value, feed_dict={
      x_value : boards,
      y_ : y_goal
    })
    
    print("updated value network from all resulting boards!")
    value_list = self.from_value_board_move_list_to_value_list(value_board_move_list)
    # value_list = np.asarray([value_list], dtype=np.float32)
    value_list = value_list.reshape((1,BOARD_SIZE*BOARD_SIZE+1))
    # print("created true value list. Its shape is :  " + str(value_list.shape) + " . Should be 1,26")
    print(value_list)


    board_input = self.board_to_input_transform(board_input)
    sess.run(train_step_policy, feed_dict={
      x_policy : board_input,
      computed_values_for_moves : value_list,
      softmax_temperature_policy : GLOBAL_TEMPERATURE 
    })

    print("updated policy network!")


  def train_policy_and_value_from_on_policy_board_after_n_steps(self, n):
    current_board, current_turn = self.generate_board_from_n_on_policy_moves(n)
    print("generated board.")
    self.train_policy_and_value_from_input(current_board, current_turn)




def train_and_save_from_n_move_board(n, batch_num=0):
  print("training on policy board after " + str(n) + " steps")
  load_path = None
  save_path = './saved_models/convnet_with_policy/trained_on_' + str(1) + '_batch.ckpt'
  if batch_num != 0:
    load_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num) + '_batch.ckpt'
    save_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num+1) + '_batch.ckpt'

  c_b = Convbot_FIVE_POLICY(load_path=load_path)

  print("training!")
  c_b.train_policy_and_value_from_on_policy_board_after_n_steps(n)
  c_b.save_to_path(save_path)
  print("saved!")











if __name__ == '__main__':
  print("training!")
  start = 240
  finish = 440
  for i in range(start, finish):
    # train_and_save_from_n_move_board((i * 7) % 20, batch_num=i)
    # train backwards
    n = int(20 - ((i - start)*20 / (finish - start)))
    train_and_save_from_n_move_board(n, batch_num=i)

  print("trained!")



# I've got a flippity dippity error!!!!!





