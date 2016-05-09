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


# TRAIN_OR_TEST = "TRAIN"
TRAIN_OR_TEST = "TEST"


NAME_PREFIX='fivebot_policy_'

BOARD_SIZE = 5


MAX_TO_KEEP = 100
KEEP_CHECKPOINT_EVERY_N_HOURS = 0.0167



def prefixize(suffix):
  return NAME_PREFIX + suffix


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
  initial = tf.constant(1.0, shape=shape)
  return tf.Variable(initial, name=prefixize(suffix))


def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

# def conv2d_skip(x,W):
#   return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')

# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



x = tf.placeholder(tf.float32, [None, BOARD_SIZE*BOARD_SIZE], name=prefixize('x'))
y_ = tf.placeholder(tf.float32, [None,1], name=prefixize("y_"))


W_conv1 = weight_variable([3,3,1,10], suffix="W_conv1")
b_conv1 = bias_variable([10], suffix="b_conv1")

x_image = tf.reshape(x,[-1,BOARD_SIZE,BOARD_SIZE,1], name=prefixize("x_image"))

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name=prefixize("h_conv1"))


# W_conv2 = weight_variable([2,2,5,10])
# b_conv2 = bias_variable([10])

# h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)


W_fc1 = tf.Variable(tf.random_uniform([3*3*10,1],-0.1,0.1), name=prefixize("W_fc1"))
b_fc1 = last_row_bias([1], suffix="b_fc1")

h_conv1_flat = tf.reshape(h_conv1, [-1, 3*3*10], name=prefixize("h_conv1_flat"))
y_conv = tf.nn.tanh(tf.matmul(h_conv1_flat, W_fc1) + b_fc1, name=prefixize("y_conv"))

# W_fc3 = weight_variable([128, 1])
# b_fc3 = bias_variable([1])

# h_fc3

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name=prefixize("cross_entropy"))
mean_square = tf.reduce_mean(tf.reduce_sum((y_ - y_conv)**2, reduction_indices=[1]), name=prefixize("mean_square"))

error_metric = mean_square

# AdamOptimizer = tf.train.AdamOptimizer(1e-4, name=prefixize("Adam_Optimizer"))
# train_step = AdamOptimizer.minimize(error_metric, name=prefixize("train_step"))

AdamOptimizer = tf.train.AdamOptimizer(1e-4)
train_step = AdamOptimizer.minimize(error_metric)

# train_step = tf.train.AdamOptimizer(1e-4).minimize(error_metric, name=prefixize("train_step"))

all_variables = tf.all_variables()
if TRAIN_OR_TEST == "TRAIN":
  relavent_variables = all_variables
elif TRAIN_OR_TEST == "TEST":
  relavent_variables = [v for v in all_variables if (type(v.name) is unicode) and (v.name.startswith(NAME_PREFIX))]
else:
  raise Exception("TRAIN_OR_TEST must be TRAIN or TEST. Duh.")
  # relavent_variables = [v for v in all_variables if (type(v.name) is unicode) and (v.name.startswith(NAME_PREFIX))]

# AdamOptimizer_variable_names = AdamOptimizer.get_slot_names()
# print(AdamOptimizer_variable_names)
# AdamOptimizer_variables = [tf.train.AdamOptimizer.get_slot(AdamOptimizer, name) for name in AdamOptimizer_variable_names]
# print(AdamOptimizer_variables)
# relavent_variables.extend(AdamOptimizer_variables)
# print([v.name for v in all_variables])
# print([v.name for v in relavent_variables])
# # print([type(v.name) for v in all_variables])
# print(AdamOptimizer_variables)



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

    random_board, next_turn = util.generate_random_board((BOARD_SIZE,BOARD_SIZE), num_moves)
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

    flattened_board = simulator_input_board.reshape((1,BOARD_SIZE*BOARD_SIZE))
    simulated_board_conclusion = self.get_results_of_board(
      simulator_input_board, np.zeros((BOARD_SIZE,BOARD_SIZE)), -1
    )
    if simulated_board_conclusion is None:
      print("Lasted more than 300.")
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
    return simulator_input_board.reshape(BOARD_SIZE*BOARD_SIZE), [winner]





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

  # def OLD_get_best_move(self, board_matrix, previous_board, current_turn):
  #   # print("Simulating turn...")
  #   valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
    
  #   if len(valid_moves) == 0:
  #     return None

  #   new_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_moves]
  #   valid_moves.append(None)
  #   new_boards.append(copy(board_matrix))



  #   if current_turn == -1:
  #     new_boards = [(-1 * board) for board in new_boards]

  #   value_of_new_boards = [self.evaluate_board(b) for b in new_boards]

  #   value_move_pairs = zip(value_of_new_boards, valid_moves)
  #   # print(value_move_pairs)
  #   # value_move_pairs = sorted(value_move_pairs)
  #   best_value_move_pair = max(value_move_pairs)
  #   # print(best_value_move_pair)

  #   # print("\n\n\ngetting best move from convnet")

  #   return best_value_move_pair[1]




  def get_best_move(self, board_matrix, previous_board, current_turn):
    """
    As I said before: If current_turn==1, then you simulate one move ahead,
    and output the move with the highest score.
    If current_turn==-1, then you flip the board, simulate one move ahead as
    if you are black, and output the move with the highest score (e.g. the
    highest chance of FAKE BLACK, AKA WHITE, winning).

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
    
    valid_moves.append(None)
    new_boards.append(copy(board_matrix))

    value_of_new_boards = [self.evaluate_board(b) for b in new_boards]
    value_move_pairs = zip(value_of_new_boards, valid_moves)
    best_value_move_pair = max(value_move_pairs)

    return best_value_move_pair[1]



  def generate_board_from_n_on_policy_moves(self, n):
    
    previous_board = np.zeros((BOARD_SIZE,BOARD_SIZE)) 
    current_board = np.zeros((BOARD_SIZE,BOARD_SIZE))

    current_turn = 1
    moves = []
    for i in xrange(n):
      # if (len(moves) >=2) and 
      best_move = self.get_best_move(current_board, previous_board, current_turn)
      previous_board = current_board
      current_board = util.update_board_from_move(current_board, best_move, current_turn)



      continue

    return
    pass





  def get_result_of_board_from_random_policy(self, board_matrix, previous_board, current_turn, move_list=None):
    if move_list is None and util.boards_are_equal(board_matrix, previous_board):
      move_list = [None]
    if move_list is None:
      move_list=[]

    while not ((len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None)):
      # print ("move: " + str(len(move_list)))
      if len(move_list) > 300:
        print("simulation lasted more than 300 moves")
        return None
      # valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
      valid_move = util.output_one_valid_move(board_matrix, previous_board, current_turn)
      # best_move = self.get_best_move(board_matrix, previous_board, current_turn)
      if valid_move is None:
        new_board = copy(board_matrix)
      else:
        new_board = util.update_board_from_move(board_matrix, valid_move, current_turn)
      move_list.append(valid_move)
      previous_board = board_matrix
      board_matrix = new_board
      current_turn *= -1

    return copy(board_matrix)


  def get_results_of_board(self, board_matrix, previous_board, current_turn, move_list=None):
    if move_list is None and util.boards_are_equal(board_matrix, previous_board):
      move_list = [None]
    if move_list is None:
      move_list=[]

    while not ((len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None)):
      # print ("move: " + str(len(move_list)))
      if len(move_list) > 300:
        print("simulation lasted more than 300 moves")
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


  def generate_on_policy_after_n_moves(self, num_moves):
    pass


  def gather_all_possible_results(self, board_input, previous_board, current_turn):
    """
    Shit, is this wrong too? I should switch it beforehand, just to be safe.
    
    To judge a board, I ALWAYS need to make sure it's white's turn.
    So I make the initial one BLACK's turn, then simulate one move in the future,
    and then 

    BIG CHANGE! I'M GOING TO SWITCH TO DOING RANDOM SIMULATIONS AFTER
    I MAKE A DETERMINISTIC MOVE. THAT WAY I CAN BETTER EXPLORE THE STATE SPACE.


    """
    print("gathering results")

    if current_turn == -1:
      board_input = -1 * board_input
    if (current_turn == -1) and (previous_board is not None):
      previous_board = -1 * previous_board

    valid_moves = list(util.output_all_valid_moves(board_input, previous_board, 1))

    resulting_boards = [util.update_board_from_move(board_input, move, 1) for move in valid_moves]
    zipped = zip(valid_moves, resulting_boards)
    # end_results_of_boards = [self.get_results_of_board(new_board, board_input, -1, [move])
    #           for (move, new_board) in zipped] #this is -1, because it is next turn.

    


    end_results_of_boards = [self.get_result_of_board_from_random_policy(new_board, board_input, -1, [move])
              for (move, new_board) in zipped] #this is -1, because it is next turn.          
    


    
    print("all boards simulated")

    boards_before_and_after = zip(resulting_boards, end_results_of_boards)
    # Filter it to get rid of nones.
    boards_before_and_after = [(before, after) for (before, after) in boards_before_and_after
                                  if (before is not None) and (after is not None)]

    boards_before = [before for (before, after) in boards_before_and_after] #filtered
    boards_after = [after for (before, after) in boards_before_and_after] #filtered

    true_value_of_boards = [util.determine_winner(board) for board in boards_after]
    # returning true values with all moves one after input.
    return zip(true_value_of_boards, boards_before)


    # values_and_boards = [(util.determine_winner(after), before)]                              
    

    # true_value_of_boards = [util.determine_winner(board) for board in end_results_of_boards]
    # return zipped(true_value_of_boards, resulting_boards)





    return

    # valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
    # resulting_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_moves]
    # zipped = zip(valid_moves, resulting_boards)
    # end_results_of_boards = [self.get_results_of_board(new_board, board_input, (-1*current_turn), [move])
    #           for (move, new_board) in zipped]
    # true_value_of_boards = [util.determine_winner(board) for board in end_results_of_boards]
    # print("results gathered")
    # return zipped(true_value_of_boards, resulting_boards)


  def train_from_input_board(self, board_input, current_turn):
    """
    If the current_turn is 1, that means that these boards have white going first.
    That's good. Our evaluator tells us, if it's white's turn, what are the
    chances that black wins. 
    
    If the current_turn is -1, that means that they have black going first.
    In that case, since our evaluator tells us the chance of black winning if
    it is white's turn, that means we need to 


    The winner in all_results is the chance that BLACK wins. If it's white's turn
    that means we're evaluating from white's perspective. I think maybe I do
    have a negative wrong. Time to write it all out again.

    If it's white's turn, he wants to do the move that's most likely
    to have him win. If you flip the board, and also flip who it is,
    and do the analysis, and BLACK wins, that would mean that WHITE would
    win in the original game. So, to choose the best move, if it's white's
    turn, you flip the board, make it's black's turn, and figure out what's
    most likely to make black win.

    If current_turn == -1:
      flip the input board. Generate all possible moves for BLACK (1) going
      on the flipped board. Evaluate all of these boards. Choose the one that
      says its most likely that BLACK wins.

    To train, if it's white's turn:
    if current_turn == -1:
      flip the input board. Generate all possible moves (and corresponding
      positions) for BLACK (1) going on the flipped board. Then, take
      these games to their natural conclusion. If BLACK wins
      (determine_winner returns 1), that's a positive result 
      (meaning it would be a good move). You also do evaluation on
      all of these possible next-positions. A high result means that,
      if it's WHITE's turn now, that you think BLACK is going to win.
      So, after this, you optimize so that the generated value is closer 
      to the calculated one for ALL of the boards.

    """
    if current_turn == -1:
      board_input = -1* board_input

    all_results = self.gather_all_possible_results(board_input, None, 1) 
    # that's one because I flipped the board before.
    
    y_goal = np.asarray([[result[0]] for result in all_results])

    boards = np.asarray([result[1] for result in all_results])
    boards = boards.reshape((-1, BOARD_SIZE*BOARD_SIZE))

    print("about to train")
    for i in range(3): #just silly and arbitrary
      sess.run(train_step, feed_dict={
        x : boards,
        y_ : y_goal
      })
    print("trained")

  def train_from_empty_board(self):
    starting_board = np.zeros((BOARD_SIZE,BOARD_SIZE))
    starting_turn = 1
    print("about to train")
    self.train_from_input_board(starting_board, starting_turn)
    print("trained")


  def train_from_board_after_n_moves(self, num_moves):
    random_board, starting_turn = util.generate_random_board((BOARD_SIZE,BOARD_SIZE), num_moves)
    print("about to train on random board after " + str(num_moves) + "moves")
    self.train_from_input_board(random_board, starting_turn)
    print("trained")




  






    # all_results = self.gather_all_possible_results(board_input, current_turn)





    pass
















def train_convbot_on_randoms(batch_num=0):

  load_path = None
  save_path = './saved_models/convnet_with_policy/trained_on_' + str(1) + '_batch.ckpt'
  if batch_num != 0:
    load_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num) + '_batch.ckpt'
    save_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num+1) + '_batch.ckpt'

  b_c = Convbot_FIVE_POLICY(load_path=load_path)
  # for i in xrange(10):
  #   for num_moves in xrange(50, 75):
  #     b_c.train_on_random_board(num_moves)
  #   b_c.save_to_path(save_path="./saved_models/trained_on_" + str(i+1) + "_epochs.ckpt")
  num_array = range(0,75)
  b_c.train_from_num_moves_array(num_array)
  b_c.save_to_path(save_path)

  # print("Done with training it seems")


def train_and_save_from_empty_input(batch_num=0):
  load_path = None
  save_path = './saved_models/convnet_with_policy/trained_on_' + str(1) + '_batch.ckpt'
  if batch_num != 0:
    load_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num) + '_batch.ckpt'
    save_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num+1) + '_batch.ckpt'

  c_b = Convbot_FIVE_POLICY(load_path=load_path)
  c_b.train_from_empty_board()
  print("trained")
  c_b.save_to_path(save_path)
  print("saved!")


def train_and_save_from_n_board_random(n, batch_num=0):
  load_path = None
  save_path = './saved_models/convnet_with_policy/trained_on_' + str(1) + '_batch.ckpt'
  if batch_num != 0:
    load_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num) + '_batch.ckpt'
    save_path = './saved_models/convnet_with_policy/trained_on_' + str(batch_num+1) + '_batch.ckpt'

  c_b = Convbot_FIVE_POLICY(load_path=load_path)
  
  for i in range(10):
    print("training on sub-batch " + str(i))
    c_b.train_from_board_after_n_moves(n)
  print("trained")
  c_b.save_to_path(save_path)
  print("saved!")













if __name__ == '__main__':
  # b_c = Basic_ConvBot()
  print("training!")
  # train_convbot_on_randoms(load_path='./saved_models/more_convnet_with_policy/trained_on_2_batch.ckpt')
  
  # train_and_save_from_empty_input(batch_num=3)
  for i in range(0, 100):
    train_and_save_from_n_board_random((i * 7) % 20, batch_num=i)
  # for i in range(39, 250):
  #   train_convbot_on_randoms(batch_num=i)

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




