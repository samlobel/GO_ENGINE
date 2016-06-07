import sys
import os
import numpy as np
import json
import time
import random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))


this_dir = os.path.dirname(os.path.realpath(__file__))

from go_util import util

from multiprocessing import Process, Queue


BOARDS_FILE = os.path.join(this_dir, 'on_policy_results.txt')
BOARD_SHAPE = (5,5)


from NNET.FIVE.convnet_better_policy_value import \
  Convbot_FIVE_POLICY_VALUE_NEWEST, get_largest_convbot_from_folder, board_to_input_transform_value


FOLDER_NAME = 'test'


def return_value_object(convbot):
  """
  I'm going to end up with a WINNNER, and a random BOARD paired with the PLAYER
  who had to decide where to go on that board.

  From this, I should return all the moves, etc. and the winner.

  """
  current_turn = 1
  move_list = []
  all_previous_boards = []
  person_up_list = []
  current_board = np.zeros(BOARD_SHAPE, dtype=np.float32)
  while True:
    if (len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None):
      print('over')
      break
    # print('going')
    # print(move_list)
    this_move = convbot.from_board_to_on_policy_move(current_board, all_previous_boards, current_turn)
    new_board = util.update_board_from_move(current_board, this_move, current_turn)

    move_list.append(this_move)
    all_previous_boards.append(current_board)
    person_up_list.append(current_turn)

    current_board = new_board
    current_turn *= -1
  print("Game lasted for " + str(len(move_list)) + " turns.")
  if len(move_list) > 50:
    print('long game. why?')
    print(move_list)
  winner = util.determine_winner(current_board)
  length_of_game = len(all_previous_boards)
  random_index = random.randint(0, length_of_game - 1)
  board_at_turn = all_previous_boards[random_index]
  player_at_turn = person_up_list[random_index]
  all_previous_then = all_previous_boards[0:random_index]


  """
  Remember, the winner here is BLACK/WHITE. But what we care about is,
  did the person who moved last turn win. In other words, if 1 played on the board,
  and the winner was 1, that should be a loss. So, to get the proper result,
  we need to do winner * -1 * player_at_turn. If 1 won, and its blacks turn to go
  on the board we see, that means that it has a bad value for white. Therefore, it
  should be a bad move.
  """
  value_goal = winner * -1.0 * player_at_turn

  inputs = board_to_input_transform_value(board_at_turn, all_previous_then, player_at_turn)
  board_array = board_at_turn.tolist()
  inputs_array = inputs.tolist()

  return_obj = {
    'board_input_serialized' : inputs_array,
    'input_board' : board_array,
    'value_goal' : value_goal,
    'current_turn' : player_at_turn
  }
  return return_obj

def test():
  convbot, _ = get_largest_convbot_from_folder(FOLDER_NAME)
  print convbot.from_board_to_on_policy_move
  for i in range(10):
    return_obj = return_value_object(convbot)
    print return_obj

# def worker_loader(b_queue, num=100000):
#   for i in xrange(num):
#     b_queue.put(i)
#   print 'loaded all.'

def worker_transform(r_queue, num=100000):
  convbot, _ = get_largest_convbot_from_folder(FOLDER_NAME)
  for i in xrange(num):
    try:
      value_obj = return_value_object(convbot)
      if value_obj is None:
        print 'hmmmmm'
        continue
      obj_string = json.dumps(value_obj) + '\n'
      r_queue.put(obj_string)

    except Exception:
      print 'exception in writer!'
      break


def worker_writer(r_queue):
  i = 0
  time_now = time.time()
  with open(BOARDS_FILE, 'a') as f_out:
    while True:
      try:
        board_str = r_queue.get(block=True, timeout=10)
        f_out.write(board_str)
        i += 1
        if i % 1000 == 0:
          print "boards written: " + str(i)
        if i % 5000 == 0:
          print "time taken: " + str(time.time() - time_now)
      except Exception:
        print 'end of results queue!'
        break


def kick_off(num=200000):
  print 'starting'
  RESULT_QUEUE = Queue(maxsize=1000)  
  NUM_WORKERS = None
  try:
    NUM_WORKERS = int(sys.argv[1])
  except:
    NUM_WORKERS=4
  print 'num workers: ' + str(NUM_WORKERS)
  num_per = num // NUM_WORKERS
  proc_out = Process(target=worker_writer, args=[RESULT_QUEUE])
  proc_out.start()
  for i in range(NUM_WORKERS):
    proc_transform = Process(target=worker_transform, args=[RESULT_QUEUE, num_per])
    proc_transform.start()
    print "process kicked off: " + str(i)
  print "processes kicked off"




if __name__ == '__main__':
  # test()
  kick_off(100)












