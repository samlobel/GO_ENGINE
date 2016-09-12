import json
import numpy as np

import sys
import os

# from Queue import Queue
import time
from multiprocessing import Process, Queue
import math
# from threading import Thread

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from go_util import util

from NNET.NEW_ATTEMPT import clean_convnet

# MIN_TURN=10
# MAX_TURN=19



# FILENAME_IN = os.path.join('.', 'training_data', 'random_boards_'+str(MIN_TURN)+'_to_'+str(MAX_TURN)+'.txt')
# FILENAME_OUT = os.path.join('.', 'training_data', 'random_board_results_'+str(MIN_TURN)+'_to_'+str(MAX_TURN)+'.txt')


def get_filename_in(min_turn, max_turn):
  FILENAME_IN = os.path.join('.', 'training_data', 'random_boards_'+str(min_turn)+'_to_'+str(max_turn)+'.txt')
  return FILENAME_IN

def get_filename_out(min_turn, max_turn):
  FILENAME_OUT = os.path.join('.', 'training_data', 'random_board_results_'+str(min_turn)+'_to_'+str(max_turn)+'.txt')
  return FILENAME_OUT


def board_to_result_obj(BOT, read_obj, games_to_average=5):
  """
  There's some trickiness here. Because, it's not actually about who wins,
  its about whether YOU win. And for the purposes of the policy network, YOU
  are the person who just moved. Because, you look ahead one move and you say,
  which has the best value for ME, the person before that board? So,
  if the next person to go is black, it's the value for WHITE. Meaning that 
  if determine_winner gives you 1, and it's white's (-1) turn now, that is good.
  
  white_to_go_value : chance BLACK wins given that white is to go next.
  -- should just be determine_winner().
  black_to_go_value : chance WHITE wins given that black is to go next.
  -- should be -1 * determine_winner(). Because if black wins, that's bad.

  # And when I train on it, I'll do black_to_go_average_results on a board saying
  # that black is ME and white is YOU, etc.

  """
  board = read_obj['board']
  np_board = np.asarray(board, dtype=np.float32)
  turn = read_obj['turn']
  num_moves = read_obj['num_moves']
  total_val = 0.0

  start = time.time()


  for i in range(games_to_average):
    result = clean_convnet.get_results_of_self_play(BOT, np_board, None, turn, num_moves)
    """
    This has always been the tricky part to parse in your head.
    If it's black's turn to go, that means that the get_best_move will
    be calling it from white. Meaning that you want the person who
    is NOT going next to win. So -1*turn*result.
    If turn is -1, that means that result=1 will yield 1, which is right.
    If turn is -1, a result of -1 will yield -1, which is also right.
    """
    if result is None:
      print "\n\n\n\nunterminating game. Don't know why, but it's bad.\n\n\n\n"
      continue
    total_val += (-1.0*turn*result)

  finish = time.time()
  print "one simulation took: " + str(finish - start) + " seconds"

  average_val = (total_val / games_to_average)
  return {
    'board' : board,
    'turn' : turn,
    'num_moves' : num_moves,
    'val' : average_val
  }

  # for i in range(games_to_average):
  #   black_to_go_total_value += (-1.0 * util.determine_random_winner_of_board(np_board, 1, None))
  #   white_to_go_total_value += util.determine_random_winner_of_board(np_board, -1, [])

  # black_to_go_average_value = (black_to_go_total_value + 0.0) / games_to_average
  # white_to_go_average_value = (white_to_go_total_value + 0.0) / games_to_average

  # return {
  #   'board' : board,
  #   'black_to_go_average_value' : black_to_go_average_value,
  #   'white_to_go_average_value' : white_to_go_average_value
  # }

# def read_boards_write_results(write_path):
#   i = 0
#   with open(write_path, 'a') as f_out:
#     for read_obj in random_board_iterator():
#       i += 1
#       if i % 100 == 0:
#         print 'finished board: ' + str(i)
#       # board_np = np.asarray(board, dtype=np.float32)
#       result_obj = board_to_result_obj(read_obj)
#       json_result = json.dumps(result_obj)
#       f_out.write(json_result)
#       f_out.write('\n')
#   print 'done with writing'


# Q_in = Queue()
# Q_out = Queue()

# def worker():
#   board = Q.get()
#   result_obj = board_to_result_obj(board)


# BOARD_QUEUE = Queue(maxsize=100)

def worker_loader(b_queue, FILENAME_IN):
  i = 0
  with open(FILENAME_IN, 'r') as f_in:
    while True:
      i += 1
      line = f_in.readline()
      line = line.strip()
      if line == '':
        print 'END OF FILE'
        break
      board_obj = json.loads(line)
      b_queue.put(board_obj, block=True, timeout=50)
      # print('put in number ' + str(i))
      # size = b_queue.qsize()
      # print('size: ' + str(size))
      # print str(len(b_queue))
      # print 'result loaded'

      # if i % 100 == 0:
      #   print "read in " + str(i)
      

# BOARD_RESULT_QUEUE = Queue(maxsize=100)

def worker_transform(b_queue, r_queue):
  print 'INITIALIZING BOT'
  BOT = clean_convnet.get_best_bot()
  print 'BOT INITIALIZED'
  i = 0
  while True:
    try:
      i += 1
      if i % 50 == 0:
        print 'transforming number ' + str(i) + ' for this queue'
      board_obj = b_queue.get(block=True, timeout=50)#, timeout=10)
      # print 'board to result:'
      result_obj = board_to_result_obj(BOT, board_obj)
      # print 'board to result completed.'
      r_queue.put(result_obj)
    except Exception as e:
      print 'exception in transformer!'
      print e
      break

def worker_writer(r_queue, FILENAME_OUT):
  print "IN WORKER WRITER"
  print "WRITING TO FILE: " + str(FILENAME_OUT)
  i = 0
  time_now = time.time()
  print "In Write Results: "
  print "FIlename Out: " + str(FILENAME_OUT)
  with open(FILENAME_OUT, 'a', buffering=1) as f_out:
    while True:
      try:
        i += 1
        result_obj = r_queue.get(block=True, timeout=50)#, timeout=10)
        json_result = json.dumps(result_obj)
        f_out.write(json_result)
        f_out.write('\n')
        if i % 50 == 0:
          print "wrote out " + str(i)
        if i % 100 == 0:
          print "DONE with 100. took " + str(time.time() - time_now) + " time"
      except Exception as e:
        print 'exception in writer!'
        print e
        print type(e)
        break




if __name__ == '__main__':
  print 'starting'
  # Mandatory arguments.
  BOARD_QUEUE = Queue(maxsize=1000)
  BOARD_RESULT_QUEUE = Queue(maxsize=1000)
  NUM_WORKERS = None
  MIN_TURN = None
  MAX_TURN = None 
  if len(sys.argv) != 4:
    print "Bad arguments. Should be three optional ones."
    print str(sys.argv)
    raise Exception("Include all arguments next time. Its your own fault really")
  NUM_WORKERS = int(sys.argv[1])
  MIN_TURN = int(sys.argv[2])
  MAX_TURN = int(sys.argv[3])
  if math.isnan(NUM_WORKERS) or math.isnan(MIN_TURN) or math.isnan(MAX_TURN):
    print "One of the arguments wasn't a number. here they are."
    print str(sys.argv)
    raise Exception('Non-integer arguments. Try harder next time alright?')
  FILENAME_IN = get_filename_in(MIN_TURN, MAX_TURN)
  FILENAME_OUT = get_filename_out(MIN_TURN, MAX_TURN)


  print 'num workers: ' + str(NUM_WORKERS)
  proc_in = Process(target=worker_loader, args=[BOARD_QUEUE, FILENAME_IN])
  proc_in.start()
  time.sleep(1.0)
  proc_out = Process(target=worker_writer, args=[BOARD_RESULT_QUEUE, FILENAME_OUT])
  proc_out.start()
  for i in range(NUM_WORKERS):
    proc_transform = Process(target=worker_transform, args=[BOARD_QUEUE,BOARD_RESULT_QUEUE])
    proc_transform.start()
    print "process kicked off: " + str(i)
  print "processes kicked off"





