import json
import numpy as np

import sys
import os

from Queue import Queue
import time
from threading import Thread

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from go_util import util









def random_board_iterator():
  with open('./random_boards.txt') as f:
    while True:
      to_yield = f.readline().strip()
      if to_yield == '':
        break
      try:
        to_yield =  json.loads(to_yield)
        yield to_yield
      except Exception:
        print "fail on to_yield=" + str(to_yield)
  print 'exit iterator'

def board_to_result_obj(board, games_to_average=5):
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
  np_board = np.asarray(board)
  black_to_go_total_value = 0.0
  white_to_go_total_value = 0.0
  for i in range(games_to_average):
    black_to_go_total_value += (-1.0 * util.determine_random_winner_of_board(np_board, 1, []))
    white_to_go_total_value += util.determine_random_winner_of_board(np_board, -1, [])

  black_to_go_average_value = (black_to_go_total_value + 0.0) / games_to_average
  white_to_go_average_value = (white_to_go_total_value + 0.0) / games_to_average

  return {
    'board' : board,
    'black_to_go_average_value' : black_to_go_average_value,
    'white_to_go_average_value' : white_to_go_average_value
  }

def read_boards_write_results(write_path):
  i = 0
  with open(write_path, 'a') as f_out:
    for board in random_board_iterator():
      i += 1
      if i % 100 == 0:
        print 'finished board: ' + str(i)
      # board_np = np.asarray(board, dtype=np.float32)
      result_obj = board_to_result_obj(board)
      json_result = json.dumps(result_obj)
      f_out.write(json_result)
      f_out.write('\n')
  print 'done with writing'


# Q_in = Queue()
# Q_out = Queue()

# def worker():
#   board = Q.get()
#   result_obj = board_to_result_obj(board)


BOARD_QUEUE = Queue(maxsize=100)

def worker_loader():
  i = 0
  filename_in = './random_boards.txt'
  with open(filename_in, 'r') as f_in:
    while True:
      i += 1
      line = f_in.readline()
      line = line.strip()
      if line == '':
        print 'END OF FILE'
        break
      board_obj = json.loads(line)
      BOARD_QUEUE.put(board_obj)
      if i % 100 == 0:
        print "read in " + str(i)
      

BOARD_RESULT_QUEUE = Queue(maxsize=100)

def worker_transform():
  while True:
    try:
      board_obj = BOARD_QUEUE.get(block=True, timeout=10)
      result_obj = board_to_result_obj(board_obj)
      BOARD_RESULT_QUEUE.put(result_obj)
    except Exception:
      print 'exception in writer!'
      break

def worker_writer():
  filename_out = './random_board_results_from_queue.txt'
  i = 0
  time_now = time.time()
  with open(filename_out, 'w', buffering=1) as f_out:
    while True:
      try:
        i += 1
        result_obj = BOARD_RESULT_QUEUE.get(block=True, timeout=10)
        json_result = json.dumps(result_obj)
        f_out.write(json_result)
        f_out.write('\n')
        if i % 100 == 0:
          print "wrote out " + str(i)
        if i % 200 == 0:
          print "DONE. took " + str(time.time()) + " time"
      except Exception:
        print 'exception in writer!'
        break



    




if __name__ == '__main__':
  # print 'read_boards: '
  # read_boards_write_results('./random_board_results.txt')
  # print 'done!'

  NUM_TRANSFORM_THREADS = 4

  print 'read boards:'
  T_read = Thread(target=worker_loader)
  T_read.daemon = True
  T_read.start()
  T_write = Thread(target=worker_writer)
  T_write.daemon = True
  T_write.start()
  for i in range(NUM_TRANSFORM_THREADS):
    T_write = Thread(target=worker_transform)
    T_write.daemon = True
    T_write.start()
    print 'thread ' + str(i) + ' kicked off.'

  print 'threads kicked off.'
  BOARD_QUEUE.join()
  BOARD_RESULT_QUEUE.join()

  print 'done'



