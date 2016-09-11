import sys
import os
import numpy as np
import json
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))


this_dir = os.path.dirname(os.path.realpath(__file__))

from go_util import util

from multiprocessing import Process, Queue
import random


# BOARDS_FILE = os.path.join(this_dir, 'training_data','random_boards.txt')
BOARD_SHAPE = (5,5)

MIN_TURN=10
MAX_TURN=19

# BOARDS_FILE = os.path.join(this_dir, \
#   'training_data','random_boards_'+str(MIN_TURN)+'_to_'+str(MAX_TURN)+'.txt')

def get_boards_file(min_turn, max_turn):
  return os.path.join(this_dir, \
    'training_data','random_boards_'+str(min_turn)+'_to_'+str(max_turn)+'.txt')





# def write_random_board(num_moves):
#   board, turn = util.generate_random_board(BOARD_SHAPE, num_moves)
#   if board is None:
#     print "board is none."
#     return
#   to_write = {'board' : board, 'turn' : turn, 'num_moves': num_moves}
#   board_ = np.ndarray.tolist(board)
#   to_write = {'board' : board_, 'turn' : turn, 'num_moves': num_moves}
#   to_write = json.dumps(to_write)
#   with open(BOARDS_FILE, 'a') as file:
#     file.write(to_write)
#     file.write('\n')
#   # print "board written"

def return_random_board(num_moves):
  board, turn = util.generate_random_board(BOARD_SHAPE, num_moves)
  if board is None:
    print "board is none."
    return None
  board_ = np.ndarray.tolist(board)
  to_return = {
    'board' : board_,
    'num_moves' : num_moves,
    'turn' : turn
  }
  # board_str = json.dumps(board_)
  return to_return
  # with open(BOARDS_FILE, 'a') as file:
  #   file.write(board_str)
  #   file.write('\n')
  # print "board written"

def worker_transform(b_queue, r_queue, MIN_TURN, MAX_TURN):
  while True:
    try:
      i = b_queue.get(block=True, timeout=50)
      j = random.randchoice(MIN_TURN,MAX_TURN)
      board_arr = return_random_board(j)
      if board_arr is None:
        continue
      board_str = json.dumps(board_arr)
      board_str += '\n'
      r_queue.put(board_str)
    except Exception:
      print 'exception in writer!'
      break

def worker_writer(r_queue, BOARDS_FILE):
  i = 0
  time_now = time.time()
  with open(BOARDS_FILE, 'a', buffering=1) as f_out:
    while True:
      try:
        board_str = r_queue.get(block=True, timeout=50)
        f_out.write(board_str)
        i += 1
        if i % 1000 == 0:
          print "boards written: " + str(i)
        if i % 5000 == 0:
          print "time taken: " + str(time.time() - time_now)
      except Exception:
        print 'end of results queue!'
        break










if __name__ == '__main__':
  num = 0
  Q_in = Queue(maxsize=10000)
  for i in range(8000):
    Q_in.put(i)

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
  if MIN_TURN >= MAX_TURN:
    print "MIN SHOULD BE LESS THAN MAX, DOOFUS"
    raise Exception("MIN SHOULD BE LESS THAN MAX, DOOFUS")

  BOARDS_FILE = get_boards_file(MIN_TURN,MAX_TURN)
  Q_out = Queue(maxsize=100)
  proc_out = Process(target=worker_writer, args=[Q_out, BOARDS_FILE])
  proc_out.start()
  for j in range(NUM_WORKERS):
    proc_transform = Process(target=worker_transform, args=[Q_in,Q_out])
    proc_transform.start()
    print "process kicked off: " + str(j)
  print "processes kicked off"



  # for cycle in xrange(9000):
  # # while True:
  #   num += 1
  #   print "cycle complete " + str(num)
  #   for i in xrange(25):
  #     write_random_board(i)
