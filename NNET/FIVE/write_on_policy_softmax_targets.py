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


BOARDS_FILE = os.path.join(this_dir, 'random_boards.txt')
BOARD_SHAPE = (5,5)


from NNET.FIVE.convnet_better_policy_value import \
  Convbot_FIVE_POLICY_VALUE_NEWEST, get_largest_convbot_from_folder, board_to_input_transform_value


FOLDER_NAME = 'test'




# def random_board_iterator():
#   with open(BOARDS_FILE) as f:
#     while True:
#       to_yield = f.readline().strip()
#       if to_yield == '':
#         break
#       try:
#         to_yield =  json.loads(to_yield)
#         yield to_yield
#       except Exception:
#         print "fail on to_yield=" + str(to_yield)
#   print 'exit iterator'


def board_to_result_obj(convbot, board, current_turn=1, temp=0.5):
  board_np = np.asarray(board, dtype=np.float32)
  board_input = board_to_input_transform_value(board_np, [], current_turn)
  softmax_for_board = convbot.create_softmax_from_value_function(board_np, [], current_turn, temperature=temp)
  softmax_array = softmax_for_board.tolist()
  board_input_array = board_input.tolist()

  result_obj = {
    'current_turn' : current_turn,
    'input_board' : board,
    'board_input_serialized' : board_input_array,
    'softmax_output_serialized' : softmax_array
  }
  return result_obj




def worker_loader(b_queue):
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
      b_queue.put(board_obj)
      if i % 100 == 0:
        print "read in " + str(i)


def worker_transform(b_queue, r_queue):
  convbot, _ = get_largest_convbot_from_folder(FOLDER_NAME)
  while True:
    try:
      board_obj = b_queue.get(block=True, timeout=10)
      current_turn = random.choice([-1,1])
      result_obj = board_to_result_obj(convbot, board_obj, current_turn=current_turn)
      r_queue.put(result_obj)
    except Exception:
      print 'exception in writer!'
      break



def worker_writer(r_queue):
  filename_out = './random_board_softmax_output.txt'
  i = 0
  time_now = time.time()
  with open(filename_out, 'a', buffering=1) as f_out:
    while True:
      try:
        i += 1
        result_obj = r_queue.get(block=True, timeout=10)
        json_result = json.dumps(result_obj)
        f_out.write(json_result)
        f_out.write('\n')
        if i % 25 == 0:
          print "wrote out " + str(i)
        if i % 100 == 0:
          print "DONE. took " + str(time.time() - time_now) + " time"
      except Exception:
        print 'exception in writer!'
        break





# def test():
#   convbot, largest = get_largest_convbot_from_folder(FOLDER_NAME)
#   for board in random_board_iterator():
#     result_obj = board_to_result_obj(convbot, board, 1, 0.5)
#     print result_obj


def kick_off():
  print 'starting'
  BOARD_QUEUE = Queue(maxsize=100)
  BOARD_RESULT_QUEUE = Queue(maxsize=100)
  NUM_WORKERS = None
  try:
    NUM_WORKERS = int(sys.argv[1])
  except:
    NUM_WORKERS=4
  print 'num workers: ' + str(NUM_WORKERS)
  proc_in = Process(target=worker_loader, args=[BOARD_QUEUE])
  proc_in.start()
  proc_out = Process(target=worker_writer, args=[BOARD_RESULT_QUEUE])
  proc_out.start()
  for i in range(NUM_WORKERS):
    proc_transform = Process(target=worker_transform, args=[BOARD_QUEUE,BOARD_RESULT_QUEUE])
    proc_transform.start()
    print "process kicked off: " + str(i)
  print "processes kicked off"





if __name__ == '__main__':
  kick_off()





