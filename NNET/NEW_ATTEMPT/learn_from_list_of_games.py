import json
import numpy as np

import sys
import os

# from Queue import Queue
import time
from multiprocessing import Process, Queue
# from threading import Thread

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))

from go_util import util

from NNET.NEW_ATTEMPT import clean_convnet

MIN_TURN = 10
MAX_TURN=19
def get_filename_in(MIN_TURN, MAX_TURN):
    return os.path.join('.', 'training_data', 'random_board_results_'+str(MIN_TURN)+'_to_'+str(MAX_TURN)+'.txt')
# FILENAME_IN = os.path.join('.', 'training_data', 'random_board_results_'+str(MIN_TURN)+'_to_'+str(MAX_TURN)+'.txt')

TIMES_THROUGH = 1

def board_results_iterator():
  BATCH_SIZE = 200
  with open(FILENAME_IN, 'r') as f:
    while True:
      to_yield = [f.readline() for i in xrange(BATCH_SIZE)]
      if to_yield[-1] == '':
        print('done with iteration, hit end of file.')
        # print(to_yield)
        break
      to_yield = [json.loads(x.strip()) for x in to_yield]
      yield to_yield
  print("exit iterator")


if __name__ == '__main__':
  BOT = clean_convnet.get_best_bot()
  err_arr = []
  if len(sys.argv) != 3:
    print "Bad arguments. Should be three optional ones."
    print str(sys.argv)
    raise Exception("Include all arguments next time. Its your own fault really")
  MIN_TURN = int(sys.argv[1])
  MAX_TURN = int(sys.argv[2])

  if math.isnan(MIN_TURN) or math.isnan(MAX_TURN):
    print "One of the arguments wasn't a number. here they are."
    print str(sys.argv)
    raise Exception('Non-integer arguments. Try harder next time alright?')
  MIN_TURN_VN_IND = BOT.turn_number_to_network_index(MIN_TURN)
  MAX_TURN_VN_IND = BOT.turn_number_to_network_index(MAX_TURN)
  if MIN_TURN_VN_IND != MAX_TURN_VN_IND:
    print "doesn't match."
    raise Exception("Min and Max are from different indices, ya dingus!")
  VN_NUMBER = MIN_TURN_VN_IND


  for i in range(TIMES_THROUGH):
    print 'DOING IT NOW. Informative...'
    for result_obj_list in board_results_iterator():
      err = BOT.learn_from_list_of_games(result_obj_list, VN_NUMBER)
      err_arr.append(err)
      if len(err_arr) % 100 == 0:
        print err_arr
        print len(err_arr)
    print "DONE! Saving."
    BOT.save_in_next_slot()
    print "Done. Saved" 
    config = BOT.load_config()
    config[VN_NUMBER] = True
    BOT.write_config(config)

