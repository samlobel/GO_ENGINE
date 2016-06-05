import json
import numpy as np

import sys
import os

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

def board_to_result_obj(board):
  np_board = np.asarray(board)
  result_for_black = util.determine_random_winner_of_board(np_board, 1, [])
  result_for_white = util.determine_random_winner_of_board(np_board, -1, [])
  return {
    'board' : board,
    'blacks_first_result' : result_for_black,
    'whites_first_result' : result_for_white
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


if __name__ == '__main__':
  print 'read_boards: '
  read_boards_write_results('./random_board_results.txt')
  print 'done!'



