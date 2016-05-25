import sys
import os
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.'))


this_dir = os.path.dirname(os.path.realpath(__file__))

from go_util import util


BOARDS_FILE = os.path.join(this_dir, 'random_boards.txt')
BOARD_SHAPE = (9,9)
def write_random_board(num_moves):
  board, turn = util.generate_random_board(BOARD_SHAPE, num_moves)
  if board is None:
    print "board is none."
    return
  board_ = np.ndarray.tolist(board)
  board_str = json.dumps(board_)
  with open(BOARDS_FILE, 'a') as file:
    file.write(board_str)
    file.write('\n')
  # print "board written"


if __name__ == '__main__':
  num = 0
  for i in xrange(1500)
  while True:
    num += 1
    print "cycle complete " + str(num)   
    for i in xrange(100):
      write_random_board(i)
