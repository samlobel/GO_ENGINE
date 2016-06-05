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

  """
  np_board = np.asarray(board)
  black_to_go_total_value = 0.0
  white_to_go_total_value = 0.0
  for i in range(games_to_average):
    black_to_go_total_value += util.determine_random_winner_of_board(np_board, 1, [])
    white_to_go_total_value += util.determine_random_winner_of_board(np_board, -1, [])

  black_average_results = (black_total_results + 0.0) / games_to_average
  white_average_results = (white_total_results + 0.0) / games_to_average

  # result_for_black = util.determine_random_winner_of_board(np_board, 1, [])
  # result_for_white = util.determine_random_winner_of_board(np_board, -1, [])
  return {
    'board' : board,
    'blacks_first_result' : black_average_results,
    'whites_first_result' : white_average_results
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



