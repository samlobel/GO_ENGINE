import numpy as np
import tensorflow as tf

from copy import deepcopy as copy

from util import *

from sample_boards import *





def test_boards_are_equal():
  assert boards_are_equal(copy(b1_4x4), copy(b1_4x4))
  assert boards_are_equal(copy(b2_4x4), copy(b2_4x4))
  assert not boards_are_equal(copy(b1_4x4), copy(b2_4x4))
  assert not boards_are_equal(copy(b2_4x4), copy(b3_4x4))
  print "test_boards_are_equal: success!"

def test_move_is_on_board():
  assert move_is_on_board(copy(b1_4x4), (3,3))
  assert not move_is_on_board(copy(b1_4x4), (3,4))
  assert not move_is_on_board(copy(b1_4x4), (-1,2))
  assert move_is_on_board(copy(b1_4x4), (0,3))
  try:
    a = move_is_on_board("not an array", (3,3))
    print "shouldn't get here!"
    assert False
  except Exception as e:
    # print "good, got here."
    assert True 
  try:
    a = move_is_on_board(copy(b1_4x4), "stevie wonder")
    print "shouldn't get here!"
    assert False
  except Exception as e:
    # print "good, got here."
    assert True

  print "test_move_is_on_board: success!"



def test_get_value_for_spot():
  # print get_value_for_spot(copy(b1_4x4), (0,0))
  assert (get_value_for_spot(copy(b1_4x4), (0,0)) == 0)
  assert (get_value_for_spot(copy(b1_4x4), (3,0)) == 0)
  assert (get_value_for_spot(copy(b1_4x4), (0,3)) == 0)
  assert (get_value_for_spot(copy(b2_4x4), (0,1)) == 1)
  assert (get_value_for_spot(copy(b2_4x4), (0,2)) == -1)
  print "test_get_value_for_spot: success!"


def test_set_value_for_spot():
  copy_b1 = copy(b1_4x4)
  assert (get_value_for_spot(copy_b1, (0,0)) == 0)
  set_value_for_spot(copy_b1, (0,0), 1)
  assert (get_value_for_spot(copy_b1, (0,0)) == 1)
  set_value_for_spot(copy_b1, (0,0), -1)
  assert (get_value_for_spot(copy_b1, (0,0)) == -1)
  print "test_set_value_for_spot: success!"

def test_spot_is_open():
  for move_tuple in move_tuples_on_board(b1_4x4):
    assert spot_is_open(b1_4x4, move_tuple)
  assert spot_is_open(b2_4x4, (0,0))
  assert not spot_is_open(b2_4x4, (1,0))
  assert not spot_is_open(b2_4x4, (1,1))
  print "test_spot_is_open: success!"
  


def test_spot_is_color():
  for move_tuple in move_tuples_on_board(b1_4x4):
    assert spot_is_color(b1_4x4, move_tuple, 0)
  assert spot_is_color(b2_4x4, (0,0), 0)
  assert spot_is_color(b2_4x4, (1,0), 1)
  assert spot_is_color(b2_4x4, (1,1), 1)
  print "test_spot_is_color: success!"

def test_get_neighbors_on_board():
  assert (set(get_neighbors_on_board(b1_4x4, (0,0))) == set([(0,1),(1,0)]))
  assert (set(get_neighbors_on_board(b1_4x4, (1,1))) == set([(0,1),(1,0),(1,2),(2,1)]))
  assert (set(get_neighbors_on_board(b1_4x4, (3,3))) == set([(3,2),(2,3)]))
  print "test_get_neighbors_on_board: success!"

def test_get_neighbors_of_color():
  assert (set(get_neighbors_of_color(b1_4x4, (0,0), 0)) == set([(0,1),(1,0)]))
  assert (set(get_neighbors_of_color(b1_4x4, (1,1), 0)) == set([(0,1),(1,0),(1,2),(2,1)]))
  assert (set(get_neighbors_of_color(b1_4x4, (1,1), 0)) == set([(0,1),(1,0),(1,2),(2,1)]))
  assert (set(get_neighbors_of_color(b2_4x4, (1,1), 0)) == set([]))
  assert (set(get_neighbors_of_color(b2_4x4, (1,1), 1)) == set([(0,1),(1,0)]))
  assert (set(get_neighbors_of_color(b2_4x4, (1,1), -1)) == set([(2,1),(1,2)]))
  print "test_get_neighbors_of_color: success!"

# def test_count_liberties():
#   assert True
#   print "test_count_liberties: success!"

def test_count_liberties_around_stone():
  assert (count_liberties_around_stone(b2_4x4, (1,1)) == 1)
  assert (count_liberties_around_stone(b2_4x4, (0,1)) == 1)
  assert (count_liberties_around_stone(b2_4x4, (0,2)) == 6)
  assert (count_liberties_around_stone(b4_4x4, (1,1)) == 4)
  assert (count_liberties_around_stone(b4_4x4, (1,2)) == 4)
  assert (count_liberties_around_stone(b4_4x4, (2,2)) == 4)
  assert (count_liberties_around_stone(b3_4x4, (3,3)) == 1)
  assert (count_liberties_around_stone(b5_4x4, (0,0)) == 4)
  assert (count_liberties_around_stone(b5_4x4, (1,1)) == 0)
  print "test_count_liberties_around_stone: success!"

def test_get_group_around_stone():
  assert (get_group_around_stone(b3_4x4, (0,0)) == \
      set([(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(2,0),(1,0)]))
  assert(get_group_around_stone(b3_4x4, (1,1)) == set([(1,1),(1,2),(2,2)]))
  print "test_get_group_around_stone: success!"

# def test_confirm_group_is_one_color():
#   confirm_group_is_one_color(b3_4x4, get_group_around_stone(b3_4x4, (0,0)))
#   confirm_group_is_one_color(b3_4x4, get_group_around_stone(b3_4x4, (1,1)))

#   # assert True
#   print "test_confirm_group_is_one_color: success!"

# def test_remove_stones_in_group():
#   copy_3 = copy(b3_4x4)
#   end_goal = np.asarray(
#     [[ 0, 0, 0, 0],
#      [ 0,-1,-1, 0],
#      [ 0, 0,-1, 0],
#      [ 0, 0, 0, 0]]
#   )
#   edited_3 = remove_stones_in_group()
#   assert True
#   print "test_remove_stones_in_group: success!"


def test_count_liberties_around_group():
  b2_end_goal = np.asarray([
    [ 0, 1,-6, 0],
    [ 1, 1,-6, 0],
    [-6,-6,-6, 0],
    [ 0, 0, 0, 0]
  ])
  l_map = output_liberty_map(b2_4x4)
  # print l_map
  # print b2_end_goal
  assert boards_are_equal(b2_end_goal, l_map)
  assert boards_are_equal(b1_4x4, output_liberty_map(b1_4x4)) 

def test_remove_group_around_stone():
  copy_3 = copy(b3_4x4)
  end_goal = np.asarray(
    [[ 0, 0, 0, 0],
     [ 0,-1,-1, 0],
     [ 0, 0,-1, 0],
     [ 0, 0, 0, 0]]
  )
  remove_group_around_stone(copy_3, (0,0))
  # print copy_3
  assert boards_are_equal(end_goal, copy_3)

  copy_3 = copy(b3_4x4)
  end_goal = np.asarray(
    [[ 1, 1, 1, 1],
     [ 1, 0, 0, 1],
     [ 1, 0, 0, 1],
     [ 1, 1, 1, 1]]
  )
  remove_group_around_stone(copy_3, (1,1))
  assert boards_are_equal(end_goal, copy_3)
  print "test_remove_group_around_stone: success!"

def test_spot_is_suicide():
  assert True
  print "test_spot_is_suicide: success!"

def test_update_board_from_move():
  assert True
  print "test_update_board_from_move: success!"

def test_move_makes_duplicate():
  assert True
  print "test_move_makes_duplicate: success!"

def test_move_is_valid():
  assert True
  print "test_move_is_valid: success!"

def test_output_all_valid_moves():
  all_valid = set(output_all_valid_moves(copy(b1_4x4), copy(b1_4x4), 1))
  all_moves = set(move_tuples_on_board(b1_4x4))
  assert all_moves == all_valid
  all_valid = set(output_all_valid_moves(copy(b1_4x4), copy(b1_4x4), -1))
  assert all_moves == all_valid
  print "test_output_all_valid_moves: success!"

def test_determine_owner_of_free_space():
  assert True
  print "test_determine_owner_of_free_space: success!"

def test_def_score_board():
  assert True
  print "test_def_score_board: success!"

def test_generate_random_board():
  pass

def test_move_is_eye():
  assert not move_is_eye(copy(b7_4x4), (0,0),1)
  assert move_is_eye(copy(b7_4x4), (3,0),1)
  assert not move_is_eye(copy(b7_4x4), (1,2),-1)


if __name__ == '__main__':
  test_move_is_eye()

  # test_boards_are_equal()
  # test_move_is_on_board()
  # test_get_value_for_spot()
  # test_set_value_for_spot()
  # test_spot_is_open()
  # test_spot_is_color()
  # test_get_neighbors_on_board()
  # test_get_neighbors_of_color()
  # # test_count_liberties() #Only a helper function. Can test through parent function.
  # test_count_liberties_around_stone()
  # test_get_group_around_stone()
  # test_count_liberties_around_group()
  # # test_confirm_group_is_one_color()
  # # test_remove_stones_in_group()
  # test_remove_group_around_stone()
  # test_spot_is_suicide()
  # test_update_board_from_move()
  # test_move_makes_duplicate()
  # test_move_is_valid()
  # test_output_all_valid_moves()
  # test_determine_owner_of_free_space()
  # test_def_score_board()




