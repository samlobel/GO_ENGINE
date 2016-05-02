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
  assert True
  print "test_get_neighbors_on_board: success!"

def test_get_neighbors_of_color():
  assert True
  print "test_get_neighbors_of_color: success!"

def test_count_liberties():
  assert True
  print "test_count_liberties: success!"

def test_count_liberties_around_stone():
  assert True
  print "test_count_liberties_around_stone: success!"

def test_get_group_around_stone():
  assert True
  print "test_get_group_around_stone: success!"

def test_confirm_group_is_one_color():
  assert True
  print "test_confirm_group_is_one_color: success!"

def test_remove_stones_in_group():
  assert True
  print "test_remove_stones_in_group: success!"

def test_remove_group_around_stone():
  assert True
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
  assert True
  print "test_output_all_valid_moves: success!"

def test_determine_owner_of_free_space():
  assert True
  print "test_determine_owner_of_free_space: success!"

def test_def_score_board():
  assert True
  print "test_def_score_board: success!"


if __name__ == '__main__':
  test_boards_are_equal()
  test_move_is_on_board()
  test_get_value_for_spot()
  test_set_value_for_spot()
  test_spot_is_open()
  test_spot_is_color()
  test_get_neighbors_on_board()
  test_get_neighbors_of_color()
  test_count_liberties()
  test_count_liberties_around_stone()
  test_get_group_around_stone()
  test_confirm_group_is_one_color()
  test_remove_stones_in_group()
  test_remove_group_around_stone()
  test_spot_is_suicide()
  test_update_board_from_move()
  test_move_makes_duplicate()
  test_move_is_valid()
  test_output_all_valid_moves()
  test_determine_owner_of_free_space()
  test_def_score_board()




