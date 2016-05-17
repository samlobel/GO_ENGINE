import numpy as np
import tensorflow as tf

from copy import deepcopy as copy
import random


# def get_board_shape(board_matrix):
#   return board.shape

"""
An interesting thing is that, maybe to prevent repitition, the only
thing we can do is pass in the ENTIRE go history. I don't know,
this seems a little excessive, but it would prevent the repetition that
I get stuck on all the time. It would be a pretty expensive operation
is the only thing.

I would have to compare position AND whose turn it is, because it's
a different position if its a different player.

I think I do need to do this. That sucks, that's a lot of work. But I've got to.

"""





def flatten_list(l):
  return [item for sublist in l for item in sublist]


def unflatten_list(l, desired_shape):
  return np.asarray(l).reshape(desired_shape)




def boards_are_equal(b1, b2):
  return np.array_equal(b1,b2)

def board_is_empty(b1):
  if b1.shape is None:
    raise Exception('non-numpy array passed to board_is_empty')
  b2 = np.zeros(b1.shape)
  return boards_are_equal(b1,b2)


def from_move_tuple_to_index(board_matrix, tup):
  shape = board_matrix.shape
  if shape is None:
    raise Exception("bad board passed to from_move_tuple_to_index")
  BOARD_SIZE = shape[0]
  if tup is None:
    return BOARD_SIZE*BOARD_SIZE
  else:
    r,c = tup
    if (r >= BOARD_SIZE) or (c >= BOARD_SIZE):
      raise Exception("rows and columns must both be present on board")
    elif (r < 0) or (c < 0):
      raise Exception("rows and columns must both be present on board")
    else:
      index = BOARD_SIZE*r + c
      return index



def move_tuples_on_board(board_matrix):
  length, width = board_matrix.shape
  for l in xrange(length):
    for w in xrange(width):
      move_tuple = (l,w)
      yield move_tuple


def move_is_on_board(board_matrix, move_tuple):
  """
  A simple check to see if a move is on the board, or out of bounds. Utility function.
  """


  if type(move_tuple) is not tuple:
    raise Exception("move tuple must be a tuple, in move_is_on_board")

  board_shape_x, board_shape_y = board_matrix.shape
  if board_shape_x != board_shape_y:
    raise Exception("Non-square board (or non-numpy board) passed to move_is_on_board")

  if (move_tuple[0] < 0) or (move_tuple[0] >= board_shape_x):
    return False
  
  if (move_tuple[1] < 0) or (move_tuple[1] >= board_shape_y):
    return False

  return True

def get_value_for_spot(board_matrix, move_tuple):
  """
  such a Utility function.
  """
  if not move_is_on_board(board_matrix, move_tuple):
    print "checking value for spot off board"
    raise Exception("illegal move passed to get_value_for_spot")
  return board_matrix[move_tuple[0]][move_tuple[1]]


def set_value_for_spot(board_matrix, move_tuple, intended_value):
  """
  In place.
  """
  if not move_is_on_board(board_matrix, move_tuple):
    print "checking value for spot off board"
    raise Exception("illegal move passed to get_value_for_spot")
  board_matrix[move_tuple[0]][move_tuple[1]] = intended_value






def spot_is_open(board_matrix, move_tuple):
  """
  A simple check to see if a spot is open. Utility funciton.
  """
  if not move_is_on_board(board_matrix, move_tuple):
    print "move passed to spot_is_open is not on board!"
    raise Exception("illegal move passed to spot_is_open")

  return (get_value_for_spot(board_matrix, move_tuple) == 0)


def spot_is_color(board_matrix, move_tuple, color):
  if not move_is_on_board(board_matrix, move_tuple):
    print "move passed to spot_is_open is not on board!"
    raise Exception("illegal move passed to spot_is_open")

  return (get_value_for_spot(board_matrix, move_tuple) == color)
  

def get_neighbors_on_board(board_matrix, spot_tuple):
  """
  A simple check to get a spot's surrounding peices, if they're on the board.
  """
  if not move_is_on_board(board_matrix, spot_tuple):
    print "move passed to spot_is_open is not on board!"
    raise Exception("illegal move passed to spot_is_open")
  x, y = spot_tuple
  possible_spots = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
  possible_spots = [p for p in possible_spots if move_is_on_board(board_matrix, p)]
  return possible_spots


def get_neighbors_of_color(board_matrix, spot_tuple, color):
  if color not in (-1,0,1):
    raise Exception("Invalid color passed to get_neighbors_of_color")
  all_neighbors = get_neighbors_on_board(board_matrix, spot_tuple)
  neighbors_of_color = [n for n in all_neighbors if spot_is_color(board_matrix, n, color)]
  return neighbors_of_color
  

# Actually, the easiest way to check if a spot is suicide is, copy the board, 
# and then you can just check the liberties of the whole thing. No need to 
# do this nonsense with treating the first move differently.






def count_liberties(board_matrix, set_of_seen, border_set, current_player,
        liberties_so_far=0):
  """
  So, if something is in the border set, it's not in the set of seen. It's 
  a BFS!
  """
  if current_player not in (-1,1):
    print "current player not -1 or 1."
    raise Exception("bad validation for current_player in count_liberties")

  set_of_seen = set([])
  while len(border_set) != 0:
    set_of_seen = set_of_seen.union(border_set)
    new_border_set = set([])
    for move_tuple in border_set:
      if spot_is_open(board_matrix, move_tuple):
        liberties_so_far += 1
        continue
      elif spot_is_color(board_matrix, move_tuple, current_player):
        neighbors_on_board = set(get_neighbors_on_board(board_matrix, move_tuple))
        new_neighbors_on_board = neighbors_on_board.difference(set_of_seen)
        new_border_set = new_border_set.union(new_neighbors_on_board)
        continue
      else:
        continue
    border_set = new_border_set

  return liberties_so_far

    

  # return liberties_so_far

  # if len(border_set) == 0:
  #   return liberties_so_far

  # # border_set = set(border_set) #might be unnecessary, we'll see.
  # # set_of_seen = set(set_of_seen)
  # new_set_of_seen = set_of_seen.union(border_set)
  # new_border_set = set([])



  # for move_tuple in border_set:
  #   if spot_is_open(board_matrix, move_tuple):
  #     liberties_so_far += 1
  #     continue
  #   elif spot_is_color(board_matrix, move_tuple, current_player):
  #     neighbors_on_board = set(get_neighbors_on_board(board_matrix, move_tuple))
  #     neighbors_on_board = neighbors_on_board.difference(new_set_of_seen)
  #     new_border_set = new_border_set.union(neighbors_on_board)
  #   else:
  #     # This is one of the other color. you just stop.
  #     continue

  # return count_liberties(board_matrix, new_set_of_seen,
  #        new_border_set, current_player, liberties_so_far)




def count_liberties_around_stone(board_matrix, spot_in_group):
  if spot_is_open(board_matrix, spot_in_group):
    raise Exception("Counting liberties of empty spot")

  color = get_value_for_spot(board_matrix, spot_in_group)
  liberties = count_liberties(board_matrix, set([]), set([spot_in_group]), color, 0)
  return liberties


def get_group_around_stone(board_matrix, spot_tuple):
  """
  As an extra check, I should debatably have the color of the stone in this, 
  but that's a little cumbersome
  """
  if spot_is_open(board_matrix, spot_tuple):
    raise Exception("trying to get the group around an empty spot.")

  group_color = get_value_for_spot(board_matrix, spot_tuple)

  border_set = set([spot_tuple])
  set_of_seen = set([])
  while len(border_set) != 0:
    set_of_seen = set_of_seen.union(border_set)
    new_border_set = [get_neighbors_of_color(board_matrix, m_t, group_color) for m_t in border_set]
    new_border_set = set(flatten_list(new_border_set))
    new_border_set = new_border_set.difference(set_of_seen)
    border_set = new_border_set

  return set_of_seen

def count_liberties_around_group(board_matrix, group_set):
  all_neighbors = set([])
  for spot in group_set:
    spot_neighbors = get_neighbors_of_color(board_matrix, spot, 0)
    for spn in spot_neighbors:
      all_neighbors.add(spn)
  return len(all_neighbors)

def output_liberty_map(board_matrix):
  """
  Checks each location. If empty, output zero.
  If not, get total liberties surrounding group. Set value in
  liberty map to this total. Then, set this same value for all
  of the other group members, so that you can save on computation.

  NOTE: For white, it should output negative liberties.

  """
  liberty_map = np.zeros(board_matrix.shape)
  for mt in move_tuples_on_board(board_matrix):
    spot_value = get_value_for_spot(board_matrix, mt)
    if spot_value == 0:
      continue
    elif not spot_is_open(liberty_map, mt):
      # That's if you've already set it.
      continue
    else:
      surrounding_group = get_group_around_stone(board_matrix, mt)
      liberties_surrounding_group = count_liberties_around_group(board_matrix, surrounding_group)
      output_value = spot_value * liberties_surrounding_group
      for spot in surrounding_group:
        set_value_for_spot(liberty_map, spot, output_value)

  return liberty_map





# def confirm_group_is_one_color(board_matrix, group):
#   """
#   This is a sanity check, because I'm doing hard stuff.
#   """
#   if len(group) == 0:
#     return
  
#   colors = set([ get_value_for_spot(board_matrix, s) for s in group ])
#   if len(colors) != 1:
#     raise Exception("Looks like group is not one color, that means I messed up somewhere.")
#   return

def group_is_one_color(board_matrix, group):
  if len(group) == 0:
    return True

  colors = set([ get_value_for_spot(board_matrix, s) for s in group ])
  if len(colors) == 1:
    return True
  else:
    return False


def remove_stones_in_group(board_matrix, group):
  """
  In place. Helper function.
  """
  for spot in group:
    set_value_for_spot(board_matrix, spot, 0)



def remove_group_around_stone(board_matrix, spot_tuple):
  """
  Function that removes a group. Should only be called if the group has no liberties.
  """
  group_around_stone = get_group_around_stone(board_matrix, spot_tuple)
  # confirm_group_is_one_color(board_matrix, group_around_stone)
  remove_stones_in_group(board_matrix, group_around_stone)
  return








def spot_is_suicide(board_matrix, move_tuple, current_player):
  """
  A check for suicide. It's suicide if it makes the piece you just put down
  get taken off the board. This would be the case if you put a piece on
  the only eye. I think I should do a BFS from the spot in question, following 
  a path if it's your color, stopping if it's theirs or wall, and returning false
  if you see an empty.

  Actually, you also need to see if it takes over any of the adjacent spaces, 
  in which case it's okay. To do that, you need to check the liberties of 
  all of the adjacent spots of the opposing color. If it's just one, that means
  that putting it there would capture some of their spots.

  We also need to check if it would make captures. I would do this by checking all
  of the neighbors for spots that are the opposite colors, and seeing if any
  of them would be converted. I guess that's the first thing I should do.
  """
  if not spot_is_open(board_matrix, move_tuple):
    raise Exception("Asking about a taken spot for spot_is_suicide")

  board_copy = copy(board_matrix)
  set_value_for_spot(board_copy, move_tuple, current_player)

  other_player_neighbors = get_neighbors_of_color(board_copy, move_tuple, -1*current_player)
  other_player_liberties = [count_liberties_around_stone(board_copy, m_t) for m_t in other_player_neighbors]
  for o_p_l in other_player_liberties:
    if o_p_l == 0:
      return False
    else:
      continue

  # At this point, we know it won't remove any neighbors.
  this_spot_liberties = count_liberties_around_stone(board_copy, move_tuple)

  if this_spot_liberties == 0:
    return True
  else:
    return False


def update_board_from_move(board_matrix, move_tuple, current_player):
  """
    NOT IN PLACE!
    Maybe, I should make everything not in-place. But that really might slow down my program by like 
    361 times. That sounds pretty dreadful because it's already going to be SO slow.
    Places a stone on the board at the required location. This isn't going to be trivial.
    one way: 
    There are a few steps. The first is clumping the board into groups. The next step is to
    figure out how many free sides these clumps have. If it's zero, you're out of luck.

    NOTE: THIS HAPPENS AFTER YOU SEE IF SOMETHING IS AS SUICIDE, BUT IT IS A STEP
    IN DETERMINING WHETHER SOMETHING IS A LEGAL MOVE, BECAUSE YOU NEED TO SEE IF IT
    DUPLICATES AT ALL.

  """

  if move_tuple is None:
    return copy(board_matrix)

  if not spot_is_open(board_matrix, move_tuple):
    raise Exception("Invalid move, cannot place stone on taken square")

  if current_player not in (-1,1):
    raise Exception("Invalid move, player must be -1 or 1")


  new_board = copy(board_matrix)
  set_value_for_spot(new_board, move_tuple, current_player)

  n_o_c = get_neighbors_of_color(new_board, move_tuple, -1*current_player)
  for n in n_o_c:
    if not (get_value_for_spot(new_board, n) == -1*current_player):
      continue
    libs = count_liberties_around_stone(new_board, n)
    if libs == 0:
      remove_group_around_stone(new_board, n)

  #   if libs == 0:
  # while len(n_o_c) != 0:
  #   n = n_o_c[0]
  #   libs = count_liberties_around_stone(new_board, n)
  #   if libs == 0:
  #     remove_group_around_stone(new_board, n)
  #   n_o_c = get_neighbors_of_color(new_board, move_tuple, -1*current_player)

  return new_board



def move_makes_duplicate(board_matrix, move_tuple, current_player, all_previous_boards):
  """
  this should happen after suicide. I won't call suicide in it though because they
  should always be called in tandem.
  BUT, it also happens if someone passes, which would preclude double-passing to end.
  """
  if move_tuple is None:
    return False
  if len(all_previous_boards) == 0:
    return False

  first = all_previous_boards[0]
  if len(first.shape) != 2:
    print all_previous_boards
    raise Exception("Calling move_makes_duplicate with the above as all_previous_boards")

  # I'm really relying on always having an accurate list of these. So, I can say:
  updated_board = update_board_from_move(board_matrix, move_tuple, current_player)

  # place_to_start == -1
  # if current_player == 1:
  #   place_to_start = 0
  # elif current_player == -1:
  #   place_to_start = 1
  # else:
  #   raise Exception("current_player is not -1 or 1, in move_makes_duplicate")

  # indices_to_compare = range(place_to_start, len(all_previous_boards), 2)
  # You always want to compare to the most recent previous, because that was YOU!
  all_indices = range(0, len(all_previous_boards))
  indices_to_compare = list(reversed(all_indices))[0::2]
  # print('current_player: ' + str(current_player) + '. list of indices: ')
  # print(indices_to_compare)
  for index in indices_to_compare:
    # print(index)
    board_at_index = all_previous_boards[index]
    if boards_are_equal(board_at_index, updated_board):
      return True
    else:
      continue
  # If it gets here, there were no duplicates.
  return False




def move_is_valid(board_matrix, move_tuple, current_player, all_previous_boards):
  """
  The three reasons it wouldn't be are that 
  a) it makes a duplicate position.
  b) it's a suicide move.
  c) there's already something there.
  """

  if all_previous_boards is None:
    raise Exception("You can pass a list to all_previous_boards, but not nothing!")

  if move_tuple is None:
    return True

  if not spot_is_open(board_matrix, move_tuple):
    return False

  if spot_is_suicide(board_matrix, move_tuple, current_player):
    return False

  # print "in move_is_valid"
  if move_makes_duplicate(board_matrix, move_tuple, current_player, all_previous_boards):
    return False

  # I think that's it.
  return True


# def valid_move_is_sensible(board_matrix, move_tuple, current_player, previous_board):
#   """
#   The only time its REALLY stupid to fill in a move is if it makes it so you only
#   have one eye. If it doesn't cause a capture, and the piece you
#   just put down only has one liberty. Then it's a bad move.

#   ONLY CALL AFTER spot_is_suicide and move_makes_duplicate.

#   update board.
#   check if it took any neighbors. If so, output True
#   check if the stone you just placed has only one liberty.
#   If so, output False. Otherwise, output True.
#   """
#   if move_tuple is None:
#     raise Exception("I don't think that anyone is going to call this function with\
#       none")

#   neighbors_on_board = get_neighbors_on_board(board_matrix, move_tuple)
#   neighbors_and_color = set([(spot, get_value_for_spot(board_matrix, spot)) for spot in neighbors_on_board])
#   updated_board = update_board_from_move(board_matrix, move_tuple, current_player)
#   new_neighbors_and_color = set([(spot, get_value_for_spot(updated_board, spot)) for spot in neighbors_on_board])
#   if neighbors_and_color != new_neighbors_and_color:
#     # This means something was taken, in which case it is an okay thing to do.
#     return True
#   else:
#     # This means nothing was taken.
#     liberties_around_stone = count_liberties_around_stone(updated_board, move_tuple)
#     if liberties_around_stone == 0:
#       raise Exception("Dead stone? This probably means that you didn't check for \
#         spot_is_suicide, which is a cardinal sin.")
#     elif liberties_around_stone == 1:
#       # You never want to make a move that makes you have no liberties.
#       return False
#     else:
#       return True







# def move_is_sensible_and_valid(board_matrix, move_tuple, current_player, previous_board):

#   pass



def output_one_valid_move(board_matrix, all_previous_boards, current_player):
  free_spaces = 0
  shape = board_matrix.shape
  # print shape
  l1 = range(shape[0])
  l2 = range(shape[1])
  random.shuffle(l1)
  random.shuffle(l2)

  for i in range(shape[0]):
    for j in range(shape[1]):
      if spot_is_open(board_matrix, (i,j)):
        free_spaces += 1
  if random.random() < (1.0/(free_spaces+1)):
    return None

  for i in l1:
    for j in l2:
      if move_is_valid(board_matrix, (i,j), current_player, all_previous_boards):
        return (i,j)
  # print "No Valid Moves. Huh."
  return None



  """
  First, with probability 1/free_spaces, output None.
  """



def output_all_valid_moves(board_matrix, all_previous_boards, current_player):
  """
  given a board and a previous board, retuns an array of all possible
  move-tuples.
  """
  valid_moves = set([])
  length, width = board_matrix.shape
  for m_t in move_tuples_on_board(board_matrix):
    if move_is_valid(board_matrix, m_t, current_player, all_previous_boards):
      valid_moves.add(m_t)

  valid_moves.add(None)
  return valid_moves

def output_valid_moves_boardmap(board_matrix, all_previous_boards, current_player):
  if board_matrix is None:
    raise Exception("I dont really know how to handle board_matrix being none in output_valid_moves_boardmap")
  valid_moves = output_all_valid_moves(board_matrix, all_previous_boards, current_player)
  boardmap = np.zeros(board_matrix.shape)
  for move in valid_moves:
    if move is None:
      continue
    else:
      set_value_for_spot(boardmap, move, 1)
  return boardmap

def output_valid_moves_mask(board_matrix, all_previous_boards, current_player):
  if board_matrix is None:
    raise Exception("I dont really know how to handle board_matrix being none in output_valid_moves_boardmap")
  valid_moves = output_all_valid_moves(board_matrix, all_previous_boards, current_player)
  shape = board_matrix.shape
  move_array = np.zeros(shape[0]*shape[1] +1, dtype=np.float32)

  for move in valid_moves:
    index = from_move_tuple_to_index(board_matrix, move)
    move_array[index] = 1.0
    
  return np.asarray(move_array)





def determine_owner_of_free_space(board_matrix, spot_tuple):
  """
  This is an interesting one. How this will work is, you take a free space and
  expand outwards, keeping track of the maximal border set. At the end,
  you evaluate the color of all of these border elements, and if it is monotone,
  you give it to that color. If not, you don't. Easy as pie, except for the writing
  part.

  Invariant: frontier is not part of border or seen.

  COULD BE VERY MUCH OPTIMIZED. It checks each individually, but it really just
  needs to do groupings, which would speed it up by like 360 times. Probably
  worth fixing.

  NOT DONE!
  """
  if not spot_is_open(board_matrix, spot_tuple):
    raise Exception("Non-empty spot passed to determine_owner_of_free_space. That doesn't make much sense.")
  border_spots = set([])
  seen_spots = set([])
  frontier = set([spot_tuple])

  while len(frontier) != 0:
    seen_spots = seen_spots.union(frontier)
    open_frontier = set([])
    for spot in frontier:
      if spot_is_open(board_matrix, spot):
        open_frontier.add(spot)
      else:
        border_spots.add(spot)

    new_frontier = set(flatten_list([get_neighbors_on_board(board_matrix, f_elem)
                      for f_elem in open_frontier]))
    new_frontier = new_frontier.difference(seen_spots)
    frontier = new_frontier

  if len(border_spots) == 0:
    # That means that there's nothing on the field.
    return 0

  color_groups = set([get_value_for_spot(board_matrix, b_s) for b_s in border_spots])

  if len(color_groups) == 0:
    raise Exception("Whaat? No colors in group?")
  elif 0 in color_groups:
    raise Exception("Somehow a blank space ended up in the border")
  elif len(color_groups) != 1:
    # That means it shared a border with both players.
    return 0
  else:
    color_elem = color_groups.pop()
    return color_elem





  # surrounded = group_is_one_color(board_matrix, border_spots)





def determine_owner_of_any_space(board_matrix, spot_tuple):
  if not spot_is_open(board_matrix, spot_tuple):
    return get_value_for_spot(board_matrix, spot_tuple)
  else:
    owner = determine_owner_of_free_space(board_matrix, spot_tuple)
    return owner






"""
I guess, it would make sense to have a function that checked the liberties of 
every taken square, to make sure none of them are zero. Because if they are, 
then that's sort of an illegal board. I think that this false eye thing is 
really screwing me up. How to check for it? I guess, the number of liberties 
that something has is the number of spots 
So, to score the board, first you have to go through and see all the things that
have only one eye. No, because as I said before, if you have something like that,
then the games will continue until they're full up.
"""

def output_score_map(current_board):
  board_copy = copy(current_board)
  score_map = np.zeros(current_board.shape)

  for tup in move_tuples_on_board(board_copy):
    owner = determine_owner_of_any_space(board_copy, tup)
    set_value_for_spot(score_map, tup, owner)
  return score_map



def score_board(current_board):
  """
  Looks like this one is going to be tough... It sounds like there's debate for
  when a group is considered captured. especially if you consider non-optimal play.
  I may have to hard code some rules for final-counting.

  A problem is that it doesn't really carry games out to their logical conclusion,
  so for example it would score the example game on wikipedia incorrectly.

  The solution is that these things shouldn't stop playing until the very end.
  Why stop early if you have infinite patience?
  """
  board_copy = copy(current_board)
  if board_is_empty(current_board):
    return {
    'pos' : 0,
    'neg' : 0
  }

  for tup in move_tuples_on_board(board_copy):
    if spot_is_open(board_copy, tup):
      owner = determine_owner_of_free_space(board_copy, tup)
      if owner == 0:
        continue
      else:
        set_value_for_spot(board_copy, tup, owner)


  num_neg = 0
  num_pos = 0

  for tup in move_tuples_on_board(board_copy):
    spot_value = get_value_for_spot(board_copy, tup)
    if spot_value == 0:
      continue
    elif spot_value == 1:
      num_pos += 1
    elif spot_value == -1:
      num_neg += 1
    else:
      print board_copy
      raise Exception("Looks like at the end of this game, there is something that is not 0,-1,or 1. Board printed above.")

  return {
    'pos' : num_pos,
    'neg' : num_neg
  }


def determine_winner(current_board, handicap=0.0):

  scores = score_board(current_board)
  score_difference = scores['pos'] - scores['neg'] - handicap
  if score_difference > 0:
    return 1.0
  elif score_difference < 0:
    return -1.0
  else:
    print "Warning, a board should never tie. But maybe it should"
    return 0.0



def generate_random_board(board_shape, total_moves):
  """
  Generates random board, starting with black's move.
  Returns the generated board, plus the person who gets to move next.
  """
  all_previous_boards = []
  # last_board = np.zeros(board_shape)
  current_board = np.zeros(board_shape)
  next_turn = 1
  for i in xrange(total_moves):
    valid_moves = output_all_valid_moves(current_board, all_previous_boards, next_turn)
    if len(valid_moves) == 0:
      print "Looks like we got stuck at some point, try simulating less far."
      return None, None
    valid_move = random.choice(list(valid_moves))
    new_board = update_board_from_move(current_board, valid_move, next_turn)
    # last_board = current_board
    all_previous_boards.append(current_board)
    current_board = new_board
    next_turn = next_turn * -1
  return current_board, next_turn


if __name__ == '__main__':

  random_board, start = generate_random_board((5,5),10)
  print random_board
  for i in range(10):
    print output_one_valid_move(random_board, [], start)
  # print output_one_valid_move(random_board, random_board, start)
  # print output_one_valid_move(random_board, random_board, start)
  # print output_one_valid_move(random_board, random_board, start)
  # print output_one_valid_move(random_board, random_board, start)

  # print output_one_valid_move(generate_random_board((5,5), None, 10))
  # print output_one_valid_move(np.zeros((2,2)), np.zeros((2,2)), 1)
  # print output_one_valid_move(np.zeros((2,2)), np.zeros((2,2)), 1)
  # print output_one_valid_move(np.zeros((2,2)), np.zeros((2,2)), 1)
  # print output_one_valid_move(np.zeros((2,2)), np.zeros((2,2)), 1)
  # print output_one_valid_move(np.zeros((2,2)), np.zeros((2,2)), 1)
  # print generate_random_board((9,9), 0)
#   b2 = generate_random_board((9,9), 25)
#   b3 = generate_random_board((9,9), 25)
#   print b1
#   print b2
#   print b3










  



