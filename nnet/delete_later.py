"""
Git should make this unnecessary, but I'm a bozo so...
"""

def OLD_gather_all_possible_results(self, board_input, previous_board, current_turn):
  """
  Shit, is this wrong too? I should switch it beforehand, just to be safe.
  
  To judge a board, I ALWAYS need to make sure it's white's turn.
  So I make the initial one BLACK's turn, then simulate one move in the future,
  and then 

  BIG CHANGE! I'M GOING TO SWITCH TO DOING RANDOM SIMULATIONS AFTER
  I MAKE A DETERMINISTIC MOVE. THAT WAY I CAN BETTER EXPLORE THE STATE SPACE.


  """
  print("gathering results")

  if current_turn == -1:
    board_input = -1 * board_input
  if (current_turn == -1) and (previous_board is not None):
    previous_board = -1 * previous_board

  valid_moves = list(util.output_all_valid_moves(board_input, previous_board, 1))

  resulting_boards = [util.update_board_from_move(board_input, move, 1) for move in valid_moves]
  zipped = zip(valid_moves, resulting_boards)
  # end_results_of_boards = [self.get_results_of_board_on_policy(new_board, board_input, -1, [move])
  #           for (move, new_board) in zipped] #this is -1, because it is next turn.

  


  end_results_of_boards = [self.get_result_of_board_from_random_policy(new_board, board_input, -1, [move])
            for (move, new_board) in zipped] #this is -1, because it is next turn.          
  


  
  print("all boards simulated")

  boards_before_and_after = zip(resulting_boards, end_results_of_boards)
  # Filter it to get rid of nones.
  boards_before_and_after = [(before, after) for (before, after) in boards_before_and_after
                                if (before is not None) and (after is not None)]

  boards_before = [before for (before, after) in boards_before_and_after] #filtered
  boards_after = [after for (before, after) in boards_before_and_after] #filtered

  true_value_of_boards = [util.determine_winner(board) for board in boards_after]
  # returning true values with all moves one after input.
  return zip(true_value_of_boards, boards_before)


  # values_and_boards = [(util.determine_winner(after), before)]                              
  

  # true_value_of_boards = [util.determine_winner(board) for board in end_results_of_boards]
  # return zipped(true_value_of_boards, resulting_boards)





  return

  # valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
  # resulting_boards = [util.update_board_from_move(board_matrix, move, current_turn) for move in valid_moves]
  # zipped = zip(valid_moves, resulting_boards)
  # end_results_of_boards = [self.get_results_of_board_on_policy(new_board, board_input, (-1*current_turn), [move])
  #           for (move, new_board) in zipped]
  # true_value_of_boards = [util.determine_winner(board) for board in end_results_of_boards]
  # print("results gathered")
  # return zipped(true_value_of_boards, resulting_boards)



































def get_result_of_board_from_random_policy(self, board_matrix, previous_board, current_turn, move_list=None):
  if move_list is None and util.boards_are_equal(board_matrix, previous_board):
    move_list = [None]
  if move_list is None:
    move_list=[]

  while not ((len(move_list) >= 2) and (move_list[-1] is None) and (move_list[-2] is None)):
    # print ("move: " + str(len(move_list)))
    if len(move_list) > 300:
      print("simulation lasted more than 300 moves")
      return None
    # valid_moves = list(util.output_all_valid_moves(board_matrix, previous_board, current_turn))
    valid_move = util.output_one_valid_move(board_matrix, previous_board, current_turn)
    # best_move = self.get_best_move(board_matrix, previous_board, current_turn)
    if valid_move is None:
      new_board = copy(board_matrix)
    else:
      new_board = util.update_board_from_move(board_matrix, valid_move, current_turn)
    move_list.append(valid_move)
    previous_board = board_matrix
    board_matrix = new_board
    current_turn *= -1

  return copy(board_matrix)























  
