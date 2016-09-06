
class GoBot(object):
  
  def __init__(self):
    pass

  def get_value_for_move(self, board_data, current_player, previous_board):
    raise NotImplementedError("Must Subclass get_value_for_move!")

  def get_best_move(self, board_data,  previous_board, current_player, turn_number=None):
    raise NotImplementedError("Must Subclass return_best_move")



