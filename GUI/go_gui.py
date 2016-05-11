from Tkinter import *
import Tkinter


import numpy as np
import tensorflow as tf
from copy import deepcopy as copy

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.'))

# sys.path.append(os.path.abspath('../.'))

from go_util import util

# from NNET.NINE.more_basic_convnet import Very_Basic_ConvBot
# from NNET.NINE.basic_convnet import Basic_ConvBot as Convbot
# from NNET.NINE.better_learning_convnet import Better_Learner_ConvBot
from NNET.FIVE.basic_convnet import Convbot_FIVE
# from NNET.FIVE.convnet_with_features import Convbot_FIVE_FEATURES
from NNET.FIVE.convnet_with_policy import Convbot_FIVE_POLICY
from NNET.FIVE.convnet_policy_features import Convbot_FIVE_POLICY_FEATURES

from NNET.NINE.random_mover import Random_Mover

import time

# print util.flatten_list([[1,2,3],[4,5,6],[7,8,9]])


# AI_1 = Convbot()
# AI_2 = Random_Mover()


class Board:
  def __init__(self):
    # if shape[0] != shape[1]:
    #   raise Exception("Non-square board?!")
    self.width = 600
    self.height = self.width
    self.padding = 60

    self.root = Tk()
    self.frame = Frame(self.root)
    self.frame.pack()

    l1 = Label(self.frame, text="Black is:")
    l1.pack(side='left')
    self.black_player = StringVar(self.root)
    self.black_player.set("HUMAN")
    self.black_player_dropdown = OptionMenu(self.frame, self.black_player, "HUMAN", "AI")
    self.black_player_dropdown.pack(side='left')

    l2 = Label(self.frame, text="White is:")    
    l2.pack(side='left')
    self.white_player = StringVar(self.root)
    self.white_player.set("HUMAN")
    self.white_player_dropdown = OptionMenu(self.frame, self.white_player, "HUMAN", "AI")
    self.white_player_dropdown.pack(side='left')

    l3 = Label(self.frame, text="Board dims:")
    l3.pack(side='left')
    self.columns = IntVar(self.root)
    self.columns.set(5)
    self.column_dropdown = OptionMenu(self.frame, self.columns, 5, 9, 11, 13, 19)
    self.column_dropdown.pack(side='left')

    b = Button(master = self.frame, text="Reset Game", command=self.reset)
    b.pack(side='right')

    self.canvas = Canvas(master = self.root,
                            height = self.height, width = self.width,
                            background = 'tan')
    self.canvas.pack()


    self.lowestTwo = Frame(self.root)


    b_2 = Button(master = self.lowestTwo, text="Pass Move", command=self.pass_move)
    b_2.pack(side='right')

    self.bottomframe = Frame(self.lowestTwo)
    self.bottomframe.pack(side='top')


    self.error_text = StringVar()
    self.error_text.set("")

    self.error_label_text = Label(master=self.bottomframe, textvariable=self.error_text, fg="red", font=("Helvetica", 24))
    self.error_label_text.pack(side='left')

    self.turn_symbol = Label(master=self.bottomframe, text="")
    self.turn_symbol.pack(side='right')
    self.turn_label = Label(master=self.bottomframe, text="TURN: ")
    self.turn_label.pack(side='right')








    self.lowestframe = Frame(self.lowestTwo)
    self.lowestframe.pack(side='bottom')


    self.move_count = StringVar()
    self.move_count.set(str(0))
    self.move_count_label = Label(master=self.lowestframe, textvariable=self.move_count)
    self.move_count_label.pack(side='right')
    l_bl_sc = Label(master=self.lowestframe, text="Move Count: ")
    l_bl_sc.pack(side='right')




    self.black_score = StringVar()
    self.black_score.set(str(0))
    self.black_score_label = Label(master=self.lowestframe, textvariable=self.black_score)
    self.black_score_label.pack(side='right')
    l_bl_sc = Label(master=self.lowestframe, text="Black Score: ")
    l_bl_sc.pack(side='right')


    self.white_score = StringVar()
    self.white_score.set(str(0))
    self.white_score_label = Label(master=self.lowestframe, textvariable=self.white_score)
    self.white_score_label.pack(side='right')
    l_bl_sc = Label(master=self.lowestframe, text="White Score: ")
    l_bl_sc.pack(side='right')


    

    self.lowestTwo.pack(side='bottom', fill=X)















    self.board_width = (self.width - 2*self.padding)

    self.color_map = {
      1 : 'black',
      -1 : 'white'
    }

    self.inverse_color_map = {
      'black' : 1,
      'white': -1
    }

    def handler(event, self=self):
      return self.intercept_board_click(event)

    self.canvas.bind('<Button-1>', handler)

    self.disabled = False

    self.reset()

  def pass_move(self):
    self.clicked_at_location(None)
    return





  def set_error_label(self, text):
    self.error_text.set(text)

  def set_scores(self, white, black):
    self.black_score.set(str(black))
    self.white_score.set(str(white))
    self.move_count.set(str(len(self.moves)))

  def set_label_for_turn(self):
    if self.turn == 1:
      self.turn_symbol.config(text="BLACK", fg="white", bg="black")
    elif self.turn == -1:
      self.turn_symbol.config(text="WHITE", fg="black", bg="white")
    else:
      raise Exception("turn should always be 1 or -1")



  def reset(self):
    self.line_gap = self.board_width / (self.columns.get() - 1)
    self.radius = self.line_gap / 2.2
    self.turn = 1
    self.moves = []
    self.board_metadata = {
      'shape' : (self.columns.get(), self.columns.get()),
      'white_player' : self.white_player.get(),
      'black_player' : self.black_player.get(),
      'white_AI' : None,
      'black_AI' : None
    }
    self.set_scores(0,0)

    if self.board_metadata['black_player'] == 'AI':
      # self.board_metadata['black_AI'] = Convbot(load_path="../NNET/NINE/saved_models/basic_convnet/trained_on_1_batch.ckpt")
      # self.board_metadata['black_AI'] = Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_91_batch.ckpt")
      self.board_metadata['black_AI'] = Convbot_FIVE_POLICY_FEATURES(load_path="../NNET/FIVE/saved_models/convnet_feat_pol/trained_on_139_batch.ckpt")
      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      
      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_39_batch.ckpt")
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_10_batch.ckpt")
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_61_batch.ckpt")
      # 
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_61_batch.ckpt")
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_10_batch.ckpt")
      # Convbot_FIVE()
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_25_batch.ckpt")

      # 

      # 
      
      # Convbot_FIVE_FEATURES()
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_99_batch.ckpt")
      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_91_batch.ckpt")
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_91_batch.ckpt")
      # Better_Learner_ConvBot('../NNET/NINE/saved_models/better_learning_convnet/trained_on_4_batch.ckpt')
      # Better_Learner_ConvBot()
          

      # Very_Basic_ConvBot('../NNET/NINE/saved_models/more_basic_convnet/trained_on_1_batch.ckpt')
      
      # self.board_metadata['black_AI'] = Very_Basic_ConvBot('../NNET/NINE/saved_models/more_basic_convnet/trained_on_1_batch.ckpt')
      
    if self.board_metadata['white_player'] == 'AI':
      # self.board_metadata['white_AI'] = Random_Mover(shape=(self.columns.get(),self.columns.get()))
      self.board_metadata['white_AI'] = Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # Convbot_FIVE_POLICY_FEATURES(load_path="../NNET/FIVE/saved_models/convnet_feat_pol/trained_on_19_batch.ckpt")
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_440_batch.ckpt")
       # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_440_batch.ckpt")
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_10_batch.ckpt")
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_10_batch.ckpt")
      # Convbot_FIVE_POLICY(load_path="../NNET/FIVE/saved_models/convnet_with_policy/trained_on_50_batch.ckpt")

      # Convbot_FIVE_POLICY()

      # 
      
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_100_batch.ckpt")
      # Convbot_FIVE_FEATURES()
      # Convbot_FIVE(load_path="../NNET/FIVE/saved_models/basic_convnet/trained_on_200_batch.ckpt")
      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # Better_Learner_ConvBot('../NNET/NINE/saved_models/better_learning_convnet/trained_on_1_batch.ckpt')
      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # self.board_metadata['white_AI'] = Very_Basic_ConvBot('../NNET/NINE/saved_models/more_basic_convnet/trained_on_168_batch.ckpt')

      # Random_Mover(shape=(self.columns.get(),self.columns.get()))
      # Very_Basic_ConvBot('../NNET/NINE/saved_models/more_basic_convnet/trained_on_39_batch.ckpt')
      # self.board_metadata['white_AI'] = Convbot(load_path="../NNET/NINE/saved_models/basic_convnet/trained_on_7_batch.ckpt")
      # self.board_metadata['white_AI'] = Convbot(load_path="../NNET/NINE/saved_models/basic_convnet/trained_on_5_batch.ckpt")
      # self.board_metadata['white_AI'] = Very_Basic_ConvBot(load_path="../NNET/NINE/saved_models/more_basic_convnet/trained_on_10_batch.ckpt")
      # self.board_metadata['white_AI'] = Very_Basic_ConvBot()
      # self.board_metadata['white_AI'] = Very_Basic_ConvBot('../NNET/NINE/saved_models/more_basic_convnet/trained_on_1_batch.ckpt')
      


    self.previous_board = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)
    self.board_data = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)
    self.canvas.delete('all')

    self.draw_background()
    self.set_label_for_turn()
    self.disabled = False
    self.move_with_ai_if_applicable()


  def draw_circle_at_location(self, c, r, color):
    inter_x, inter_y = self.centers[c][r]
    rad = self.radius
    self.canvas.create_oval(inter_x-rad, inter_y-rad, inter_x+rad, inter_y+rad, fill=self.color_map[color])

  def draw_board_data(self):
    self.draw_background()
    for tup in util.move_tuples_on_board(self.board_data):
      if util.spot_is_color(self.board_data, tup, -1):
        self.draw_circle_at_location(tup[0],tup[1], -1)
      elif util.spot_is_color(self.board_data, tup, 1):
        self.draw_circle_at_location(tup[0],tup[1], 1)
      else:
        continue
    scores = util.score_board(self.board_data)
    self.set_scores(scores['neg'], scores['pos'])
    print "board drawn."
    self.root.update()


  def move_with_ai_if_applicable(self):
    turn = self.turn
    the_ai = None
    if turn == 1:
      the_ai = self.board_metadata['black_AI']
    elif turn == -1:
      the_ai = self.board_metadata['white_AI']
    else:
      raise Exception("Turn is not 1 or -1")

    if the_ai is None:
      print "no AI for player " + str(turn)
      return

    time.sleep(0.1)
    best_move = the_ai.get_best_move(self.board_data, self.previous_board, self.turn)
    print best_move
    self.clicked_at_location(best_move)

    


  def disable_if_done(self):
    if (len(self.moves) >= 2) and (self.moves[-1] is None) and (self.moves[-2] is None):
      self.disabled = True

  def switch_turn(self):
    self.turn = -1*self.turn
    self.set_label_for_turn()
    self.disable_if_done()
    self.move_with_ai_if_applicable()

  def clicked_at_location(self, click_location):
    if self.disabled:
      print "disabled, cannot click"
      return
    if util.move_is_valid(self.board_data, click_location, self.turn, self.previous_board):
      print "clicking for turn: " + str(self.color_map[self.turn]) + "\n\n\n"
      new_board = util.update_board_from_move(self.board_data, click_location, self.turn)
      self.previous_board = self.board_data
      self.board_data = new_board
      self.moves.append(click_location)
      print self.board_data
      
      self.draw_board_data()
      self.switch_turn()

      
    # self.draw_circle_at_location(c, r, 'black')





  def draw_background(self):
    self.canvas.delete('all')
    cols = self.columns.get()
    for i in range(cols):
      variable = self.padding + (self.board_width * i / (cols - 1))
      self.canvas.create_line(self.padding, variable, self.width - self.padding, variable, width=3)
      self.canvas.create_line(variable, self.padding, variable, self.width - self.padding, width=3)

    self.centers = [
      [(self.padding + (self.board_width*r)/(cols-1),
       self.padding + (self.board_width*c)/(cols-1)) for c in range(cols)] for r in range(cols)
      ]

  def intercept_board_click(self, event):
    print "clicked!"
    x = event.x
    y = event.y
    print (x,y)
    for c in xrange(self.columns.get()):
      for r in xrange(self.columns.get()):
        inter_x, inter_y = self.centers[c][r]
        if (abs(inter_x - x) < self.radius) and (abs(inter_y - y) < self.radius):
          print "hit intersection " + str(inter_x) + ", " + str(inter_y)
          # self.draw_circle_at_location(c,r)
          # self.turn = -1*self.turn
          self.clicked_at_location((c, r))
          return
    print "not intercepted"


  


    

  def run(self):
    self.root.mainloop()

a = Board()
a.run()

