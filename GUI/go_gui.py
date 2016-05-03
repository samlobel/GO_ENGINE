from Tkinter import *
import Tkinter


import numpy as np
import tensorflow as tf
from copy import deepcopy as copy

import sys
import os

sys.path.append(os.path.abspath('../.'))

from go_util import util

# print util.flatten_list([[1,2,3],[4,5,6],[7,8,9]])



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
    self.columns.set(13)
    self.column_dropdown = OptionMenu(self.frame, self.columns, 5, 9, 11, 13, 19)
    self.column_dropdown.pack(side='left')

    b = Button(master = self.frame, text="Reset Game", command=self.reset)
    b.pack(side='right')

    self.canvas = Canvas(master = self.root,
                            height = self.height, width = self.width,
                            background = 'tan')
    self.canvas.pack()


    self.bottomframe = Frame(self.root)
    self.bottomframe.pack(side='bottom')


    self.error_label_text = StringVar()
    self.error_label_text.set("beginning")

    self.error_label_text = Label(master=self.bottomframe, textvariable=self.error_label_text, fg="red", font=("Helvetica", 24))
    self.error_label_text.pack(side='left')

    self.turn_symbol = Label(master=self.bottomframe, text="")
    self.turn_symbol.pack(side='right')
    self.turn_label = Label(master=self.bottomframe, text="TURN: ")
    self.turn_label.pack(side='right')












    self.board_width = (self.width - 2*self.padding)

    self.color_map = {
      1 : 'black',
      -1 : 'white'
    }

    def handler(event, self=self):
      return self.intercept_board_click(event)

    self.canvas.bind('<Button-1>', handler)

    self.reset()





  def set_error_label(self, text):
    self.error_label_text.set(text)

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
      'black_player' : self.black_player.get()
    }

    self.previous_board = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)
    self.board_data = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)
    self.canvas.delete('all')

    self.draw_background()
    self.set_label_for_turn()


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
    print "board drawn."


    

  def switch_turn(self):
    self.turn = -1*self.turn
    self.set_label_for_turn()

  def clicked_at_location(self, c, r):
    if util.move_is_valid(self.board_data, (c,r), self.turn, self.previous_board):
      new_board = util.update_board_from_move(self.board_data, (c,r), self.turn)
      self.previous_board = self.board_data
      self.board_data = new_board
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
          self.clicked_at_location(c, r)
          return
    print "not intercepted"


  


    

  def run(self):
    self.root.mainloop()

a = Board()
a.run()

