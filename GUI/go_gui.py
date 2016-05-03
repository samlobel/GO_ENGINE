from Tkinter import *
import Tkinter


import numpy as np
import tensorflow as tf

import sys
import os

sys.path.append(os.path.abspath('../.'))

from go_util import util

# print util.flatten_list([[1,2,3],[4,5,6],[7,8,9]])



class Board:
  def __init__(self, shape=(13,13)):
    if shape[0] != shape[1]:
      raise Exception("Non-square board?!")
    self.width = 600
    self.height = self.width
    self.radius = 10
    self.padding = 60

    self.columns = shape[0]
    self.rows = shape[1]

    self.board_width = (self.width - 2*self.padding)

    self.board_data = np.zeros(shape, dtype=np.int)


    self.turn = 1
    self.moves = []


    self.root = Tk()
    
    self.canvas = Canvas(master = self.root,
                                  height = self.height, width = self.width,
                                  background = 'tan')
    self.canvas.pack()
    self.draw_background()
    

  def switch_turn(self):
    pass

  def draw_background(self):
    cols = self.columns
    for i in range(cols):
      variable = self.padding + (self.board_width * i / (cols - 1))
      self.canvas.create_line(self.padding, variable, self.width - self.padding, variable, width=3)
      self.canvas.create_line(variable, self.padding, variable, self.width - self.padding, width=3)

    self.centers = [
      [(self.padding + (self.board_width*r)/(cols-1),
       self.padding + (self.board_width*c)/(cols-1)) for c in cols] for r in cols
      ]

  def intercept_board_click(self, event):
    x = event.x
    y = event.y
    for c in xrange(self.columns):
      for r in xrange(self.columns):
        if abs(self.centers(c,r))
    

  def run(self):
    self.root.mainloop()

a = Board()
a.run()

