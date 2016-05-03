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
  def __init__(self):
    # if shape[0] != shape[1]:
    #   raise Exception("Non-square board?!")
    self.width = 600
    self.height = self.width
    self.padding = 60

    self.root = Tk()
    self.frame = Frame(self.root)
    self.frame.pack()
    self.columns = IntVar(self.root)
    self.columns.set(13)
    self.column_dropdown = OptionMenu(self.frame, self.columns, 5, 9, 11, 13, 19)
    self.column_dropdown.pack(side='left')

    b = Button(master = self.frame, text="Reset Game", command=self.reset)
    b.pack(side='right')

    # self.columns = shape[0]
    # self.rows = shape[1]

    self.board_width = (self.width - 2*self.padding)

    self.line_gap = self.board_width / (self.columns.get() - 1)
    self.radius = self.line_gap / 2.2
    # if line_gap < self.radius*2:
    #   raise Exception("Overlapping circles!")

    self.board_data = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)

    self.color_map = {
      1 : 'black',
      -1 : 'white'
    }

    self.turn = 1
    self.moves = []


    
    
    self.canvas = Canvas(master = self.root,
                                  height = self.height, width = self.width,
                                  background = 'tan')
    self.canvas.pack()
    self.draw_background()

    def handler(event, self=self):
      return self.intercept_board_click(event)

    self.canvas.bind('<Button-1>', handler)

    def reset(self):
      self.line_gap = self.board_width / (self.columns.get() - 1)
      self.radius = self.line_gap / 2.2
      self.turn = 1
      self.moves = []
      self.board_data = np.zeros((self.columns.get(), self.columns.get()), dtype=np.int)
      self.canvas.delete('all')
      self.draw_background()


  def draw_circle_at_location(self, c, r):
    inter_x, inter_y = self.centers[c][r]
    rad = self.radius
    self.canvas.create_oval(inter_x-rad, inter_y-rad, inter_x+rad, inter_y+rad, fill=self.color_map[self.turn])
    

  def switch_turn(self):
    pass




  def draw_background(self):
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
          self.draw_circle_at_location(c,r)
          self.turn = -1*self.turn
          return
    print "not intercepted"


  


    

  def run(self):
    self.root.mainloop()

a = Board()
a.run()

