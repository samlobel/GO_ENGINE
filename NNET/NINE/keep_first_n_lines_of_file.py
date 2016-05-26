def keep_n(filename, n):
  temp_name = filename + '_temp'
  with open(filename, 'r') as f:
    with open(temp_name, 'w') as f_t:
      for i in xrange(n):
        l = f.readline()
        f_t.write(l)
  print "all done!"



if __name__ == '__main__':
  keep_n('./random_boards.txt', 150000)


