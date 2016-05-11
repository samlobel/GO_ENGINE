a = 5

def change_a():
  global a
  test_change()
  print a
  a -= 1
  print a
  test_change()



def test_change():
  global a
  print a


change_a()
