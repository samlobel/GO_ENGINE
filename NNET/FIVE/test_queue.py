from Queue import Queue
import time
from threading import Thread

Q_in = Queue(maxsize=25)
Q_out = Queue(maxsize=25)

def queue_loader():
  for i in xrange(1000):
    Q_in.puts(i)
    # print 'put ' + str(i) + ' in queue'



def worker_in():
  """
  Takes from q_in, negates, puts in q_out
  """
  while True:
    try:
      item = Q_in.get(block=True, timeout=10)
      neg = -1 * item
      Q_out.put(neg)
      time.sleep(0.1)
      Q_in.task_done()
    except Exception:
      print 'worker done.'

def worker_out():
  while True:
    try:
      item = Q_out.get(block=True, timeout=10)
      print "TAKEN: " + str(item)
      time.sleep(0.1)
      Q_out.task_done()
    except Exception:
      print 'worker done'

NUM_WORKER_IN = 5
NUM_WORKER_OUT = 1
if __name__ == '__main__':
  t_loader = Thread(target=queue_loader)
  t_loader.daemon = True
  t_loader.start()
  for i in range(NUM_WORKER_IN):
    t_in = Thread(target=worker_in)
    t_in.daemon = True
    t_in.start()
  for i in range(NUM_WORKER_OUT):
    t_out = Thread(target=worker_out)
    t_out.daemon = True
    t_out.start()
  print 'all running now.'

  print 'running'
  Q_in.join()
  Q_out.join()
  print 'ran'
  




  


