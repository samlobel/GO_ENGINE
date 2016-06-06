import os
this_dir = os.path.dirname(os.path.realpath(__file__))


def make_path_from_folder_and_batch_num(folder_name, batch_num):
  save_path = os.path.join(this_dir + '/saved_models/convnet_pol_val_good', str(folder_name))
  file_name = 'trained_on_batch_' + str(batch_num) + '.ckpt'
  save_path = os.path.join(save_path, file_name)
  return save_path

def get_largest_batch_in_folder(f_name):
  folder = os.path.join(this_dir, 'saved_models', 'convnet_pol_val_good')
  filename = os.path.join(folder, f_name,'largest.txt')
  f = open(filename, 'r')
  content = f.read()
  content = content.strip()
  latest = int(content)
  f.close()
  return latest






def delete_from_folder(folder_name, denominator, MAX):

  largest = get_largest_batch_in_folder(folder_name)

  num_deleted = 0

  i = 0
  num_seen = 0
  while True:
    i += 1
    if i == largest:
      continue
    if i >= MAX:
      print "at max."
      break
    filename = make_path_from_folder_and_batch_num(folder_name, i)
    meta_filename = filename + '.meta'
    if os.path.exists(filename):
      num_seen += 1
      if (num_seen) % denominator == 0:
        print "deleting " + str(filename)
        os.remove(filename)
        os.remove(meta_filename)
        print "deleted."
        num_deleted += 1
  
  print "all done."
  print "num_deleted: " + str(num_deleted)





    







if __name__ == '__main__':
  print "main. deleting 3 out of 4 models."
  # for folder in ['1','2','3','4','5']:
    # print "deleting from folder " + str(folder)
    # delete_from_folder(folder, 2, 5000)
    # delete_from_folder(folder, 2, 5000)
  delete_from_folder('test', 2, 5000)
  print "after"
  # print make_path_from_folder_and_batch_num("1","1")
  # print make_path_from_folder_and_batch_num("5","2")