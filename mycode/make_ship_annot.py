#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from DOTA import DOTA
import dota_utils as util
from PIL import Image
import pylab
import glob
pylab.rcParams['figure.figsize'] = (20.0, 20.0)

f = open('mycode/train_split_ship_imgList.txt', 'r')
lines = f.readlines()
f.close()

num_ships_all = 0
orig_annot_path = '../DOTA/train_split/labelTxt/'
target_annot_path = '../DOTA/train_split/ship_label/'
img_path = '../DOTA/train_split/images/'

for idx,line in enumerate(lines):
  line = line.strip()
  
#for idx,file in enumerate(glob.glob('../DOTA/train_split/labelTxt/*.txt')):
#  line = osp.basename(file)[:-4]
  print('Image ', idx, line)
#  img = Image.open(img_path+line+'.png')
  # size: w, h
#  size = img.size
  size = [1000, 600]
  annot_file = orig_annot_path+line+'.txt'
  f = open(annot_file, 'r')
  objs = f.readlines()
  f.close()  
 
  target_annot = open(target_annot_path+line+'.txt', 'w')
  target_annot.write(str(size[0])+' '+str(size[1])+'\n')
  ship_num = 0
  for obj in objs:
    obj_ = obj.strip().split()
    if len(obj_)<4:
      continue
    if obj_[8] == 'ship':
      ship_num = ship_num + 1
      target_annot.write(obj)
  target_annot.close()
  print('#ships: ', ship_num)
  num_ships_all = num_ships_all + ship_num

print('num all ships: ', num_ships_all)
print('num images: ', len(lines))
