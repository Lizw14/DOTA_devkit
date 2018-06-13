
# coding: utf-8

# # Usage for parse ground truth file and show
# This is a demo for parse, visulize and split the data, the demo use the data in example folder,
# but the folder does not neccessaryly include all the categories.

# In[ ]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import pylab
pylab.rcParams['figure.figsize'] = (20.0, 20.0)

from ImgSplit import splitbase

# find images with ship
example = DOTA('../DOTA/train')
imgids = example.getImgIds(catNms=['ship'])
imgids.sort()

# build split database
split = splitbase(r'../DOTA/train', 
                 r'../DOTA/train_split', choosebestpoint=True)

# split images containing ships
f = open('../DOTA/train/train_ship_imgList.txt', 'w')
for id in imgids:
    print(id)
    f.write(id+'\n')
    split.SplitSingle(id, 1, '.png')
f.close()

print(len(imgids))

