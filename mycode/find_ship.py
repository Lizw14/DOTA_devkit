#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import pylab
pylab.rcParams['figure.figsize'] = (20.0, 20.0)

example = DOTA('../DOTA/train_split')

imgids = example.getImgIds(catNms=['ship'])
imgids.sort()

with open('mycode/train_split_ship_imgList.txt', 'w') as f:
  for id in imgids:
    f.write(id+'\n')

print(len(imgids))
