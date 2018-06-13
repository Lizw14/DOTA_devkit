
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


# In[ ]:

dota = DOTA('/Users/lizw/Documents/thesis/DOTA/train')


# In[31]:

cat=['ship']
imgids = dota.getImgIds(catNms=cat)
print(len(imgids))

for imgid in imgids[1:50]:
#    imgid = imgids[0]
    img = dota.loadImgs(imgid)[0]
    
    
    # In[33]:
    plt.figure()    
    plt.axis('off')
    
#    plt.imshow(img)
    #plt.show()
    
    
    # In[34]:
    
    anns = dota.loadAnns(imgId=imgid, catNms=cat)
    # print(anns)
    dota.showAnns(anns, imgid, 2)
    plt.savefig('mycode/images/'+str(imgid)+'_.png')

