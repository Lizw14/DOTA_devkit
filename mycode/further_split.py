# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys


import ImgSplit
import os
import os.path as osp
import glob
import shutil
import numpy as np
from PIL import Image
import random
import cv2

def mkdir(folder):
    if not osp.exists(folder):
        os.mkdir(folder)

def read_DOTA_gtbox_and_label(txt_path):

    """
    :param xml_path: the path of DOTA label txt file
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    f = open(txt_path, 'r')
    try:
        img_width, img_height = [int(x) for x in f.readline().strip().split()]
    except:
        img = Image.open(osp.dirname(osp.dirname(txt_path))+'/images/'+osp.basename(txt_path)[:-4]+'.png')
        img_width, img_height = img.size

    lines = f.readlines()
    f.close()

    num_objs = len(lines)
    boxes = []
    for idx, line in enumerate(lines):
        annot = line.strip().split()
        try:
            box =  [int(x) for x in annot[0:8]]
        except:
            box =  [round(float(x)) for x in annot[0:8]]
        box.append('555')
        boxes.append(box)
        
    gtbox_label = np.array(boxes, dtype=np.int32)  # [x1, y1. x2, y2, x3, y3, x4, y4, label]

    return img_height, img_width, gtbox_label

def further_split(source_path, target_split_path, is_copy=0, fold=2, f=None):
    
    split = ImgSplit.splitbase(source_path,  target_split_path+'/tmp', choosebestpoint=True)
    
    
    for idx, file in enumerate(glob.glob(source_path+'/labelTxt'+'/*.txt')):

        try:

            tmp_height, tmp_width, tmp_label = read_DOTA_gtbox_and_label(file)
            if tmp_label.shape[0]>50:
                split.subsize_w = round(tmp_width/2+10)
                split.slide_w = round(tmp_width/2+2)
                split.subsize_h = round(tmp_height/2+10)
                split.slide_h = round(tmp_height/2+2)
                split.SplitSingle(osp.basename(file)[:-4], 1, '.png')
                if is_copy==0:
                    os.remove(file)
                    os.remove(source_path+'/images/'+osp.basename(file)[:-4]+'.png')
#                else:
#                    if f is not None:
#                        f.write(file+'\n')
            elif is_copy==1 and tmp_label.shape[0]>0:
#                shutil.copyfile(file,target_split_path+'/labelTxt/'+osp.basename(file))
#                shutil.copyfile(source_path+'/images/'+osp.basename(file)[:-4]+'.png',target_split_path+'/images/'+osp.basename(file)[:-4]+'.png')
                f.write(file+'\n')
            elif is_copy==0 and tmp_label.shape[0]>0:
                shutil.move(file,target_split_path+'/labelTxt/'+osp.basename(file))
                shutil.move(source_path+'/images/'+osp.basename(file)[:-4]+'.png',target_split_path+'/images/'+osp.basename(file)[:-4]+'.png')
            elif is_copy==0 and tmp_label.shape[0]<1:
                os.remove(file)
                os.remove(source_path+'/images/'+osp.basename(file)[:-4]+'.png')
            
            print(str(fold)+'\t'+str(tmp_label.shape[0])+'\t'+osp.basename(file))
        except:
            print('ERROR:', file)
            if is_copy==0:
                if osp.exists(file):
                    os.remove(file)
                if osp.exists(source_path+'/images/'+osp.basename(file)[:-4]+'.png'):
                    os.remove(source_path+'/images/'+osp.basename(file)[:-4]+'.png')



    if len(glob.glob(target_split_path+'/tmp/labelTxt/*.txt'))>0:
        further_split(target_split_path+'/tmp', target_split_path, is_copy=0, fold=fold+1)

def pad2size(source_path, target_path, width, height, rate=1):
    for file in glob.glob(source_path+'/labelTxt/*.txt'):
        img = cv2.imread(source_path+'/images/'+osp.basename(file)[:-4]+'.png')
        if (rate != 1):
            img = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        #600 1000 3
        img_height, img_width, tmp = img.shape
        left = random.randint(0, width-img_width)
        right = width-img_width-left
        top = random.randint(0, height-img_height)
        bottom = height-img_height-top
        img_ = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        cv2.imwrite(target_split_path+'/images/'+osp.basename(file)[:-4]+'.png', img_)
        in_file = open(file, 'r')
        lines = in_file.readlines()
        in_file.close()
        out_file = open(target_split_path+'/labelTxt/'+osp.basename(file), 'w')
        for line in lines:
            line = line.strip().split()
            for i in range(4):
                try:
                    line[i*2] = str(int(line[2*i])*rate + left)
                    line[i*2+1] = str(int(line[2*i+1])*rate + top)
                except:
                    line[i*2] = str(round(float(line[2*i])*rate) + left)
                    line[i*2+1] = str(round(float(line[2*i+1])*rate) + top)
            line = ' '.join(line)+'\n'
            out_file.write(line)
        out_file.close()

if __name__ == '__main__':
    #txt_path = '/mnt/lustre/thesis/DOTA/train_split/ship_label/P2792__1__900___500.txt'
    #a,b,c = read_DOTA_gtbox_and_label(txt_path)
    #print(a,b,c)

    target_split_path = '../DOTA/train_split/further_50_noPad'
    source_path = '../DOTA/train_split'
    mkdir(target_split_path)
    mkdir(target_split_path+'/images')
    mkdir(target_split_path+'/labelTxt')
    mkdir(target_split_path+'/tmp')
    mkdir(target_split_path+'/tmp/images')
    mkdir(target_split_path+'/tmp/labelTxt')
    f = open('../DOTA/train_split/further_50/no_need_to_split_list.txt', 'w')
    further_split(source_path, target_split_path, 1, f=f)
#    pad2size(target_split_path, target_split_path, width=1000, height=600, rate=1)
    f.close()
#    convert_pascal_to_tfrecord()
