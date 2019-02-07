import sys
sys.path.append("fake_texts/") 
#sys.path.append("/home/leander/AI/repos/gen_text/TextRecognitionDataGenerator/fonts") 
import argparse
import os, errno
import random
import string
from tqdm import tqdm

from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)

from data_generator import FakeTextDataGenerator

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug_transformations import augmentations
from skimage.transform import resize
import cv2



def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict


def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """

    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    else:
        return [os.path.join("/home/leander/AI/repos/gen_text/TextRecognitionDataGenerator/fonts/latin/", font) for font in os.listdir("/home/leander/AI/repos/gen_text/TextRecognitionDataGenerator/fonts/latin/")]


import numpy as np
from nltk import word_tokenize
import torch

import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,batch_size,epoch_size=10,random_strings=True,num_words=5,transform=False,width=-1,alignment=1,height=32):
        'Initialization'
        #General args:
        self.transform=transform
        self.random_strings=random_strings
        self.num_words=num_words
        #How much data we want to generate in a single epoch 
        self.epoch_size=epoch_size
        self.batch_size=batch_size

        def take_10(x):

            if len(x)>self.num_words:
                return x[0:self.num_words]
            else:
                return x 

        #Text gen specific stuff 
        self.thread_count=6
        self.language="en"
        self.count=32
        #If we want to use random seq, alternatively we use Wikipedia ( needs internet)
        self.random_sequences=False
        self.include_letters=True
        self.include_numbers=True
        self.include_symbols=True
        #When source wikipedia, how many words we want to include
        self.length=10
        #If we want to have variable word lengths, "length is maximum)
        self.random=True
        #The height of the image
        self.format=height
        #Skeqing angle
        self.skew_angle=0
        self.random_skew=True
        self.use_wikipedia=True
        self.blur=0
        self.random_blur=True

        """Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures""",
        self.background=1

        """Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random""",

        self.distorsion=0
        """Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both"""
        self.distorsion_orientation=0
        self.width=width
        self.orientation=0
        self.text_color='#282828'
        self.space_width=0.1
        self.extension="jpg"
        self.handwritten=False
        self.name_format=0
        self.alignment=alignment

        #This shoule be done on init We also do string generation in the init of the dataset
        pool = ''
        pool += string.ascii_letters
        pool += "0123456789"
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
        pool += ' '
        self.keys = list(pool)
        self.values = np.array(range(1,len(pool)+1))
        self.dictionary = dict(zip(self.keys, self.values))
        self.fonts = load_fonts("en")
        
        self.decode_dict=dict((v,k) for k,v in self.dictionary.items())
        self.decode_dict.update({93 : "OOK"})

        ###Get strings
        strings = []
        #We try to load strings from wikipedia, if not we return random strings. 
        if self.random_strings==False:
            try:
                #Num words just doesnt work here I think it takes one sentence always
                strings = create_strings_from_wikipedia(1, self.batch_size*self.epoch_size, "en")
            except:
                print("Connection issues")
                strings = create_strings_randomly(self.num_words, False, batch_size*epoch_size,
                                          True, True,True, "en")
        else:
            strings = create_strings_randomly(self.num_words, False, batch_size*epoch_size,
                          True, True,True, "en")

        ###Get Images 
        #Here we actually take up to n words, by word tokenizing and then taking n 
        strings =[" ".join(take_10(word_tokenize(x))) for x in strings]
        #Next we split into cahracter
        strings_=[list(x)for x in strings]
        #self.strings=strings
        #self.strings_=strings_
        #Then we convert to interger, 93 for symbols we dont know
        self.strings_int=[[self.dictionary[x] if x in self.keys else 93 for x in m ] for m in strings_]
        #Then we get the lengths, we need for loss
        self.strings_len=[len(x)for x in self.strings_int]
        string_count = len(strings)
        #We can write it in a neat list comprehension, enough optimization for me haha 
        self.image_list=[np.expand_dims(np.array(FakeTextDataGenerator.generate(*j)),0) for j in zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [self.fonts[random.randrange(0, len(self.fonts))] for _ in range(0, string_count)],
                    [self.format] * string_count,
                    [self.extension] * string_count,
                    [self.skew_angle] * string_count,
                    [self.random_skew] * string_count,
                    [self.blur] * string_count,
                    [self.random_blur] * string_count,
                    [self.background] * string_count,
                    [self.distorsion] * string_count,
                    [self.distorsion_orientation] * string_count,
                    [self.handwritten] * string_count,
                    [self.name_format] * string_count,
                    [self.width] * string_count,
                    [self.alignment] * string_count,
                    [self.text_color] * string_count,
                    [self.orientation] * string_count,
                    [self.space_width] * string_count )]

        
        self.seq = augmentations

    def __len__(self):
        'Denotes the total number of samples'
        return self.batch_size*self.epoch_size
    def transform(self,img):
        return self.seq.augment_images(img)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X=self.image_list[index]#/255
        y=self.strings_int[index]
        y_len=len(y)
        
        
        if self.transform== True: 
            X=self.seq.augment_images(X)
        X=np.squeeze(X)  
        #X=cv2.cvtColor(X,cv2.COLOR_RGB2GRAY)
        #X=cv2.cvtColor(X,cv2.COLOR_GRAY2RGB)
        
    #    resize_shape=(12,int(12*X.shape[1]/X.shape[0]))
   #     X = resize(X,resize_shape,mode="constant") 
#
#        resize_shape=(32,int(32*X.shape[1]/X.shape[0]))
 #       X = resize(X,resize_shape,mode="constant")     
  #      
        X=np.expand_dims(X,0) 
        #X =X/255
        x_len=X.shape[2]

        return X, y,x_len,y_len
    
# a simple custom collate function, just to show the idea
#Basically the collate gets calleda list of the putputs of all the batches

def my_collate(batch):
    #some shapes 
    one,height,wi,channels=batch[0][0].shape
    print(wi)
    #batch size
    batch_size=len(batch)
    #get hte max witdth for padding 
    widths=np.array([x[2] for x in batch])
    max_width=np.max(widths)
    #get label in a long array 
    label_stack=np.concatenate([x[1] for x in batch])
    #get all the lengths 
    length_stack_y=np.array([x[3] for x in batch])
    #Here we will inster images, aka we pad them 
    img_stack=np.zeros(shape=(batch_size,height,max_width,channels))

    #We loop over the batch once 
    for enu,img in enumerate(batch):
        shape=img[2]
        img_stack[enu,:,0:shape,:]=np.squeeze(img[0])
        
    img_stack=torch.tensor(img_stack).cuda().float().permute((0,3,1,2))
    label_stack=torch.tensor(label_stack, dtype=torch.int32).cuda()
    widths=torch.tensor(widths,dtype=torch.int32)
    length_stack_y=torch.tensor(length_stack_y,dtype=torch.int32)
        
    return img_stack,label_stack,widths,length_stack_y

