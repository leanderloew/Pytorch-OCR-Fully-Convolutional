import sys
#sys.path.append("/home/leander/AI/repos/gen_text/TextRecognitionDataGenerator/") 
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
    def __init__(self,batch_size,epoch_size=10,random_strings=True,num_words=5,transform=False):
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
        self.format=32
        #Skeqing angle
        self.skew_angle=0
        self.random_skew=True
        self.use_wikipedia=True
        self.blur=0
        self.random_blur=True

        """Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures""",
        self.background=0

        """Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random""",

        self.distorsion=0
        """Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both"""
        self.distorsion_orientation=0
        self.width=-1
        self.orientation=0
        self.text_color='#282828'
        self.space_width=1.0
        self.extension="jpg"
        self.handwritten=False
        self.name_format=0
        self.alignment=1

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
        sometimes_5 = lambda aug: iaa.Sometimes(0.5, aug)
        sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
        
        self.seq = iaa.Sequential([
            sometimes_1(iaa.CropAndPad(
                percent=(-0.02, 0.02),
                pad_mode=["edge"],
                pad_cval=(0, 255)
            )),
            #This inverts completely inverts the text ( make black white etc.)
            sometimes_1(iaa.Invert(1, per_channel=True)), 

            #This does some affine transformations
            sometimes_1(iaa.Affine(
                scale={"x": (0.8, 1), "y": (0.8, 1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0, 0), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-2, 2), # rotate by -45 to +45 degrees
                shear=(-2, 2), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=["edge"]# use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            sometimes_1(iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 3)), # blur image using local means with kernel sizes between 2 and 7
                #iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
            ])),
            sometimes_1(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            sometimes_1(iaa.AddToHueAndSaturation((-20, 20))), # change hue and saturation
            sometimes_1(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            sometimes_1(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.2)), # move pixels locally around (with random strengths)
            sometimes_1(iaa.PiecewiseAffine(scale=(0.001, 0.005))), 

        ])

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
        x_len=X.shape[2]
        y_len=len(y)
        
        if self.transform== True: 
            X=self.transform(X)/255
        else:
            x=x/255
        #Here we get some more distorition from Imgaug. 
        #
        

        return X, y,x_len,y_len
    
# a simple custom collate function, just to show the idea
#Basically the collate gets calleda list of the putputs of all the batches

def my_collate(batch):
    #some shapes 
    one,height,wi,channels=batch[0][0].shape
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg