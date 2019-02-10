import sys
sys.path.append("fake_texts/") 
#sys.path.append("/home/leander/AI/repos/gen_text/TextRecognitionDataGenerator/fonts") 
import argparse
import os, errno
import random
from random import randint
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



#Basically we want to add the

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
        return [os.path.join("fonts/latin/", font) for font in os.listdir("fonts/latin/")]


import numpy as np
from nltk import word_tokenize
import torch

import torch
from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,epoch_size=10,random_strings=True,num_words=5,transform=False,width=-1,alignment=1,height=32):
        'Initialization'
        #Still in inti
        self.transform=True
        self.random_sequences=False
        self.random_strings=True
        self.include_letters=True
        self.include_numbers=True
        self.include_symbols=True
        self.length=10
        #self.random=False
        self.format=32
        self.use_wikipedia=False
        self.text_color='#282828'
        self.orientation=0
        self.extension="jpg"
        self.handwritten=False
        self.name_format=0

        pool = ''
        pool += "abcdefghijklmnopqrstuvwxyz"
        pool += "0123456789"
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
        pool += ' '
        self.keys = list(pool)
        self.values = np.array(range(1,len(pool)+1))
        self.dictionary = dict(zip(self.keys, self.values))
        self.fonts = load_fonts("en")

        self.decode_dict=dict((v,k) for k,v in self.dictionary.items())
        self.decode_dict.update({67 : "OOK"})
        self.seq = augmentations

    def __len__(self):
        'Denotes the total number of samples'
        return 10000
    def transform(self,img):
        return self.seq.augment_images(img)

    def __getitem__(self, index):
        #We are actually only getting one item here obvoiusly!


        num_words=randint(5,6)
        language="en"
        count=32
        skew_angle=0
        random_skew=True
        blur=0
        random_blur=True
        background=randint(0,1)
        distorsion=randint(0,2)
        distorsion_orientation=randint(0,2)

        #as class function
        def take_10(x):

            if len(x)>num_words:
                return x[0:num_words]
            else:
                return x 

        #if random.random()>0.8:
        #    self.width=random.randint(500,800)
        #else:
        width=-1

        alignment=random.randint(0,2)

        strings = []
        strings = create_strings_randomly(num_words, False, 1,
                      True, True,True, "en")

        strings =[" ".join(take_10(word_tokenize(x))) for x in strings]

        strings_=[list(x)for x in strings]
        self.strings_int=[[self.dictionary[x.lower() ] if x.lower()  in self.keys else 67 for x in m ] for m in strings_]
        self.strings_len=[len(x)for x in self.strings_int]
        string_count = len(strings)

        #What we do here is we space the words quite far appart. 
        if random.random()>0.5:
            width_scale=random.random()*900
            space_width=width_scale/np.sum(self.strings_len)
        else:
            width_scale=random.random()
            space_width=width_scale/np.sum(self.strings_len)
            if random.random()>0.85 and np.max(self.strings_len)<30:
                width=random.randint(500,800)

        image_list=[np.expand_dims(np.array(FakeTextDataGenerator.generate(*j)),0) for j in zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [self.fonts[random.randrange(0, len(self.fonts))] for _ in range(0, string_count)],
                    [self.format] * string_count,
                    [self.extension] * string_count,
                    [skew_angle] * string_count,
                    [random_skew] * string_count,
                    [blur] * string_count,
                    [random_blur] * string_count,
                    [background] * string_count,
                    [distorsion] * string_count,
                    [distorsion_orientation] * string_count,
                    [self.handwritten] * string_count,
                    [self.name_format] * string_count,
                    [width] * string_count,
                    [alignment] * string_count,
                    [self.text_color] * string_count,
                    [self.orientation] * string_count,
                    [space_width] * string_count )]
        
        X=image_list[0]
        y=self.strings_int[0]
        
        y_len=len(y)
        
        if self.transform== True: 
            X=self.seq.augment_images(X)
            
        #X=np.squeeze(X)   

        #X=np.expand_dims(X,0) 
        X =X/255
        x_len=X.shape[2]

        return X, y,x_len,y_len
    
