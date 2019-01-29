#We might even be able to use the dataset from another guy 
import random
from glob import glob
from os.path import basename
from skimage.transform import resize

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import re
import string
import cv2
from fake_texts.imgaug_transformations import augmentations

class hwrDataset(Dataset):
    def __init__(self, input_folder="/home/leander/AI/data/OCR_handwritten/",width=32):
        self.width=width
        train_threshold = 0.75

        pool = ''
        pool += string.ascii_letters
        pool += "0123456789"
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
        pool += ' '
        self.keys = list(pool)
        self.values = np.array(range(1,len(pool)+1))
        self.dictionary = dict(zip(self.keys, self.values))

        self.decode_dict=dict((v,k) for k,v in self.dictionary.items())
        self.decode_dict.update({93 : "OOK"})


        files=glob(input_folder+"img/*/*/*.png")
        name_to_file = {}
        rows = []

        for file_name in files:
            name_to_file[basename(file_name).rstrip(".png")] = file_name

        # Reduce the time it takes for this if needed.
        for line in open(input_folder+"txt/sentences.txt", "r").readlines():  
            parts = line.split(" ")
            if parts[0] in name_to_file:
                gt = " ".join(parts[9:]).rstrip("\n")

                # Filters out characters not in the alphabet.
                processed_gt = "".join([k for k in gt if k in pool]).replace("|" ," ")
                processed_gt=[self.dictionary[x] if x in self.keys else 93 for x in processed_gt ]
                if len(processed_gt) > 0:
                    rows.append([parts[0], name_to_file[parts[0]], processed_gt])
        self.data = pd.DataFrame(rows[:int(len(rows) * train_threshold)], columns=["name", "path", "groundtruth"])
        
    def __len__(self):
        return int(self.data.__len__())

    def __getitem__(self, index):
        
        try:
            img = Image.open(list(self.data.iloc[[index]].path)[0]).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        except Exception:
            print(index)
        img=cv2.cvtColor(np.array(img),cv2.COLOR_GRAY2RGB)
        

        
        img=augmentations.augment_image(np.array(img))/255
        label = self.data.iloc[[index]].groundtruth
        label=np.array(label.values[0])
        
        
        resize_shape=(self.width,int(self.width*img.shape[1]/img.shape[0]))
        
        img = resize(img,resize_shape,mode="constant")     
        img = np.expand_dims(img,0)
        
        #We also need to downscale the image
        
        #how to give out as strings 
        #[self.decode_dict[x]  for x in strings ]
        return img, label,img.shape[2],len(label)
