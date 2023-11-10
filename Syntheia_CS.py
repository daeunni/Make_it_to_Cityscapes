import imageio
imageio.plugins.freeimage.download()

import warnings;warnings.filterwarnings('ignore')
from PIL import Image 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 

num_classes = 19
ignore_label = 255
img_postfix = '.png'

# Syn label -> CS trainID !! (from RobustNet)
trainid_to_trainid = {
        0: ignore_label,  # void
        1: 10,            # sky
        2: 2,             # building
        3: 0,             # road
        4: 1,             # sidewalk
        5: 4,             # fence
        6: 8,             # vegetation
        7: 5,             # pole
        8: 13,            # car
        9: 7,             # traffic sign
        10: 11,           # pedestrian - person
        11: 18,           # bicycle
        12: 17,           # motorcycle
        13: ignore_label, # parking-slot
        14: ignore_label, # road-work
        15: 6,            # traffic light
        16: 9,            # terrain
        17: 12,           # rider
        18: 14,           # truck
        19: 15,           # bus
        20: 16,           # train
        21: 3,            # wall
        22: ignore_label  # Lanemarking
        }

def map_values(value):
    return trainid_to_trainid.get(value, value)

if __name__ == '__main__' : 
    label_root = '/d1/daeun/semi/TorchSemiSeg/DATA/Synthia/GT/LABELS/'
    new_label_root = '/d1/daeun/semi/TorchSemiSeg/DATA/Synthia/GT/cs_labels/'

    labels = os.listdir(label_root)

    for name in tqdm(labels) : 
        mask = imageio.imread(label_root + name, format='PNG-FI')[:, :, 0]          # MUST this format 
        new_arr = np.vectorize(map_values)(mask)
        imageio.imwrite(new_label_root + name, new_arr)
