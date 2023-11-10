
import imageio
import warnings;warnings.filterwarnings('ignore')
from PIL import Image 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 

def gen_id_to_ignore():
    global id_to_ignore_or_group
    for i in range(66):
        id_to_ignore_or_group[i] = ignore_label

    ### Convert each class to cityscapes one
    ### Road
    # Road
    id_to_ignore_or_group[13] = 0
    # Lane Marking - General
    id_to_ignore_or_group[24] = 0
    # Manhole
    id_to_ignore_or_group[41] = 0

    ### Sidewalk
    # Curb
    id_to_ignore_or_group[2] = 1
    # Sidewalk
    id_to_ignore_or_group[15] = 1

    ### Building
    # Building
    id_to_ignore_or_group[17] = 2

    ### Wall
    # Wall
    id_to_ignore_or_group[6] = 3

    ### Fence
    # Fence
    id_to_ignore_or_group[3] = 4

    ### Pole
    # Pole
    id_to_ignore_or_group[45] = 5
    # Utility Pole
    id_to_ignore_or_group[47] = 5

    ### Traffic Light
    # Traffic Light
    id_to_ignore_or_group[48] = 6

    ### Traffic Sign
    # Traffic Sign
    id_to_ignore_or_group[50] = 7

    ### Vegetation
    # Vegitation
    id_to_ignore_or_group[30] = 8

    ### Terrain
    # Terrain
    id_to_ignore_or_group[29] = 9

    ### Sky
    # Sky
    id_to_ignore_or_group[27] = 10

    ### Person
    # Person
    id_to_ignore_or_group[19] = 11

    ### Rider
    # Bicyclist
    id_to_ignore_or_group[20] = 12
    # Motorcyclist
    id_to_ignore_or_group[21] = 12
    # Other Rider
    id_to_ignore_or_group[22] = 12

    ### Car
    # Car
    id_to_ignore_or_group[55] = 13

    ### Truck
    # Truck
    id_to_ignore_or_group[61] = 14

    ### Bus
    # Bus
    id_to_ignore_or_group[54] = 15

    ### Train
    # On Rails
    id_to_ignore_or_group[58] = 16

    ### Motorcycle
    # Motorcycle
    id_to_ignore_or_group[57] = 17

    ### Bicycle
    # Bicycle
    id_to_ignore_or_group[52] = 18
    
def map_values(value):
    return id_to_ignore_or_group.get(value, value)

if __name__ == '__main__' : 
    num_classes = 19 #65
    ignore_label = 255 #65
    color_mapping = []
    id_to_trainid = {}
    id_to_ignore_or_group = {}
    
    gen_id_to_ignore()
    
    tr_path = '/d1/daeun/semi/TorchSemiSeg/DATA/mapillary/training/labels/'
    val_path = '/d1/daeun/semi/TorchSemiSeg/DATA/mapillary/validation/labels/'

    new_tr_path = '/d1/daeun/semi/TorchSemiSeg/DATA/mapillary/training/cs_labels/'
    new_val_path = '/d1/daeun/semi/TorchSemiSeg/DATA/mapillary/validation/cs_labels/'

    tr_list = os.listdir(tr_path)
    val_list = os.listdir(val_path)

    print(len(tr_list))
    print(len(val_list))
    
    # train dataset 
    for name in tqdm(tr_list) : 
        map_array = np.array(Image.open(tr_path + name)) 
        new_arr = np.vectorize(map_values)(map_array)
        imageio.imwrite(new_tr_path + name, new_arr)
