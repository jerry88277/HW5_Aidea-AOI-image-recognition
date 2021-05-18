# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:33:24 2021

@author: JerryDai
"""
import pandas as pd
import numpy as np
import os
from os import walk
from os.path import join
import re
from tqdm import tqdm
import shutil

from joblib import Parallel, delayed
        
# In[] Data Preprocess

## copy & category Train Data

train_txt = pd.read_csv('aoi_data/train.csv', sep = ',')
source_folder = 'aoi_data/train_images_original'
target_folder = 'aoi_data/train_images'

pic_dict = dict()

for i_index, i_label in enumerate(train_txt['Label']):
    
    tmp_folder_path = join(target_folder, str(i_label))
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)

    tmp_source_path = join(source_folder, train_txt.loc[i_index, 'ID'])
    tmp_target_path = join(tmp_folder_path, train_txt.loc[i_index, 'ID'])

    shutil.copyfile(tmp_source_path, tmp_target_path)

