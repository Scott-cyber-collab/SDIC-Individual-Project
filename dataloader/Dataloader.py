from typing import Dict
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset, DataLoader
from lightning.pytorch import LightningDataModule
from monai.data.wsi_reader import WSIReader
import random
from torchvision.transforms import Resize
from torchvision import models
import matplotlib.pyplot as plt
import os
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import xml.etree.ElementTree as ET
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from PIL import Image
import tifffile
import zarr
from Utils import ColourAugment
from sklearn.model_selection import KFold

class DataGenerator_HE(torch.utils.data.Dataset):

    # This is a tile-wise dataloader which is essentially a copy of the one in InferenceCodeAI_Cell_Classifier,
    # although it can also manage zarr files. It is used in inference mode to read H&E slides to be used with
    # the SAM model.

    def __init__(self, tile_dataset, config=None, target_transform=None, transform=None, ):

        super().__init__()
        self.patch_size = config['BASEMODEL']['Patch_Size']
        self.target_transform = target_transform
        self.tile_dataset = tile_dataset  # either a dataframe, or a list of dataframes.
        self.transform = transform

        if 'WSIReader' in config['BASEMODEL']:
            self.wsi_reader = WSIReader(backend=config['BASEMODEL']['WSIReader'])
        else:
            self.wsi_reader = None

        if 'HE_File' in config['DATA']:
            store = tifffile.imread(config['DATA']['HE_File'], aszarr=True) ## H&E
            self.HE_arr = zarr.open(store, mode='r')                
        else:
            self.HE_arr = None

        self.wsi_object_dict = {}    

    def get_wsi_object(self, image_path):
        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.wsi_reader.read(image_path)
        return self.wsi_object_dict[image_path]                

    def __len__(self):
        return int(self.tile_dataset.shape[0])

    def __getitem__(self, id):  # load patches of size [C, W, H]

        if self.wsi_reader:
            svs_path   = self.tile_dataset['SVS_PATH'].iloc[id]
            wsi_obj    = self.get_wsi_object(svs_path)#self.wsi_reader.read(svs_path)
            level      = 0  # processing done at highest zoom.
            downsample = self.wsi_reader.get_downsample_ratio(wsi_obj, level)
            x_start    = int(self.tile_dataset["coords_x"].iloc[id])
            y_start    = int(self.tile_dataset["coords_y"].iloc[id])
            try:
                patches, _ = self.wsi_reader.get_data(wsi=wsi_obj, location=(y_start, x_start), size=self.patch_size,
                                                  level=level)
            except:
                raise ValueError(
                    f"Could not read {svs_path}: location={(y_start, x_start)}, image size = {wsi_obj.resolutions['level_dimensions'][level]}.")
            patches = np.swapaxes(patches, 0, 2)

        elif self.HE_arr:
            
            print(f'HE_arr: {self.HE_arr}')
            
            loc = [int(self.tile_dataset['coords_x'].iloc[id]),int(self.tile_dataset['coords_y'].iloc[id])] # location of tile

            print(f'loc: {loc}')
            
            # Load the H&E data
            
            # ----------------------- SAARAH EDITS ----------------------- #
            h, w = (self.HE_arr.shape)[0], (self.HE_arr.shape)[1] 
            np_array = np.array(self.HE_arr)
            reshaped_array = np_array.reshape(h, w)
            # ------------------------------------------------------------ #
            
            # WAS:
            # patches = self.HE_arr[0][loc[0]:loc[0]+self.patch_size[0],loc[1]:loc[1]+self.patch_size[1], :]
            # NOW:
            # Note that reshaped_array is in dimensions (y,x)
            # Therefore we use loc[1] first as these are the y_coords
            patches = reshaped_array[loc[1]:loc[1]+self.patch_size[1], loc[0]:loc[0]+self.patch_size[0]]

        if self.transform:
            
            # --- SAARAH EDITS TO CONVERT TO UIN8 (UINT16 NOT SUPPORTED) --- #
            patches = patches.astype(np.float32)
            patches = patches / 65535.0 * 255.0
            patches = patches.astype(np.uint8)
            # -------------------------------------------------------------- #
            
            patches = np.array(resize(to_pil_image(patches), [1024,1024]))
        return patches, id, loc[0]+256, loc[1]+256
