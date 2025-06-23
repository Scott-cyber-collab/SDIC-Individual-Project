import argparse
# argparse for passing command-line arguments
import boto3
#from botocore.exceptions import NoCredentialsError

# boto3 and botocore for handling AWS S3 interactions
#S3 is Amazon's Simple Storage Service. Designed to store and retrieve any amount of data from anywhere on the web
import lightning as L
# L for managing the training and inference process using PyTorch Lightning
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from PIL import Image
from skimage.transform import resize

import time
import toml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
# All modules in this function are for various utility functions
# E.g. configuration loading, deep learning, data transformations

from Utils import TileOps, ValidateConfigs
# Custom utility modules for tile operations and validating configurations

from models.SAM_Masking import MaskGenerator
#from models.CellClassifier import base_convnet
#from models.Cell_Classifier import HE_mask_cell_classifier
# Custom model classes for mask generation and cell classification

# UNCOMMENT FOR HE.OME.TIF / COMMENT FOR DAPI
from dataloader.Dataloader import DataGenerator_HE

# UNCOMMENT FOR DAPI / COMMENT FOR HE.OME.TIF
#from dataloader.Dataloader import DataGenerator_HE
#from dataloader.Dataloader import combined_df

# Custom data loaders


config_file = r"/myriadfs/home/ucapocx/Scratch/Name/SAM_Masking/config_infer_CRC01.ini"

def load_config(config_file):
    # Loads the configuration from a TOML file
    return toml.load(config_file)


def freeze_model(model):
    # Freezes the parameters of the model to prevent them from being updated during inference
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_cell_classif_transform(config): # copied from training code.

    # Requires a set of transformations to apply to data based on configuration
    # Differentiates between using pre-trained network and training from scratch

    val_transform_list = [v2.ToImage()]

    if config['ADVANCEDMODEL']['Pretrained']:
        # if using a pretrained torchvision network, scale 0-1 and normalise
        print('Assuming pre-trained network ; scaling [0-1] then normalising according to torchvision models.')
        tr = [v2.ToDtype(torch.float32, scale=True),
              v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        print('Assuming network trained from scratch ; no data scaling done.')
        # Otherwise, just set array to float32 by default.
        tr = [v2.ToDtype(torch.float32, scale=False)]

    val_transform = v2.Compose(val_transform_list + tr)

    return val_transform    


def inference(config): 
    
    # Handles the entire inference process

    # Deal with the inference configuration file, reusing the training config files for the important parameters.
    
    # COMMENTED OUT THE 3 LINES BELOW AS THEY ARE FOR CELLCLASSIFIER
    
    #cell_classifier_configs = [base_convnet.read_config_from_checkpoint(model_path) for model_path in config['CELL_MODEL']['checkpoint']]
    #training_config = ValidateConfigs.validate_training_configs_for_inference(cell_classifier_configs)  # make sure the model(s) (if ensemble) do match
    #config = ValidateConfigs.merge_configs(config, training_config)  # important parameters from training should now be merged with infer config file

    num_workers = int(.95 * mp.Pool()._processes / config['ADVANCEDMODEL']['n_gpus'])

    # Initialise PyTorch Lightning trainer with the specified configuration
    trainer = L.Trainer(devices=config['ADVANCEDMODEL']['n_gpus'],
                        accelerator="gpu",
                        strategy="ddp",
                        logger=False,
                        precision=config['BASEMODEL']['Precision'],
                        use_distributed_sampler = False,
                        benchmark=False)    

    print(f"Available GPUs: {torch.cuda.device_count()} [{config['ADVANCEDMODEL']['n_gpus']} GPUs used for inference]")

    # Seeds random number generators for reproducibility
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    #if trainer.is_global_zero: # download data/checkpoints from S3
        #download_HE_IF_data(config)
        #download_checkpoints(config)
    
    trainer.strategy.barrier() # synchronise saves
    torch.cuda.empty_cache()


    print('=====================================================================================')
    print('1. Remove non-background tiles')

    time_start = time.time()
    
    # Commenting out line 122 because this takes in WSI and returns tiles
    # Not what we want because this uses OpenSlide, whereas we use TiffSlide
    
    # UNCOMMENT FOR HE.OME.TIF / COMMENT FOR DAPI
    tile_coords_no_background  = TileOps.get_all_tiles(config['DATA']['HE_File'], config)
    #tile_coords_no_background  = TileOps.get_nonbackground_tiles(config['DATA']['HE_File'], config)
    tile_dataset = pd.DataFrame({'coords_x': tile_coords_no_background[:, 0], 'coords_y': tile_coords_no_background[:, 1]})
    
    # -------------------------- MORE SAARAH EDITS -------------------------- #
    #td_export_filename = (r'/home/dgs1/data/Saarah/patient_data/P2A_C3H_M5_001/inference/tile_dataset.csv')
    #tile_dataset.to_csv(td_export_filename, index=False) 
    #print(f'tile_dataset exported as csv file to {td_export_filename}.')
    # ----------------------------------------------------------------------- #
    
    # Imported combined_df from Dataloader.py
    # This is being fed to class DataGenerator_HE

    # UNCOMMENT FOR DAPI / COMMENT FOR HE.OME.TIF
    #tile_dataset = combined_df
    
    # -------------------------------------------------------------------------- #
    # Subset DF to debug:
    #tile_dataset = tile_dataset[(tile_dataset['coords_x'] > 5000) & (tile_dataset['coords_x'] < 6000) & 
                            #(tile_dataset['coords_y'] > 6000) & (tile_dataset['coords_y'] < 7000)]
                            
    #num_rows = len(tile_dataset)
    #eighth_size = num_rows // 8      # An eighth of the dataframe
    #start_index = 7 * eighth_size
    #end_index = 7 * eighth_size
    #tile_dataset = tile_dataset.iloc[start_index:]
    # -------------------------------------------------------------------------- #

    # Manually sample
    tile_dataset = tile_dataset.sample(frac=1, random_state=42).reset_index(drop=True) ## Balancing
    tiles_per_gpu = len(tile_dataset) // trainer.world_size
    start_idx     = trainer.global_rank * tiles_per_gpu
    end_idx = start_idx + tiles_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(tile_dataset)    
    print(f"trainer with rank {trainer.global_rank} uses indexes {start_idx} to {end_idx} for tile_dataset originally of length { len(tile_dataset)}.")
    tile_dataset = tile_dataset[start_idx:end_idx]

    # Append patient name to tile_dataset
    tile_dataset['patient'] = config['CRITERIA']['patient']     

    time_end = time.time()
    print(f'{len(tile_dataset)} tiles to run SAM on after background removal [{int(time_end - time_start)} seconds]')

    print('=====================================================================================')
    print('2. Mask Generation')

    mask_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.half, scale=False)])
    
    # Load model and freeze layers
    config['BASEMODEL']['Input_Size'] = [64, 64] # fixed for the SAM model - this is the size of the final embedding
    model_maskgenerator = MaskGenerator(config, config['SAM_MODEL']['checkpoint'])
    model_maskgenerator.eval()
    model_maskgenerator = freeze_model(model_maskgenerator)
    
    # Create dataloader
    data = DataLoader(DataGenerator_HE(tile_dataset, config, transform = mask_transform),
                        batch_size=config['BASEMODEL']['Batch_Size_Masking'],
                        num_workers=num_workers,
                        pin_memory=False,
                        shuffle=False)
    
    print(data)

    # ----------------- SAARAH EDITS FOR MEMORY USAGE ------------------ #
    
    # Set memory management settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Clear cache before starting
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- #

    predictions         = trainer.predict(model_maskgenerator, data)

    cropped_masks       = torch.cat([cropped_mask for cropped_mask, center, idx in predictions], dim=0)##[NMask, H, W]
    centers             = torch.cat([center for cropped_mask, center, idx in predictions], dim=0) ## [NMasks,2]
    indexes             = torch.cat([idx for cropped_mask, center, idx in predictions], dim=0) ## [NMask ]

    # Create a tile dataset which fits with the structure of the model.
    dx, dy = config['BASEMODEL']['Input_Size'][1]/2, config['BASEMODEL']['Input_Size'][0]/2
    cropped_masks = cropped_masks.cpu().numpy()
    
    tile_dataset_classif = pd.DataFrame({'loc_x':tile_dataset['coords_x'].iloc[indexes.cpu().numpy()].astype(np.int64),
                                            'loc_y':tile_dataset['coords_y'].iloc[indexes.cpu().numpy()].astype(np.int64),
                                            
                                            # ---------- NEW CHANGES - SWAP X AND Y FOR MASK COORDS ---------- #
                                            'mask_x':centers[:,1].cpu().numpy().astype(np.int64)-np.int64(dx),
                                            'mask_y':centers[:,0].cpu().numpy().astype(np.int64)-np.int64(dy),
                                            # ---------------------------------------------------------------- #
                                            
                                            'mask_filename':[cropped_masks[i] for i in range(cropped_masks.shape[0])]})    
    tile_dataset_classif['patient'] = config['CRITERIA']['patient']

    print(f"Trainer with rank {trainer.global_rank} has index ranging from {torch.min(indexes)} to {torch.max(indexes)}")
    
    time_maskgeneration = time.time()
    print(f'Mask generation ({len(indexes)} masks) completed [{int(time_maskgeneration - time_end)} seconds]')    
    print(f'[rank {trainer.global_rank}] {len(tile_dataset_classif)} Cells were found!')


    print('=====================================================================================')
    print('3. Export')
    
    print(f'Tile_Dataset: {tile_dataset}\n')
    print(f'Tile_Dataset_Classif : {tile_dataset_classif}\n')
    
    #export_dataset = data.dataset.tile_dataset
    export_dataset = tile_dataset_classif
    
    print('export_dataset = tile_dataset_classif')
    print(f'export_dataset.columns: {export_dataset.columns}')
    
    # Create central locations of the mask patch and drop previous parameters
    export_dataset["cx"] = (export_dataset["loc_x"] + export_dataset["mask_x"] + 32).astype('int32')
    export_dataset["cy"] = (export_dataset["loc_y"] + export_dataset["mask_y"] + 32).astype('int32')
    export_dataset = export_dataset.drop(['loc_x', 'loc_y', 'mask_x', 'mask_y'], axis=1)
    
    def mask_to_png(row, base_path, export_mask=True):
        filename = f"{base_path}/{row['cx']}_{row['cy']}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if export_mask:
            patch_mask = row['mask_filename'].astype(int)
            img = 255 * patch_mask
            img = img.astype(np.uint8)
            image = Image.fromarray(img, 'L')  # 'L' mode for (8-bit pixels, black and white)
            image.save(filename) # the new mask path !            
            
        return filename        

    base_path_png = os.path.join(os.path.dirname(config['DATA']['HE_File']), 'inference', 'masks_png')

    export_dataset['mask_filename'] = export_dataset.apply(mask_to_png, axis=1, base_path= base_path_png, export_mask=True)
    export_dataset.to_csv(f'temp_dataset_rank_{trainer.global_rank}.csv', index=False)

    trainer.strategy.barrier() # synchronise saves
    torch.cuda.empty_cache()
    del export_dataset

    # ------------------------------------------------------------------
    # Merge into a single dataframe on rank 0.

    if trainer.is_global_zero:
        
        he_bp = os.path.dirname(config['DATA']['HE_File'])
        export_filename = os.path.join(he_bp, 'inference', 'tile_dataset_with_cell_masks.csv')

        tile_dataset_dict = {}
        for gpu_id in range(trainer.world_size):
            temp_csv_path = f'temp_dataset_rank_{gpu_id}.csv'
            dataset = pd.read_csv(temp_csv_path).reset_index(drop=True)        
            if not dataset.empty:
                tile_dataset_dict[f"{gpu_id}"] = dataset

            os.remove(temp_csv_path)  # delete temp csv file
            del dataset

        if len(tile_dataset_dict): # if found cells, export the final dataframe.
            full_df = pd.concat([value for key, value in tile_dataset_dict.items()], axis=0)
            full_df.to_csv(export_filename, index=False)
            
        print(f'Segmentation of {len(full_df)} cells sucessfully exported to:{export_filename}.')

    trainer.strategy.barrier() # synchronise before looping to next patient if there


# -------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Initialize the parser
    # Handles command-line inputs for configuration file path, no. GPUs, patient folder path
    parser = argparse.ArgumentParser(description="Run Inference on Cell Classifier")
    
    # Add arguments to the parser
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    parser.add_argument('--gpus', type=int , default=1, help='Number of GPUs to use.')
    parser.add_argument('--patient', type=str, default=None, help='Path to patient folder')

    # Parses command-line arguments
    args = parser.parse_args()  

    # Load config file
    config = load_config(args.config)

    config['DATA']['HE_File'] = os.path.join(args.patient, 'DAPI.tif')

    config['CRITERIA']['patient'] = os.path.basename(args.patient)
    config['ADVANCEDMODEL']['n_gpus'] = args.gpus

    # Call inference function    
    inference(config)

