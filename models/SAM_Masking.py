import lightning as L
import torch
import numpy as np
from skimage import measure
from segment_anything import sam_model_registry
from segment_anything.utils.amg import build_point_grid, MaskData, batch_iterator, calculate_stability_score, batched_mask_to_box

import matplotlib.pyplot as plt
from segment_anything.utils import amg
from torchvision.ops.boxes import batched_nms, box_area

from segment_anything.utils.transforms import ResizeLongestSide ## @@@
from kornia.geometry.transform import Resize ## @@@
import os
from PIL import Image

class MaskGenerator(L.LightningModule):

    def __init__(self, config, SAM_CHECKPOINT):
        super().__init__()
        self.sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        self.sam_model.eval()
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        self.config = config
        self.n_points = self.config['SAM_MODEL']['points_per_side']
        self.point_grids  = build_point_grid(self.n_points)*1024 #@@@@
        #self.point_grids = np.array([[512, 512]], dtype=np.float64) # example for single point in center of slide
        self.transform = ResizeLongestSide(1024)

        self.Input_Size = torch.as_tensor(self.config['BASEMODEL']['Input_Size'], dtype=torch.int, device=self.device)
        self.Patch_Size = torch.as_tensor(self.config['BASEMODEL']['Patch_Size'], dtype=torch.int, device=self.device)

    def forward(self, patches, cx, cy):       
        
        print(patches)
        
        input_image_torch = torch.as_tensor(patches[0], device=self.device)
        
        # ----------------------- SAARAH EDITS ------------------------ #
        
        print(f"Shape of patches: {patches.shape}")
        print(f"Shape of input_image_torch before permute: {input_image_torch.shape}")
    
        # Add a channel dimension if it does not exist
        if input_image_torch.dim() == 2:
            input_image_torch = input_image_torch.unsqueeze(0)
            print(f"Added channel dimension, new input_image_torch shape: {input_image_torch.shape}")
    
        # Ensure we have 3 channels by replicating the single channel
        if input_image_torch.shape[0] == 1:
            input_image_torch = input_image_torch.repeat(3, 1, 1)
            print(f"Replicated channel dimension, new input_image_torch shape: {input_image_torch.shape}")
            
        # ------------------------------------------------------------- #
        
        # ---------------- SAARAH EDITS FOR PERMUTE ------------------- #
        
        # Add a batch dimension and then permute to match the expected input shape [batch_size, channels, height, width]
        input_image_torch = input_image_torch.unsqueeze(0).permute(0, 1, 2, 3).contiguous()
        print(f"Shape after permute and adding batch dimension: {input_image_torch.shape}")
        
        # ------------------------------------------------------------- #
        
        # Commenting out original line 63
        #input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_images  = self.sam_model.preprocess(input_image_torch)
        self.features = self.sam_model.image_encoder(input_images)
        del input_images
        batch_size    = 400
        data          = MaskData()
        num_pts_found = 0

        for (points,) in batch_iterator(batch_size, self.point_grids):
            batch_data = MaskData()
            transformed_points = self.transform.apply_coords(points, [1024,1024])
            in_points = torch.as_tensor(transformed_points, device=self.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            in_points = in_points[:, None, :]
            in_labels = in_labels[:, None]
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=(in_points, in_labels),
                boxes=None,
                masks=None
            )

            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True ## @@
            )
            batch_data["masks"] = self.sam_model.postprocess_masks(low_res_masks,
                                                                   [1024,1024], ####
                                                                   self.config['BASEMODEL']['Patch_Size'])

            del low_res_masks
            batch_data["masks"] = batch_data["masks"].flatten(0,1)
            batch_data["iou_predictions"] = iou_predictions.flatten(0,1)
            ## Remove based on IOU
            keep_by_iou = batch_data["iou_predictions"] > self.config['SAM_MODEL']["pred_iou_thresh"]
            batch_data.filter(keep_by_iou)
            
            ##Remove based on Stability
            stability_scores = calculate_stability_score(batch_data["masks"], 0, 1.0).to(device=self.device)
            keep_by_stability = stability_scores > self.config['SAM_MODEL']["stability_score_thresh"]
            batch_data.filter(keep_by_stability)

            ## Remove per area size
            batch_data["masks"] = batch_data["masks"]>0
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"]).to(device=self.device)
            areas               = box_area(batch_data["boxes"]).to(device=self.device)
            keep_per_min = areas > self.config['SAM_MODEL']["min_mask_region_area"]
            keep_per_max = areas < self.config['SAM_MODEL']["max_mask_region_area"]
            keep_per_area = keep_per_min & keep_per_max
            batch_data.filter(keep_per_area)
            
            if batch_data["boxes"].size()[0] == 0:  ## Remove the case where there is no masks
                continue
            
            # Padding last dimension to avoid masks errors on the edge
            batch_data["masks"] = torch.nn.functional.pad(batch_data["masks"], (32, 32, 32, 32),"constant", 0)  
            centers       = []
            cropped_masks = []
            for i, mask in enumerate(batch_data["masks"]):
                bbox   = batch_data["boxes"][i]+32 ## To account for the padding                                

                center = torch.zeros((2), device=self.device, dtype=torch.int)
                center[0] = (bbox[1] + bbox[3]) / 2
                center[1] = (bbox[0] + bbox[2]) / 2

                cropped_mask = mask[int(center[0] - self.Input_Size[0] / 2): int(center[0] + self.Input_Size[0] / 2),
                                    int(center[1] - self.Input_Size[1] / 2): int(center[1] + self.Input_Size[1] / 2)]
                
                cropped_masks.append(cropped_mask)

                center[0] -= 32 ## Patches frame of reference
                center[1] -= 32                
                centers.append(center)

            batch_data['centers'] = centers
            batch_data['cropped_masks'] =  cropped_masks
            num_pts_found += len(centers)
            del batch_data["masks"]            
            data.cat(batch_data)
            
        if num_pts_found == 0:
            return None
            #data = dict()
            #data["cropped_masks"] = (torch.empty((0, 64, 64)),)
            #data["centers"] = (torch.empty((0, 2)),)
            #return data
        else:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_predictions"],
                torch.zeros_like(data["boxes"][:, 0]),  
                iou_threshold=self.config['SAM_MODEL']["box_nms_thresh"])
            data.filter(keep_by_nms)
            return data
            
            """ uncomment if you only want to keep the cell closest to the center of the query point (for single point query)
            if len(data['centers'])>1:
                target = torch.tensor([256, 256], dtype=torch.float32, device=self.device)
                distances = [torch.norm(tensor.float() - target).item() for tensor in data['centers']]
                closest_index = torch.tensor([distances.index(min(distances))], device=self.device)
                data.filter(closest_index)
            return data
            """
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        patches, ids, cx, cy = batch
        data = self.forward(patches, cx, cy)

        if data is not None:

            cropped_masks = torch.stack(data["cropped_masks"], dim=0)
            centers = torch.stack(data["centers"], dim=0)
            
            ids = ids.repeat_interleave(cropped_masks.size()[0])
            ## Collapse masks per patch in mask over all patches, make the same for centers
            cropped_masks = cropped_masks.view(-1, cropped_masks.shape[-2], cropped_masks.shape[-1])
            centers       = centers.view(-1, centers.shape[-1])        

            return cropped_masks, centers, ids #data
        
        else:

            return torch.empty((0, 64, 64)), torch.empty((0, 2)), torch.empty(0)






