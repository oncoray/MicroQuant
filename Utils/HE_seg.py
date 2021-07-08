# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:57 2021
Forward processing of HE image necrosis segmentation
Source: https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb
@author: johan
"""

import aicspylibczi
import tifffile as tf
import os
import cv2
from skimage import transform

from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
import torch

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded

class InferenceDataset():
    def __init__(self, filename, augmentation=None,
                 patch_size=512, stride=16, n_classes=3, target_pixsize = 2.5,
                 max_offset=128, n_offsets=3):
        
        self.filename = filename
        self.augmentation = augmentation
        self.resolution = 0.4418
        
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.inner = int(patch_size - 2*stride)
        self.offset = 0
        self.max_offset = max_offset
        self.n_offsets = n_offsets
        
        self.czi = aicspylibczi.CziFile(self.filename)
        self.image = self.czi.read_mosaic(scale_factor=self.resolution/target_pixsize, C=0).squeeze()
        self.resolution = target_pixsize
        
        # transpose if necessary so that dimensions are CXY
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        self.image = self.image[::-1, :, :]
        
        self.prediction = np.zeros_like(self.image, dtype='float32')
        
        # assign indeces on 2d grid 
        self.create_coords()
        
    def create_coords(self, stepsize=32):
        
        image = self.image
        self.N_map = np.zeros_like(image)
        
        x = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, stepsize)
        y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, stepsize)
        
        # check if forbidden pixels are in the vicinity of these pixels
        self.coords = []
        for _x in x:
            for _y in y:
                
                patch = image[:,
                              _x - self.patch_size//2: _x + self.patch_size//2,
                              _y - self.patch_size//2: _y + self.patch_size//2].sum(axis=0)
                
                if np.sum(patch == 0) > 100 or np.sum(patch >= 3*254) > 100:
                    continue
                else:
                    self.coords.append([_x, _y])
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, key):
        
        try:
            coord = self.coords[key]
            
            ps = self.patch_size
        
            image = self.image[:,
                               coord[0] - ps//2 : coord[0] + ps//2,
                               coord[1] - ps//2 : coord[1] + ps//2]
        except Exception:
            pass

        return {'image': torch.from_numpy(image.copy())}
    
    def __setitem__(self, key, value):
        
        stride = 4
        coord = self.coords[key]
        ps = self.patch_size
        
        patch = value[:,
                      stride : ps - stride,
                      stride : ps - stride]
        
        self.prediction[:,
                        coord[0] - ps//2 + stride : coord[0] + ps//2 - stride,
                        coord[1] - ps//2 + stride : coord[1] + ps//2 - stride] +=patch
        
        self.N_map[:,
                   coord[0] - ps//2 + stride : coord[0] + ps//2 - stride,
                   coord[1] - ps//2 + stride : coord[1] + ps//2 - stride] += 1
        
            
    def predict(self, model, device='cuda', batch_size=6):
        
        with torch.no_grad():            
                
            dataloader = DataLoader(self, batch_size=batch_size,
                    shuffle=False, num_workers=0)

            # iterate over dataloader
            tk0 = tqdm(dataloader)
            
            for i, data in enumerate(tk0):
                
                data['image'] = data['image'].to(device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                
                tk0.set_postfix(batch=i)
                
                # iteratively write results from batches to prediction map
                for b_idx in range(out.shape[0]):
                    self[i*batch_size + b_idx] = out[b_idx]
                        
        # average predictions
        # self.prediction = np.divide(self.prediction, self.N_map)     
    
    def postprocess(self, project=True, **kwargs):
        """
        Export prediction map to file with deflation compression
        """
        
        filename = kwargs.get('filename', None)
        resize = kwargs.get('resize', True)
        
        if project:
            self.prediction = np.argmax(self.prediction, axis=0)
            
        if resize:
            w = self.czi.get_mosaic_bounding_box().w
            h = self.czi.get_mosaic_bounding_box().h
            self.prediction = cv2.resize(self.prediction,
                                         (w, h),
                                         interpolation=cv2.INTER_NEAREST)
        if filename is not None:
            tf.imwrite(filename, self.prediction)
        
        return self.prediction


