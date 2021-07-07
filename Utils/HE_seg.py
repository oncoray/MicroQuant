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

import yaml
from tqdm import tqdm
import numpy as np

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch

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
        
        czi = aicspylibczi.CziFile(self.filename)
        self.image = czi.read_mosaic(C = 0, scale_factor=self.resolution/target_pixsize)
        self.image = self.image[::-1, :, :]
        self.resolution = target_pixsize
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
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
        
        stride = 2
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
        self.prediction = np.divide(self.prediction, self.N_map)     
    
    def postprocess(self, filename, project=True, **kwargs):
        """
        Export prediction map to file with deflation compression
        """
        
        filename = kwargs.get('filename', None)
        
        if project:
            self.prediction = np.argmax(self.prediction, axis=0)
        if filename is not None:
            tf.imwrite(filename, self.prediction)
        
        return self.prediction


def segment(fname_in, fname_config, fname_model, **kwargs):
    
    fname_out = kwargs.get('fname_out', '')
    STRIDE = kwargs.get('stride', 16)
    MAX_OFFSET = kwargs.get('max_offset', 64)
    N_OFFSETS = kwargs.get('n_offsets', 10)
    SERIES = kwargs.get('series', 2)
    
    # read config
    with open(fname_config, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)    
        
    IMG_SIZE = int(cfg['Input']['IMG_SIZE']/2)
    batch_size = int(cfg['Hyperparameters']['BATCH_SIZE'] * 4)
    PIX_SIZE = cfg['Input']['PIX_SIZE']
    N_CLASSES = cfg['Hyperparameters']['N_CLASSES']

    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
        model.load_state_dict(torch.load(fname_model, map_location=torch.device('cpu'))['model_state_dict'])
    else:
        DEVICE = 'cuda'
        model.load_state_dict(torch.load(fname_model)['model_state_dict'])
    
    
    model = model.to(DEVICE)
    model.eval()
        
    ds = InferenceDataset(fname_in,
                          patch_size=IMG_SIZE,
                          stride=STRIDE,
                          augmentation=None,
                          target_pixsize=PIX_SIZE,
                          max_offset=MAX_OFFSET,
                          n_offsets=N_OFFSETS)
    ds.predict(model, batch_size=batch_size, device=DEVICE)
    prediction = ds.postprocess()
    
    return prediction