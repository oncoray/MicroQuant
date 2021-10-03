# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:42:34 2021

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, future
from skimage import transform
from sklearn.ensemble import RandomForestClassifier
from functools import partial

import os
import cv2
import subprocess
import tifffile as tf
import h5py

import aicspylibczi
import aicsimageio

import yaml
from tqdm import tqdm

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch

from utils.utils import normalize
from utils.utils import find_Ilastik

import joblib

class IFImageDataset():
    def __init__(self, filename, classifier, **kwargs):

        self.sigma_min = kwargs.get('sigma_min', 4)
        self.sigma_max = kwargs.get('sigma_max', 16)
        self.use_dask = kwargs.get('use_dask', True)
        
        self.filename = filename
        self.classifier = classifier
        
        self.image = self.read_czi(self.filename)  # read czi
        self.clf = joblib.load(self.classifier)  # load RF classifier file
        
        self.features_func = partial(feature.multiscale_basic_features,
                                     intensity=True, edges=True, texture=True,
                                     sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                     multichannel=True, num_workers=4)    
    
    def read_czi(self, filename):
        
        try:
        # Read czi image
            self.AICS = aicsimageio.AICSImage(filename)
            
            if self.use_dask:
                self.dask_image = self.AICS.get_dask_image_data("YXC")
            else:
                self.dask_image = self.AICS.get_image_data("YXC")
            
        # use pylibczi in case of error
        except ValueError:
            Image = aicspylibczi.CziFile(filename)
            C0 = Image.read_mosaic(C=0)
            C1 = Image.read_mosaic(C=1)
            C2 = Image.read_mosaic(C=2)

            self.dask_image = np.vstack([C0, C1, C2]).transpose((1,2,0))
            
        self.dask_image = normalize(self.dask_image)
        self.prediction = np.zeros_like(self.dask_image[:,:,0])  # allocate output
        

    def predict(self, ts=512):
        
        shape = self.dask_image.shape
        X = shape[0]//ts + 1
        Y = shape[1]//ts + 1
        
        tk0 = tqdm(np.arange(0, X, 1), desc='\tRandom forest-based segmentation')
        for x in tk0:
            for y in np.arange(0, Y, 1):
                
                try:
                    img = self.dask_image[x*ts:x*ts + ts,
                                          y*ts:y*ts + ts,
                                          :]
                    features = self.features_func(img)
                    self.prediction[x*ts:x*ts + ts,
                                    y*ts:y*ts + ts] = future.predict_segmenter(features, self.clf)
                    tk0.set_postfix(Processed_tiles='{:d}/{:d}'.format(y, Y))
                except Exception:
                    pass
                
    def export(self, filename):

        tf.imwrite(filename, self.prediction)
        self.foutput = filename
                
        

# class ILPImageDataset():
#     def __init__(self, filename, classifier, **kwargs):
        
#         print('*** New image ***')
#         print(f'Source: {filename}')
        
#         self.filename = filename
#         self.classifier = classifier
        
#         self.ilastik_exe = kwargs.get('ilastik_exe', find_Ilastik())
#         self.read_czi(filename)
#         self.preprocess()
        
        
#     def read_czi(self, filename):
        
#         # Read czi image
#         self.Image = aicsimageio.AICSImage(filename)
#         self.czi = aicspylibczi.CziFile(filename)
        
#         if self.Image.dims.C == 1:
#             self.image = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0]
#             self.image = self.image.transpose((2,0,1))  # bring image to order CXY
        
#         else:
#             C0 = self.czi.read_mosaic(C = 0, scale_factor=1)[0][None, :, :]
#             C1 = self.czi.read_mosaic(C = 1, scale_factor=1)[0][None, :, :]
#             C2 = self.czi.read_mosaic(C = 2, scale_factor=1)[0][None, :, :]
#             self.image = np.vstack([C0, C1, C2])
            
        
#     def preprocess(self):
#         "Re-Save image as tif"
        
#         # normalize to predefined quartile ranges
#         self.image = normalize(self.image).astype('uint8')
#         self.image = self.image.transpose((1,2,0))[None, None, :]  # add empty T and Z axis
        
#         outname = os.path.join(
#             os.path.dirname(self.filename),
#             '1_seg',
#             os.path.basename(self.filename).replace('.czi', '') + '_norm.hdf5' 
#             )
        
#         with h5py.File(outname, "w") as f:
#             dset = f.create_dataset("data", self.image.shape, chunks=True,
#                                     dtype=self.image.dtype)
#             dset[:] = self.image

#         self.tmp_name = outname
        
#     def predict(self):
        
#         "Run Ilastik classification"
#         cmd = [
#             self.ilastik_exe, '--headless',
#             f'--project={self.classifier}',
#             '--output_format=tif',
#             '--export_dtype=uint8',
#             '--export_source=Simple Segmentation',
#             self.tmp_name
#             ]
        
#         subprocess.check_output(cmd)
        
class HEImageDataset():
    def __init__(self, filename, n_classes, **kwargs):
        
        self.target_pixsize = kwargs.get('target_pixsize', 2.5)
        self.patch_size = kwargs.get('patch_size', 128)
        self.stride = kwargs.get('stride', 16)
        self.density = kwargs.get('density', 32)
        self.augmentation = kwargs.get('augmentation', None)
        self.batch_size = kwargs.get('batch_size', 20)
        self.device = kwargs.get('device', 'cuda')
        
        self.c_order =kwargs.get('c_order', 'RGB')
        
        print(f'\tPixel size: {self.target_pixsize}')
        print(f'\tPatch size: {self.patch_size}')
        print(f'\tbatch size: {self.batch_size}')        
        
        self.filename = filename
        self.n_classes = n_classes
        
        self.read_czi(filename)
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
        self.prediction = np.zeros((self.n_classes, self.image.shape[1], self.image.shape[2]),
                                   dtype='float32')
        
        # assign indeces on 2d grid and pad to prevent index errors
        self.create_samplelocations()
        
    def read_czi(self, filename):
        
        # Read czi image
        self.Image = aicsimageio.AICSImage(filename)
        self.czi = aicspylibczi.CziFile(filename)
        
        self.resolution = self.Image.physical_pixel_sizes.X/10
        
        if self.Image.dims.C == 1:
            self.image = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0]
            self.image = self.image.transpose((2,0,1))  # bring image to order CXY
        
        else:
            C0 = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            C1 = self.czi.read_mosaic(C = 1, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            C2 = self.czi.read_mosaic(C = 2, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            self.image = np.vstack([C0, C1, C2])
            
        # re-arrange color axis if image is BGR and not RGB
        if self.c_order != 'BGR':
            self.image = self.image[::-1]
        elif self.c_order == 'RGB':
            pass
            
        
        
    def create_samplelocations(self):
        """
        Prepares a list of locations where the image will be fed forward
        through the network
        """
        
        # create sampling locations, omit half-tile sized margin at image edge
        X = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, self.density)
        Y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, self.density)
        
        self.locations = []
        self.blacklisted_locations = []
        for x in tqdm(X, desc='\tBrowsing image...'):
            for y in Y:
                patch = self.image[:,
                                   x - self.patch_size//2 : x + self.patch_size//2,
                                   y - self.patch_size//2 : y + self.patch_size//2]
                if np.sum(patch.sum(axis=0) == 0) > 150 or np.sum(patch.sum(axis=0) >= 3*np.iinfo(patch.dtype).max) >=150:
                    self.blacklisted_locations.append([x,y])
                    continue
                self.locations.append([x, y])
        
        np.random.shuffle(self.locations)
        
        
    def __len__(self):
        return len(self.locations)
    
    def __getitem__(self, key):
        
        loc = self.locations[key]
        x, y = loc[0], loc[1]
        
        patch = self.image[:,
                           x - self.patch_size//2 : x + self.patch_size//2,
                           y - self.patch_size//2 : y + self.patch_size//2].astype(float)
            
        return {'image': torch.from_numpy(patch)}
    
    def __setitem__(self, key, value):
        
        # crop center of processed tile
        patch = value[:,
                      self.stride : - self.stride,
                      self.stride : - self.stride]
        
        size = patch.shape[1]
        x, y = self.locations[key]
        
        self.prediction[:,
                        x - size//2 : x + size//2,
                        y - size//2 : y + size//2] += patch
            
    def predict(self, model):

        with torch.no_grad():
            dataloader = DataLoader(self, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)
            tk0 = tqdm(dataloader, total=len(dataloader), desc='\tTilewise forward segmentation...')
            for b_idx, data in enumerate(tk0):
                    
                data['image'] = data['image'].to(self.device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                
                # iteratively write results from batches to prediction map
                for idx in range(out.shape[0]):
                    self[b_idx*self.batch_size + idx] = out[idx]
                    
        # set blacklisted tiles to correct background
        size = self.patch_size
        for loc in self.blacklisted_locations:
            x, y = loc[0], loc[1]
            
            self.prediction[0,
                x - size//2 : x + size//2,
                y - size//2 : y + size//2] = np.ones((size, size))
                    

    def export(self, filename, **kwargs):
        """
        Export prediction map to file with deflation compression
        """
        
        softmax = kwargs.get('softmax', True)
        upscale = kwargs.get('upscale', True)
        export = kwargs.get('export', True)
        
        if upscale:
            print('\t---> Upscaling')
            self.prediction = self.prediction.transpose((1,2,0))
            factor = self.Image.dims.X/self.prediction.shape[0]
            
            # prediction = torch.from_numpy(self.prediction)
            # prediction.to(self.device)
            # upsampler = torch.nn.Upsample(size=[self.Image.dims.Y,
            #                                     self.Image.dims.X,
            #                                     np.min(self.prediction.shape)])
            # self.prediction = upsampler(prediction).detach().cpu().numpy()
            self.prediction = transform.rescale(self.prediction, scale=(factor, factor, 1.0))
        
        if softmax:
            self.prediction = np.argmax(self.prediction, axis=-1)
        
        if export:
            tf.imwrite(filename, self.prediction.astype('uint8'))
            self.foutput = filename
            
def segment_he(MQJob, MQmodel, **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Get path of input image
    ImgPath = os.path.join(MQJob.sample.dir, MQJob.sample.HE_img)
    if not os.path.exists(ImgPath):
        raise FileNotFoundError(f'Input image {ImgPath} not found!')
    
    # select usable gpu
    device = MQJob.device
    
    # processing setting
    stride = kwargs.get('stride', 16)
    output = kwargs.get('output', os.path.join(os.path.dirname(ImgPath),
                                               '1_seg', 'HE_seg_Unet.tif'))
    batch_size  = kwargs.get('batch_size', MQmodel.hyperparameters_training['BATCH_SIZE'])
    patch_size    = kwargs.get('patch_size', MQmodel.params['Input']['IMG_SIZE'])
    pxsize      = kwargs.get('pxsize', MQmodel.params['Input']['PIX_SIZE'])
    n_classes   = kwargs.get('n_classes', MQmodel.hyperparameters_training['N_CLASSES'])
    
    # Create Unet
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=n_classes, 
        activation=None,
    )
    
    # Load weights and set model to eval mode
    model.load_state_dict(torch.load(MQmodel.file_model)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    aug_forw = A.Compose([
        PadIfNeeded(min_width=patch_size, min_height=patch_size,
                    border_mode=cv2.BORDER_REFLECT)
    ])
    
    # Create Dataset for loading
    ds = HEImageDataset(ImgPath,
                        n_classes=n_classes,
                        patch_size=patch_size,
                        stride=stride,
                        augmentation=aug_forw,
                        target_pixsize=pxsize,
                        batch_size=batch_size)
    
    # run and export prediction
    ds.predict(model)
    ds.export(output, upscale=True, softmax=True)
    
    return ds.foutput

def segment_CD31_Pimo_Hoechst(MQJob, MQmodel, **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.
    MQmodel : MicroQuant model type
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Get input image
    ImgPath=os.path.join(MQJob.sample.dir, MQJob.sample.IF_img)
    
    # Processing settings
    pxsize      = kwargs.get('pxsize', MQmodel.params['Input']['PIX_SIZE'])
    n_classes   = kwargs.get('n_classes', MQmodel.hyperparameters_training['N_CLASSES'])
    classifier = kwargs.get('classifier', MQmodel.file_model)
    
    out_file = os.path.join(os.path.dirname(ImgPath), '1_seg', 'IF_seg.tif')
    
    ds = IFImageDataset(ImgPath, classifier, use_dask=False)
    
    ds.predict()
    ds.export(out_file)
    
    return ds.foutput