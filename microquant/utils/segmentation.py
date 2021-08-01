# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:42:34 2021

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

import os
import cv2
import subprocess
import tifffile as tf

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

class HEImageDataset():
    def __init__(self, filename, n_classes, **kwargs):
        
        print('*** New image ***')
        print(f'Source: {filename}')
        
        self.target_pixsize = kwargs.get('target_pixsize', 2.5)
        self.patch_size = kwargs.get('patch_size', 128)
        self.stride = kwargs.get('stride', 16)
        self.density = kwargs.get('density', 32)
        self.augmentation = kwargs.get('augmentation', None)
        self.batch_size = kwargs.get('batch_size', 20)
        self.device = kwargs.get('device', 'cuda')
        
        print(f'\tPixel size: {self.target_pixsize}')
        print(f'\tPatch size: {self.patch_size}')
        print(f'\tbatch size: {self.batch_size}')        
        
        self.filename = filename
        self.n_classes = n_classes
        
        # Read czi image
        self.Image = aicsimageio.AICSImage(self.filename)
        self.czi = aicspylibczi.CziFile(self.filename)
        
        self.resolution = self.Image.physical_pixel_sizes.X/10
        
        C0 = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
        C1 = self.czi.read_mosaic(C = 1, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
        C2 = self.czi.read_mosaic(C = 2, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
        self.image = np.vstack([C0, C1, C2])
        
        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))
        
        self.prediction = np.zeros((self.n_classes, self.image.shape[1], self.image.shape[2]),
                                   dtype='float32')
        
        # assign indeces on 2d grid and pad to prevent index errors
        self.create_samplelocations()
        
    def create_samplelocations(self):
        """
        Prepares a list of locations where the image will be fed forward
        through the network
        """
        
        # create sampling locations, omit half-tile sized margin at image edge
        X = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, self.density)
        Y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, self.density)
        
        self.locations = []
        for x in tqdm(X, desc='\tBrowsing image...'):
            for y in Y:
                patch = self.image[:,
                                   x - self.patch_size//2 : x + self.patch_size//2,
                                   y - self.patch_size//2 : y + self.patch_size//2]
                if np.sum(patch.sum(axis=0) == 0) > 150 or np.sum(patch.sum(axis=0) >= 3*np.iinfo(patch.dtype).max) >=150:
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
                                    shuffle=True, num_workers=0)        
            tk0 = tqdm(dataloader, total=len(dataloader), desc='\tTilewise forward segmentation...')
            for b_idx, data in enumerate(tk0):
                    
                data['image'] = data['image'].to(self.device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()
                
                # iteratively write results from batches to prediction map
                for idx in range(out.shape[0]):
                    self[b_idx*self.batch_size + idx] = out[idx]
                    
                    # plt.figure()
                    # plt.imshow(np.argmax(self.prediction, axis=0), vmin=0, vmax=0+self.n_classes)
                    # plt.savefig(os.path.join(r'E:\Promotion\Projects\2021_Necrotic_Segmentation\visualization', 
                    #                          f'{b_idx*self.batch_size + idx}'))
                    # plt.close()

    def export(self, filename, **kwargs):
        """
        Export prediction map to file with deflation compression
        """
        
        softmax = kwargs.get('softmax', True)
        upscale = kwargs.get('upscale', None)
        export = kwargs.get('export', True)
        
        if upscale is not None:
            assert len(upscale) == 2
            self.prediction = self.prediction.transpose((1,2,0))
            self.prediction = cv2.resize(self.prediction, upscale)
        
        if softmax:
            self.prediction = np.argmax(self.prediction, axis=0)
        
        if export:
            tf.imwrite(filename, self.prediction.astype('uint8'))
            
def segment_HE(ImgPath, **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.
    ilastik_exe : str
        DESCRIPTION.
    classifier : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    bst_model = kwargs.get('model', r'.\models\HE\segmentation_model.bin')
    params = kwargs.get('params', r'.\models\HE\segmentation_model.yaml')
    device = kwargs.get('device', 'cuda')
    stride = kwargs.get('stride', 16)
    output = kwargs.get('output', os.path.join(os.path.dirname(ImgPath),
                                               'HE_seg_Unet.tif'))
    
    if not os.path.exists(ImgPath):
        raise FileNotFoundError('Input image {ImgPath} not found!')
        
    if not os.path.exists(bst_model):
        raise FileNotFoundError('Input segmentation model {bst_model} not found!')
        
    if not os.path.exists(params):
        raise FileNotFoundError('Input parameter file {params} not found!')
    
    # read config
    with open(params, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    img_size = int(data['Input']['IMG_SIZE'])
    batch_size = int(data['Hyperparameters']['BATCH_SIZE'])
    pxsize = data['Input']['PIX_SIZE']
    n_classes = data['Hyperparameters']['N_CLASSES']
    
    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=n_classes, 
        activation=None,
    )
    
    model.load_state_dict(torch.load(bst_model)['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    aug_forw = A.Compose([
        PadIfNeeded(min_width=img_size, min_height=img_size,
                    border_mode=cv2.BORDER_REFLECT)
    ])
    
    ds = HEImageDataset(ImgPath,
                        n_classes=n_classes,
                        patch_size=img_size,
                        stride=stride,
                        augmentation=aug_forw,
                        target_pixsize=pxsize,
                        batch_size=batch_size)
    ds.predict(model)
    ds.export(output)
    
    

def segment_CD31_Pimo_Hoechst(ImgPath, method='Ilastik', **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    classifier : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Create temporary directory - for stuff
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
        
    # Preprocess image and save to temporary dir.
    img = tf.imread(ImgPath)
    img = normalize(img)
    f_img = os.path.join(tmp_dir,'_norm.tif')
    cv2.imwrite(f_img, img[::-1].transpose((1,2,0)))    
    
    if method == 'Ilastik':
        classifier = kwargs.get('classifier', None)
        ilastik_exe = kwargs.get('ilastik_exe', None)
        
        assert classifier is not None
        assert ilastik_exe is not None
        
        if not os.path.exists(classifier):
            raise FileNotFoundError(f'Classifier file {classifier} not found')
            
        if not os.path.exists(ilastik_exe):
            raise FileNotFoundError(f'Ilastik executable {ilastik_exe} not found')
        
        cmd = [
            ilastik_exe, '--headless',
            f'--project={classifier}',
            'output_format=tif',
            '--export_dtype=uint8',
            '--export_source=Simple Segmentation',
            f_img
            ]
        
        subprocess.check_output(cmd)
        
        
    
    
    

def train_IF_classifier(img, labels, **kwargs):
    """

    Parameters
    ----------
    image : nd array in XYC format
        DESCRIPTION.
    labels : nd labelled array in XY format.
        DESCRIPTION.

    Returns
    -------
    None.

    """

    sigma_min = kwargs.get('sigma_min', 1)
    sigma_max = kwargs.get('sigma_max', 16)
    intensity = kwargs.get('intensity', True)
    edges = kwargs.get('edges', True)
    texture = kwargs.get('texture', True)
    multichannel = kwargs.get('multichannel', True)
    
    # make sure labels are passed as integers
    assert labels.dtype == np.uint8
    
    # initialize features
    features_func = partial(feature.multiscale_basic_features,
                            intensity=intensity, edges=edges, texture=texture,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            multichannel=multichannel)
    
    # calculate features
    features = features_func(img)
    
    # initialize and fit RF
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    
    clf = future.fit_segmenter(labels, features, clf)
    result = future.predict_segmenter(features, clf)
    
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    ax[0].imshow(img)
    # ax[0].contour(labels)
    ax[0].set_title('Image, mask and segmentation boundaries')
    ax[1].imshow(result)
    ax[1].set_title('Segmentation')
    fig.tight_layout()
    

        
# if __name__ == '__main__':
    
#     path = r'E:\Promotion\Projects\2021_Necrotic_Segmentation\src\IF\raw'
    
#     for f in os.listdir(path):
        
#         print(f'Normalizing {f}...', end='')
        
#         fname = os.path.join(path, f)
#         img = tf.imread(fname)
#         img = normalize(img)
#         cv2.imwrite(os.path.join(path, f.replace('.tif', '') + '_norm.tif'), img[::-1].transpose((1,2,0)))
#         print('Done')
    
    # path = r'K:\Xenias_data\E16b_0039\Micromilieu\Ilastik'
    
    # _image = os.path.join(path, 'Composite_E16b_0039.czi - E16b_0039.czi #2.tif')
    # _mask = os.path.join(path, 'Simple segmentation_E16b_0039.tif')
    
    # image = tf.imread(_image)
    # mask = tf.imread(_mask)
    
    # # re-arrange input image to XY
    # if image.shape[0] == np.min(image.shape):
    #     image = image.transpose((1,2,0))
    
    # train_IF_classifier(image, mask)
    
