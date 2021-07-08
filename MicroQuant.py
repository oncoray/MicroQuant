# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 00:13:48 2021

@author: johan
"""

import os
import numpy as np

from aicsimageio import AICSImage
from aicspylibczi import CziFile
import matplotlib.pyplot as plt

from Utils.utils import czi2hdf5
from Utils.utils import segment_HE
from Utils.utils import segment_IF
from Utils.utils import register



if __name__ == '__main__':
    
    HE_seg_pixsize = 2.5
    
    HE_fname = os.path.join(os.getcwd(), 'ImgData', 'N182e_SAS_25_1_181210-Scene-1-ScanRegion0_HE.czi')
    IF_fname = os.path.join(os.getcwd(), 'ImgData', '181212_N182e_SAS_25_1_IF.czi')
    
    # reader = CziFile.read_mosaic(self, kwargs)
    pixsize = AICSImage(HE_fname).physical_pixel_sizes.X/10  # somehow there's a factor 10 here
    HE = CziFile(HE_fname)
    
    
    # Scalefactor for HE processing
    scale_factor = pixsize/HE_seg_pixsize
    
    # HE_img = HE.read_mosaic(C=0, scale_factor=scale_factor)
    # HE_img = np.flip(HE_img, axis=0)
    
    # Segment HE image
    model = os.path.join('.', 'classifiers', 'bst_model256_0.7974.bin')
    cfg = os.path.join('.', 'classifiers', 'params.yaml')
    
    HE_seg = HE_fname.replace('.czi', '_Simple_Segmentation.tif')
    segment_HE(HE_fname, cfg, model, fname_out=HE_seg)
    
    # resize segmented HE and save
    
    # Segment IF
    # model = os.path.join('.', 'classifiers', 'IF_classifier.ilp')
    
    # hdf5_file = czi2hdf5(IF_fname, replace=True)  # convert to hdf5
    # IF_seg = segment_IF(hdf5_file, model)
    
    # Register
    params = os.path.join('.', 'registration', 'elastix_parameters.txt')
    IF_seg = os.path.join('.', 'ImgData', '181212_N182e_SAS_25_1_IF_Simple_Segmentation.tif')
    # register(IF_seg, HE_seg, params)
    
    
    
    