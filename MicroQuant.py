# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 00:13:48 2021

@author: johan
"""

import os
import numpy as np
import skimageimagetr

from aicsimageio import AICSImage
from aicspylibczi import CziFile
import matplotlib.pyplot as plt

from Utils.HE_seg import segment



if __name__ == '__main__':
    
    HE_seg_pixsize = 2.5E-6
    
    HE_fname = os.path.join(os.getcwd(), 'ImgData', 'N182e_SAS_25_2_181211-Scene-1-ScanRegion0_HE.czi')
    IF_fname = os.path.join(os.getcwd(), 'ImgData', '181212_N182e_SAS_25_2_IF.czi')
    
    # reader = CziFile.read_mosaic(self, kwargs)
    pixsize = AICSImage(HE_fname).get_physical_pixel_size()[0]
    HE = CziFile(HE_fname)
    IF = CziFile(IF_fname)
    
    # Scalefactor for HE processing
    scale_factor = pixsize/HE_seg_pixsize
    
    HE_img = HE.read_mosaic(C=0, scale_factor=scale_factor)
    HE_img = np.flip(HE_img, axis=0)
    
    # Segment HE image
    model = os.path.join('.', 'classifiers', 'bst_model256_0.7826.bin')
    cfg = os.path.join('.', 'classifiers', 'params.yaml')
    
    output_file = os.path.join('.', 'ImgData', 'Segmented_HE.tif')
    segmented_HE = segment(HE_fname, output_file, cfg, model)
    
    # resize segmented HE and save