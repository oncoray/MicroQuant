# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 16:04:09 2021

@author: johan
"""

import os
import numpy as np
from skimage import morphology, measure
import tqdm
import tifffile as tf
from subprocess import check_output



def register(Moving_image_path, Target_image_path, **kwargs):
    "Register two images (moving and target)"
    
    elastix_exe = os.path.join(os.getcwd(), 'registration', 'elastix-5.0.1-win64', 'elastix.exe')
    elastix_params = os.path.join(os.getcwd(), 'registration', 'elastix_parameters.txt')

    Moving = tf.imread(Moving_image_path)
    Fixed = tf.imread(Target_image_path)
    
    Moving = simplify_labels(Moving, background_label=1)
    Fixed = simplify_labels(Fixed, background_label=0)
    
    fMoving = os.path.join(os.path.dirname(Moving_image_path), '..', '2_reg', 'Moving.tif')
    fFixed = os.path.join(os.path.dirname(Moving_image_path), '..', '2_reg', 'Fixed.tif')
    
    tf.imsave(fMoving, Moving)
    tf.imsave(fFixed, Fixed)
    
    check_output([
        elastix_exe,
        '-m', fMoving,
        '-f', fFixed,
        -'p', elastix_params,
        '-out', os.path.dirname(Target_image_path)])
    

def simplify_labels(label_image, **kwargs):
    "Filters background spots away from label image"
    
    background_label = kwargs.get('background_label', 1)
    smoothing_iterations = kwargs.get('n_smoothing', 64)
    
    binary = np.zeros_like(label_image, dtype='uint8')
    binary[label_image != background_label] = 1
    
    tk0 = tqdm.tqdm(range(smoothing_iterations), desc='\t--->Doing postprocessing 1/2')
    
    # Erosion
    for i in tk0:
        binary = morphology.erosion(binary)
        tk0.set_postfix(Erosion=i)
            
    # get largest connected component
    labels = measure.label(binary)
    assert labels.max() != 0
    binary = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    
    # Dilation
    tk0 = tqdm.tqdm(range(smoothing_iterations), desc='\t--->Doing postprocessing 2/2')
    for i in tk0:
        binary = morphology.dilation(binary)
        tk0.set_postfix(dilation=i)
        
    return binary