# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:15:38 2021

@author: johan
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import tifffile as tf
from stsci import ndimage

# Use GPU for processing
import pyclesperanto_prototype as cle
cle.select_device()

    
def fractional_value(MapA, MapB):
    return np.sum(np.multiply(MapA, MapB))/np.sum(MapB)
    

def make_IF_mask(IF):
    
    labels = dict()
    labels['Background'] = 1
    labels['Hypoxia'] = 4
    labels['CD31'] = 3
    labels['Perfusion'] = 5
    
    # Hypoxia
    Hypoxia = np.zeros_like(IF, dtype=bool)
    Hypoxia[IF == labels['Hypoxia']] = True
    
    # CD31 
    CD31 = np.zeros_like(IF, dtype=bool)
    CD31[IF == labels['CD31']] = True
    
    # Perfusion
    Perfusion = np.zeros_like(IF, dtype=bool)
    Perfusion[IF == labels['Perfusion']] = True
    
    return Hypoxia, CD31, Perfusion

def make_tumor_mask(HE):
    "Generates a binary mask of tumor areas (everything included)"
    
    labels = dict()
    labels['Background'] = 0
    labels['Necrosis'] = 1
    labels['Vital'] = 2
    labels['SMA'] = 3 
    
    # Make Tumor Mask
    Tumor = np.zeros_like(HE, dtype=bool)
    Tumor[HE != labels['Background']] = True  # set foreground to True
    Tumor[HE == labels['SMA']] = False  # remove stroma tissue
    Tumor = ndimage.binary_fill_holes(Tumor)
    
    # Make Vital Mask
    Vital = np.zeros_like(HE, dtype=bool)
    Vital[HE == labels['Vital']] = True
    
    # Make SMA Mask
    SMA = np.zeros_like(HE, dtype=bool)
    SMA[HE == labels['SMA']] = True
    
    return Tumor, Vital, SMA
    

def measure(segmented_HE, segmented_IF):
    "Measured defined features from provided input images HE and IF"
    
    HE = tf.imread(segmented_HE)
    IF = tf.imread(segmented_IF)
    
    Tumor, Vital, SMA = make_tumor_mask(HE)
    Hypoxia, CD31, Perfusion = make_IF_mask(np.multiply(Vital, IF))
    
    # Measure HE-related params
    meas = dict()
    meas['Necrotic_fraction'] = 1 -fractional_value(Vital, Tumor)
    meas['SMA_fraction'] = fractional_value(SMA, Tumor)
    
    # Measure IF-related params
    meas['Hypoxic_fraction'] = fractional_value(Hypoxia, Vital)
    meas['Vascular_fraction'] = fractional_value(CD31, Vital)
    
    # Distance related features
    EDT = ndimage.morphology.distance_transform_edt(np.invert(CD31))
    
    return meas
        
    
    
if __name__ == '__main__':
    result = measure(r'C:\Users\johan\Desktop\MQ\ImgData\3_res\HE_seg.tif',
                     r'C:\Users\johan\Desktop\MQ\ImgData\3_res\IF_transformed.tif')