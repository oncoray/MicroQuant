# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:15:38 2021

@author: johan
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tifffile as tf
from scipy import ndimage
from skimage.measure import regionprops, label

import napari

# Use GPU for processing
import pyclesperanto_prototype as cle
cle.select_device()

    
def fractional_value(MapA, MapB):
    return np.sum(np.multiply(MapA, MapB))/np.sum(MapB)


def distribution_moments(Image, Mask, **kwargs):
    
    "Returns histogram descriptors for Pixel values in an Image masked with a Mask (bool)"

    assert Mask.dtype == bool
    
    Prefix = kwargs.get('prefix', '')
    nbins = kwargs.get('nbins', 100)
    
    Results = dict()
    
    values = Image[Mask == True].flatten()    
    Results[Prefix + "_Median"] = np.median(values)
    Results[Prefix + "_Mean"] = values.mean()
    Results[Prefix + "_std"] = values.std()
    Results[Prefix + "_V10"] = np.quantile(values, 0.1)
    Results[Prefix + "_V25"] = np.quantile(values, 0.25)
    Results[Prefix + "_V75"] = np.quantile(values, 0.75)
    Results[Prefix + "_V95"] = np.quantile(values, 0.95)
    
    Histogram = dict()
    Histogram['hist'], Histogram['edges'] = np.histogram(values, nbins)
    
    return Results, Histogram
    

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

def hist_to_file(hist, edges, file):
    
    df = pd.DataFrame()
    df['Frequency'] = hist
    df['Edges'] = edges[:-1] + np.diff(edges).mean()/2.0  # save centers of histogram bars instead of edges
    
    df.to_csv(file)
    
    return 1

def make_tumor_mask(HE):
    "Generates a binary mask of tumor areas (everything included)"
    
    print('\t---> Generating masks')
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
    
def measure_vessel_sizes(label_img, **kwargs):
    
    pxsize = kwargs.get('pxsize', 0.44)
    
    label_img = label(label_img)
    regions = regionprops(label_img)
    
    Vessel_minor_radii = [prop.minor_axis_length * pxsize for prop in regions]
    # Vessel_major_radii = [prop.major_axis_length * pxsize for prop in regions]
    Vessel_area = [prop.area * pxsize for prop in regions]
    
    Vessel_area = np.histogram(Vessel_area)
    Vessel_radii = np.histogram(Vessel_minor_radii)
    
    return Vessel_area, Vessel_radii
        
    
def pretty_picture(R, G, B, label):
    "Generate a prettyfied image for visualization"
    
    
    image = np.vstack([R[None, :, :],
                       G[None, :, :],
                       B[None, :, :]]).astype(int)
    image = image.transpose((1,2,0))
    
    
    plt.imshow(image)
    plt.imshow(label, cmap='gray', alpha=0.5)
    

def measure(segmented_HE, segmented_IF, **kwargs):
    "Measured defined features from provided input images HE and IF"
    
    pxsize = kwargs.get('pxsize', 0.44)
    HE_image = kwargs.get('HE_root', '')
    IF_image = kwargs.get('IF_root', '')
    directory = kwargs.get('dir', '')
    
    res_dir = os.path.dirname(segmented_HE)
    
    HE = tf.imread(segmented_HE)
    IF = tf.imread(segmented_IF)
    
    Tumor, Vital, SMA = make_tumor_mask(HE)
    Hypoxia, CD31, Perfusion = make_IF_mask(np.multiply(Vital, IF))
    
    # Measure HE-related params
    meas = dict()
    meas['dir'] = directory
    meas['HE_input'] = HE_image
    meas['IF_input'] = IF_image
    
    meas['Necrotic_fraction'] = 1 -fractional_value(Vital, Tumor)
    meas['SMA_fraction'] = fractional_value(SMA, Tumor)
    
    # Measure IF-related params
    meas['Hypoxic_fraction'] = fractional_value(Hypoxia, Vital)
    meas['Vascular_fraction'] = fractional_value(CD31, Vital)
    
    # Vessel sizes
    Vessel_area, Vessel_radii = measure_vessel_sizes(CD31)
    hist_to_file(Vessel_area[0], Vessel_area[1], os.path.join(res_dir, 'Vessel_area_hist.csv'))
    hist_to_file(Vessel_radii[0], Vessel_radii[1], os.path.join(res_dir, 'Vessel_radii_hist.csv'))
    
    
    # Distance related features: Vessels
    EDT = ndimage.morphology.distance_transform_edt(np.invert(CD31)) * pxsize  # get EDT
    EDT_fts, EDT_hist = distribution_moments(EDT, Vital, prefix='EDT_CD31')
    meas.update(EDT_fts)    
    
    # Hypoxic-distance (HyDi) related features
    HyDi_fts, HyDi_hist = distribution_moments(np.multiply(EDT, Hypoxia), Vital, prefix='EDT_Hypoxia')
    
    # Save features
    df = pd.DataFrame(meas, index=[0])
    df.to_csv(os.path.join(res_dir, 'Features.csv'))
    
    hist_to_file(EDT_hist['hist'], EDT_hist['edges'], os.path.join(res_dir, 'EDT_hist.csv'))
    hist_to_file(HyDi_hist['hist'], HyDi_hist['edges'], os.path.join(res_dir, 'HyDi_hist.csv'))
    
    pretty_picture(CD31, Hypoxia, Perfusion, 0.5*np.multiply(Tumor, np.invert(Vital)))
    
if __name__ == '__main__':
    result = measure(r'C:\Users\johan\Desktop\MQ\Test_dir\SampleB\3_res\HE_seg.tif',
                     r'C:\Users\johan\Desktop\MQ\Test_dir\SampleB\3_res\IF_transformed.tif')