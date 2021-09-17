# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 16:04:09 2021

@author: johan
"""

import os
import numpy as np
import tqdm
import tifffile as tf
from subprocess import check_output

import cv2
from scipy import ndimage
from skimage import measure
from skimage import morphology
from skimage.measure import label, regionprops, regionprops_table

import shutil
import matplotlib.pyplot as plt

# Use GPU for processing
import pyclesperanto_prototype as cle
cle.select_device("MX230")


def register_and_transform(Moving_image_path, Target_image_path, **kwargs):
    "Register two images (moving and target)"
    
    elastix_exe         = kwargs.get('elastix_exe', os.path.join(os.getcwd(), 'registration', 'elastix-5.0.1-win64', 'elastix.exe'))
    elastix_params      = kwargs.get('elastix_params', os.path.join(os.getcwd(), 'registration', 'elastix_parameters.txt'))
    do_shape_adjustment = kwargs.get('shape_adjustment', True)
    compress_output     = kwargs.get('compress_output', False)
    
    reg_dir = os.path.abspath(os.path.join(os.path.dirname(Moving_image_path), '..', '2_reg'))

    if do_shape_adjustment:
        f = shape_adjustment(Moving_image_path, Target_image_path)
    else:
        f = 1.0
        shutil.copy(Moving_image_path, os.path.join(reg_dir, 'Moving_raw.tif'))
        shutil.copy(Target_image_path, os.path.join(reg_dir, 'Fixed_raw.tif'))
        
    # use shape-adjusted images as registration input
    Moving_image_path = os.path.join(reg_dir, 'Moving_raw.tif')
    Target_image_path = os.path.join(reg_dir, 'Fixed_raw.tif')

    # Read data
    Moving = tf.imread(Moving_image_path)
    Fixed = tf.imread(Target_image_path)
    
    # binarize segmented images
    Moving = simplify_labels(Moving, background_label=1)
    Fixed = simplify_labels(Fixed, background_label=0)
    
    # get new filenames
    fMoving = os.path.join(os.path.dirname(Moving_image_path), '..', '2_reg', 'Moving_mask.tif')
    fFixed = os.path.join(os.path.dirname(Moving_image_path), '..', '2_reg', 'Fixed_mask.tif')
    
    tf.imsave(fMoving, Moving.astype(np.uint8))
    tf.imsave(fFixed, Fixed.astype(np.uint8))
    
    # Run Elastix
    print('\t---> Running Elastix...', end='')
    check_output([
        elastix_exe,
        '-m', fMoving,
        '-f', fFixed,
        '-p', elastix_params,
        '-out', os.path.dirname(Target_image_path)])
    print('Done')
    
    # Run Transformix
    print('\t---> Running Transformix...', end='')
    check_output([
        elastix_exe.replace('elastix.exe', 'transformix.exe'),
        '-in', Moving_image_path,
        '-tp', os.path.join(reg_dir, 'TransformParameters.0.txt'),
        '-out', reg_dir])
    print('Done')
    
    # Change data type of output to uint8
    tf.imwrite(os.path.join(reg_dir, 'result.tif'),
               tf.imread(os.path.join(reg_dir, 'result.tif')).astype(np.uint8))
    
    # Rename and move transformed image to measurement directory
    meas_dir = reg_dir.replace('2_reg', '3_res')
    shutil.copy(
        os.path.join(reg_dir, 'result.tif'),
        os.path.join(meas_dir, 'IF_transformed.tif')
        )
    
    shutil.copy(
        os.path.join(reg_dir, 'Fixed_raw.tif'),
        os.path.join(meas_dir, 'HE_seg.tif')
        )
    
    # Replace output with png files to save storage
    if compress_output:
        convert_to_png([
            Moving_image_path,
            Target_image_path,
            fMoving,
            fFixed,
            os.path.join(reg_dir, 'result.tif')])
    
    return os.path.join(meas_dir, 'IF_transformed.tif'), os.path.join(meas_dir, 'HE_seg.tif')
    
    

def convert_to_png(list_of_images):
    "Converts a list of files to (compressed) png images"
    
    for file in tqdm.tqdm(list_of_images, desc='\t\tConverting output to png...'):
        image = tf.imread(file)
        cv2.imwrite(file.replace('tif', 'png'), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        os.remove(file)

def shape_adjustment(ImageA, ImageB, **kwargs):
    "Compare shapes of two images and adjust sizes approxximately"
    
    reg_dir = kwargs.get('reg_dir', os.path.abspath(os.path.join(os.path.dirname(ImageA), '..', '2_reg')))
    
    assert os.path.exists(ImageA)
    assert os.path.exists(ImageB)
    
    ImageA = tf.imread(ImageA)
    ImageB = tf.imread(ImageB)
    
    f = ImageB.shape[0] / ImageA.shape[0]
    outsize = tuple((np.asarray(ImageA.shape) * f).astype(np.uint16)[::-1])
    
    ImageA = cv2.resize(ImageA, dsize=outsize, interpolation=cv2.INTER_NEAREST)
    
    tf.imsave(os.path.join(reg_dir, 'Moving_raw.tif'), ImageA)
    tf.imsave(os.path.join(reg_dir, 'Fixed_raw.tif'), ImageB)
    
    return f
    
    

def simplify_labels(label_image, **kwargs):
    "Filters background spots away from label image"

    background_label = kwargs.get('background_label', 1)
    smoothing_iterations = kwargs.get('n_smoothing', 8)
    
    binary = np.zeros_like(label_image, dtype='uint8')
    binary[label_image != background_label] = 1
    
    # Use regionprops to find largest connected area
    label_img = label(binary)
    regions = regionprops(label_img)
    
    # find largest blob (= tissue section)
    largest = 0
    index = 0
    for props in tqdm.tqdm(regions, desc='\t\tClearing background...'):
        if props.area > largest:
            largest = props.area
            index = props.label
            
    binary[label_img != index] = 0
    binary = np.invert(binary)
    
    # find largest blo (= background) - all other islands/blobs are holes
    label_img = label(binary)
    regions = regionprops(label_img)
    
    largest = 0
    index = 0
    for props in tqdm.tqdm(regions, desc='\t\tFilling holes...'):
        if props.area > largest:
            largest = props.area
            index = props.label
            
    binary[label_img != index] = 0
    binary = np.invert(binary).astype(np.uint8)
    
    
    # input = cle.push(binary)
    # tmp = cle.create_binary_like(input)
    
    # # Erosion
    # tk0 = tqdm.tqdm(range(smoothing_iterations), desc='\t--->Doing preprocessing 1/2')
    # for i in tk0:
    #     cle.erode_labels(input, tmp)
    #     cle.erode_labels(tmp, input)
            
    # # get largest connected component
    # binary = cle.pull(input)
    # labels = measure.label(binary)
    # assert labels.max() != 0
    # binary = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    
    # # Fill holes
    # binary = ndimage.binary_fill_holes(binary)
    
    # # Dilation
    # input = cle.push(binary)
    # tk0 = tqdm.tqdm(range(smoothing_iterations), desc='\t---> Doing preprocessing 2/2')
    # for i in tk0:
    #     cle.dilate_labels(input, tmp)
    #     cle.dilate_labels(tmp, input)
    
    # return cle.pull(input)
    return binary


if __name__ == '__main__':
    register_and_transform(
        Moving_image_path=r'C:\Users\johan\Desktop\MQ\Test_dir\SampleB\1_seg\IF_seg.tif',
        Target_image_path=r'C:\Users\johan\Desktop\MQ\Test_dir\SampleB\1_seg\HE_seg_Unet.tif',
        elastix_exe = r'D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\registration\elastix-5.0.1-win64\elastix.exe',
        elastix_params = r'D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\registration\elastix_parameters.txt'
        )