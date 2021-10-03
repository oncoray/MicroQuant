# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:45:42 2021

@author: johan
"""

import os
import numpy as np
import pandas as pd
import tqdm
import shutil
from skimage import morphology
from skimage import measure
from skimage import filters

import pyclesperanto_prototype as cle
# initialize GPU
cle.select_device("MX230")
print("Used GPU: " + cle.get_device().name)

def find_Ilastik():
    
    " Find Ilastik in local installation files"
    
    root = r'C:\Program Files'
    f = None
    
    filelist = os.listdir(root)
    for file in filelist:
        if 'ilastik' in file:
            f = os.path.join(root, file, 'ilastik.exe')
            break
            
    return f

def create_file_structure(df, **kwargs):
    """
    Creates MicroQuant output directories

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with names and basedir of every HE-IF image pair

    Returns
    -------
    
    dictionary:
        dictionary with output subdirectories as keys.
    """
    
    overwrite = kwargs.get('overwrite', False)
    
    print(f'\t--> Creating directories for {len(df)} samples')
    for i, sample in df.iterrows():
        
        dirs = {
            'seg': os.path.join(sample.dir, '1_seg'),
            'reg': os.path.join(sample.dir, '2_reg'),
            'res': os.path.join(sample.dir, '3_res')
            }
        
        for key in dirs.keys():
            
            # if overwrite is set, delete old files and create new (empty) directories
            if overwrite:
                try:
                    shutil.rmtree(dirs[key])
                except:
                    pass
            
            os.mkdir(dirs[key])
        
    return dirs


def normalize(image, **kwargs):
    "Normalizes an image to quantile grayvalue range"    
    
    lower = kwargs.get('lower', {0: 0.2, 1: 0.05, 2:0.05})
    upper = kwargs.get('upper', {0: 0.99, 1: 0.98, 2:0.98})
    
    if image.shape[0] == np.min(image.shape):
        image = image.transpose((1,2,0))
    
    for i in tqdm.tqdm(range(3), desc='Processing background and normalizing...'):
        channel = image[:, :, i]
        
        input = cle.push(channel)
        background = cle.push(np.zeros_like(channel))
        
        cle.gaussian_blur(input,background, 25, 25)
        background = cle.pull(background)
        
        channel = channel - background
        channel[channel <0] = 0
        
        flat = channel[channel != 0].flatten()
        l = np.quantile(flat, q = lower[i])
        u = np.quantile(flat, q = upper[i])
        
        # normalize to 0-255
        channel = 255 * (channel- l)/(u - l)
        channel[channel < 0] = 0  # move negative values to zero
        channel[channel > 255] = 255 # move values >255 to 255
        
        image[:, :, i] = channel

    return image.astype(np.uint8)



def is_derivative_img(filename):
    
    exclude_strs = ['_transformed', 'Simple_Segmentation', 'DL', 'seg']
    return any([True if item in filename else False for item in exclude_strs])



def is_img_file(file, **kwargs):
    
    """
    Checks if a filename is an image file according to its file ending.
    
    Parameters
    ----------
    
    **kwargs : optional arguments
        - file_types: list
            List of strings which correspond to file types that should be
            recognized as acceptable image formats. Default: 'czi'
    
    """
    
    file_types = kwargs.get('file_types', ['czi'])
    checks = []
    
    for ftype in file_types:
        if ftype in file:
            checks.append(True)
        else:
            checks.append(False)
            
    return any(checks)

def check_data(df, **kwargs):
    """
    Browses through an input dataframe and examines every entry for exitance

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe with columns 'dir', 'HE_img' and 'IF_img'
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    pandas dataframe
    """
    
    check = []
    tk0 = tqdm.tqdm(df.iterrows(), desc='Checking inputs...')
    for i, sample in tk0:
        HE = os.path.join(sample.dir, sample.HE_img)
        IF = os.path.join(sample.dir, sample.IF_img)
        
        
        if os.path.exists(HE) and os.path.exists(IF):
            check.append(True)
        else:
            check.append(False)
            

    # add results to dataframe
    df.loc[:, ('exists')] = check
    print('')
    print(f'\t---> {np.sum(check)}/{len(check)} samples exist.')
    
    df = df[df.exists == True]
    
    return df
    
        
    
    

def browse_data(directory, **kwargs):
    """
    
    Browses a file structure for co-occurrences of HE and IF images.
    The paired IF and HE images are saved to a dataframe, that is then exported
    to a csv file.
    
    The resulting csv file serves as input for MicroQuant.

    Parameters
    ----------
    directory : str
        Path name of top-level path containing all IF and HE images
    outdir : str
        Directory where the overivew file should be stored
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    pandas dataframe with columns 'dir', 'HE_img' and 'IF_img'
    """
    
    separate_dirs = kwargs.get('separate_dirs', True)
    IF_identifier = kwargs.get('IF_identifier', 'IF')
    HE_identifier = kwargs.get('IF_identifier', 'HE')
    outdir = kwargs.get('outdir', None)    
    
    HE_imgs = pd.DataFrame(None, columns = ['dir', 'HE_img'])
    IF_imgs = pd.DataFrame(None, columns = ['dir', 'IF_img'])
    
    if separate_dirs:
        for root, subdirs, fnames in tqdm.tqdm(os.walk(directory), desc='Browsing files...'):
            
            # find all files that match HE identifier
            for fname in fnames:
                
                # if HE was found in dir:
                if HE_identifier in fname:
                    
                    # test for exclude strings
                    if is_derivative_img(fname):
                        continue
                    
                    # test for file types
                    if not is_img_file(fname):
                        continue
                    
                    _HE = pd.DataFrame({'dir': [root], 'HE_img': [fname]})
                    HE_imgs = HE_imgs.append(_HE, ignore_index=False)
                    
                if IF_identifier in fname:
                    
                    # test for exclude strings
                    if is_derivative_img(fname):
                        continue
                    
                    # test for file types
                    if not is_img_file(fname):
                        continue
                    
                    _IF = pd.DataFrame({'dir': [root], 'IF_img': [fname]})
                    IF_imgs = IF_imgs.append(_IF, ignore_index=False)
                    
        df = HE_imgs.merge(IF_imgs, left_on='dir', right_on='dir')
        df = df.dropna()
        
        # export dataframe
        if outdir is not None:
            df.to_csv(outdir)
            
        return df