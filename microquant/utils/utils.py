# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:45:42 2021

@author: johan
"""

import os
import numpy as np
import pandas as pd
import tqdm

def normalize(image, **kwargs):
    
    
    lower = kwargs.get('lower', {0: 0.2, 1: 0.05, 2:0.05})
    upper = kwargs.get('upper', {0: 0.99, 1: 0.98, 2:0.98})
    
    for i in range(3):
        channel = image[i, :, :]
        l = np.quantile(channel, q = lower[i])
        u = np.quantile(channel, q = upper[i])
        
        # normalize to 0-255
        channel = 255 * (channel - l)/(u - l)
        channel[channel < 0] = 0
        channel[channel > 255] = 255
        
        image[i, :, :] = channel

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