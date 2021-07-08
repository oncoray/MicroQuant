# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:49:45 2021

@author: johan
"""

import yaml
import torch
import h5py
import numpy as np
import os
import subprocess
from  skimage import restoration, filters
import dask.array as da


from aicspylibczi import CziFile
import segmentation_models_pytorch as smp
import tifffile as tf

from Utils.HE_seg import InferenceDataset

def register(moving, fixed, trafo_params):
    
    moving = tf.imread(moving)
    fixed = tf.imread(fixed)

def czi2hdf5(path, **kwargs):
    """
    
    Function to convert a czi immunofluorescent image into a respective 
    hdf5 image in place.

    Parameters
    ----------
    path : czi immunofluorescent filename
        Filename of an input czi image.

    Returns
    -------
    Filename of new hdf5 file

    """
    
    replace = kwargs.get('replace', False)
    
    directory = os.path.dirname(path)
    fname = os.path.basename(path).split('.')[0]
    
    f_hdf5 = kwargs.get('fname_out', os.path.join(directory, fname+'.h5'))
    
    if os.path.exists(f_hdf5):
        if replace:
            os.remove(f_hdf5)
        else:
            return f_hdf5
    
    IF = CziFile(path)
    
    # create file
    C0 = IF.read_mosaic(C=0)
    C1 = IF.read_mosaic(C=1)
    C2 = IF.read_mosaic(C=2)
    
    # C0 = da.from_array(C0[0], chunks=50)
    # C1 = da.from_array(C1[0], chunks=50)
    # C2 = da.from_array(C2[0], chunks=50)
    
    # subtract background
    C0 = C0 - filters.gaussian(C0[0], sigma=50)
    C1 = C1 - filters.gaussian(C1[0], sigma=50)
    C2 = C2 - filters.gaussian(C2[0], sigma=50)
    
    # # normalize
    C0 = 2**16*(C0 - C0.min())/(C0.max() - C0.min())
    C1 = 2**16*(C1 - C1.min())/(C1.max() - C1.min())
    C2 = 2**16*(C2 - C2.min())/(C2.max() - C2.min())
    
    IF = np.vstack([C0, C1, C2]).transpose((1,2,0))[None, None, :]
    
    
    f = h5py.File(f_hdf5, 'w')
    f.create_dataset('data', shape=IF.shape, dtype='uint16', data=IF, chunks=True)
    f.close()
    
    return f_hdf5

def segment_HE(fname_in, fname_config, fname_model, **kwargs):
    """
    Runs Unet-based segmentation of HE image

    Parameters
    ----------
    fname_in : str
        File location of input HE file (must be czi)
    fname_config : str
        File location of .yml configuration file for Unet
    fname_model : str
        File location of bin Unet model for segmentation_models.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    prediction : ndarray
        ndarray of segmented HE image

    """
    
    fname_out = kwargs.get('fname_out', None)
    STRIDE = kwargs.get('stride', 16)
    MAX_OFFSET = kwargs.get('max_offset', 64)
    N_OFFSETS = kwargs.get('n_offsets', 10)
    
    # read config
    with open(fname_config, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)    
        
    IMG_SIZE = int(cfg['Input']['IMG_SIZE'])
    batch_size = int(cfg['Hyperparameters']['BATCH_SIZE'])
    PIX_SIZE = cfg['Input']['PIX_SIZE']
    N_CLASSES = cfg['Hyperparameters']['N_CLASSES']

    model = smp.Unet(
        encoder_name='resnet50', 
        encoder_weights='imagenet', 
        classes=N_CLASSES, 
        activation=None,
    )
    
    if not torch.cuda.is_available():
        DEVICE = 'cpu'
        model.load_state_dict(torch.load(fname_model, map_location=torch.device(DEVICE))['model_state_dict'])
    else:
        DEVICE = 'cuda'
        model.load_state_dict(torch.load(fname_model)['model_state_dict'])
    
    
    model = model.to(DEVICE)
    model.eval()
        
    ds = InferenceDataset(fname_in,
                          patch_size=IMG_SIZE,
                          stride=STRIDE,
                          augmentation=None,
                          target_pixsize=PIX_SIZE,
                          max_offset=MAX_OFFSET,
                          n_offsets=N_OFFSETS)
    ds.predict(model, batch_size=batch_size, device=DEVICE)
    prediction = ds.postprocess(filename=fname_out)
    
    return prediction, fname_out

def segment_IF(in_fname, classifier, **kwargs):
    """
    Calls Segmentation with Ilastik on an input image and returns the labelled image
    
    Parameters
    ----------
    fname_in : str
        Filename of input image (hdf5 or tif).
    classifier : str
        File loocation of ilastik classifier file.
    ilastik : str
        location oof Ilastik executable. Default: C:\Program Files\ilastik-1.3.3post3\ilastik.exe

    Returns
    -------
    numpy ndarray
    """
    
    ilastik_exe = kwargs.get('ilastik_exe', r'C:\Program Files\ilastik-1.3.3post3\ilastik.exe')
    in_remove = kwargs.get('in_remove', True)
    
    # folder for input data
    in_folder = os.path.dirname(in_fname)
    in_file = os.path.basename(in_fname).split('.')[0]
    out_file = os.path.join(in_folder, in_file + '_Simple_Segmentation.tif')
    classifier = os.path.abspath(classifier)
    
    # generate some output file name
    curdir = os.getcwd()
    os.chdir('C:')
    out = subprocess.check_output(' '.join([
            ilastik_exe,
            r'--headless',
            r'--project="{:s}"'.format(classifier),
            r'--export_source="Simple Segmentation"',
            r'--raw_data="{:s}"'.format(in_fname),
            r'--output_format=tif',
            r'--output_filename_format="{:s}"'.format(out_file)
            ]))
    os.chdir(curdir)
    if in_remove:
        os.remove(in_fname)
        
    return out_file
        