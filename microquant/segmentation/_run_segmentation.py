# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:42:34 2021

@author: johan
"""

import os
import cv2
import numpy as np

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
import segmentation_models_pytorch as smp
import torch


def segment_he(input_file: str,
               model_file: str,
               n_classes: int,
               pxsize: float,
               output_file: str = None,
               device: str = 'cuda',
               stride: int = 16,
               batch_size: int = 16,
               tile_size: int = 256,
               **kwargs) -> np.ndarray:
    """
    Apply a trained segmentation model to an input H&E image.

    Parameters
    ----------
    input_file : str
        Path to image to be segmented
    model_file : str
        Path to torch model file to be used
    n_classes : int
        Number of distinct pixel classes present in image and known to segmentation
    pxsize : float
        Pixel size with which the model has been trained
    device : str, optional
        Used processing platform. The default is 'cuda', but can be switched
        to 'cpu', if no cuda-enabled GPU is available
    stride : int, optional
        Margin of a single tile should be discarded (in pixels).
        The default is 16.
    batch_size : int, optional
        Number of iages that are fed through the GPU simultaneously.
        The default is 16.
    tile_size : int, optional
        Size of one image tile (in pixels). The default is 256.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    from ._segment_HE import HEImageDataset

    # Get path of input image
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input image {input_file} not found!')

    if output_file is None:
        output = os.path.join(os.path.dirname(input_file),
                              '1_seg', 'HE_seg_Unet.tif')
    # batch_size  = kwargs.get('batch_size', MQmodel.hyperparameters_training['BATCH_SIZE'])
    # patch_size    = kwargs.get('patch_size', MQmodel.params['Input']['IMG_SIZE'])
    # pxsize      = kwargs.get('pxsize', MQmodel.params['Input']['PIX_SIZE'])
    # n_classes   = kwargs.get('n_classes', MQmodel.hyperparameters_training['N_CLASSES'])

    # Create Unet
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        classes=n_classes,
        activation=None,
    )

    # Load weights and set model to eval mode
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    model = model.to(device)
    model.eval()

    aug_forw = A.Compose([
        PadIfNeeded(min_width=tile_size, min_height=tile_size,
                    border_mode=cv2.BORDER_REFLECT)
    ])

    # Create Dataset for loading
    ds = HEImageDataset(input_file,
                        n_classes=n_classes,
                        patch_size=tile_size,
                        stride=stride,
                        augmentation=aug_forw,
                        target_pixsize=pxsize,
                        batch_size=batch_size)

    # run and export prediction
    ds.predict(model)
    ds.export(output, upscale=True, softmax=True)

    return ds.foutput

def segment_CD31_Pimo_Hoechst(MQJob, MQmodel, **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.
    MQmodel : MicroQuant model type
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from ._segment_IF import IFImageDataset

    # Get input image
    ImgPath=os.path.join(MQJob.sample.dir, MQJob.sample.IF_img)

    # Processing settings
    pxsize      = kwargs.get('pxsize', MQmodel.params['Input']['PIX_SIZE'])
    n_classes   = kwargs.get('n_classes', MQmodel.hyperparameters_training['N_CLASSES'])
    classifier = kwargs.get('classifier', MQmodel.file_model)

    out_file = os.path.join(os.path.dirname(ImgPath), '1_seg', 'IF_seg.tif')

    ds = IFImageDataset(ImgPath, classifier, use_dask=False)

    ds.predict()
    ds.export(out_file)

    return ds.foutput
