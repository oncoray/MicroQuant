# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:42:34 2021

@author: johan
"""

import os
import cv2

import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
import segmentation_models_pytorch as smp
import torch


def segment_he(MQJob, MQmodel, **kwargs):
    """
    Parameters
    ----------
    ImgPath : str
        DESCRIPTION.

    Returns
    -------
    None.

    """

    from ._segment_HE import HEImageDataset

    # Get path of input image
    ImgPath = os.path.join(MQJob.sample.dir, MQJob.sample.HE_img)
    if not os.path.exists(ImgPath):
        raise FileNotFoundError(f'Input image {ImgPath} not found!')

    # select usable gpu
    device = MQJob.device

    # processing setting
    stride = kwargs.get('stride', 16)
    output = kwargs.get('output', os.path.join(os.path.dirname(ImgPath),
                                               '1_seg', 'HE_seg_Unet.tif'))
    batch_size  = kwargs.get('batch_size', MQmodel.hyperparameters_training['BATCH_SIZE'])
    patch_size    = kwargs.get('patch_size', MQmodel.params['Input']['IMG_SIZE'])
    pxsize      = kwargs.get('pxsize', MQmodel.params['Input']['PIX_SIZE'])
    n_classes   = kwargs.get('n_classes', MQmodel.hyperparameters_training['N_CLASSES'])

    # Create Unet
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        classes=n_classes,
        activation=None,
    )

    # Load weights and set model to eval mode
    model.load_state_dict(torch.load(MQmodel.file_model)['model_state_dict'])
    model = model.to(device)
    model.eval()

    aug_forw = A.Compose([
        PadIfNeeded(min_width=patch_size, min_height=patch_size,
                    border_mode=cv2.BORDER_REFLECT)
    ])

    # Create Dataset for loading
    ds = HEImageDataset(ImgPath,
                        n_classes=n_classes,
                        patch_size=patch_size,
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
