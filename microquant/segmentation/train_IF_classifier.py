# -*- coding: utf-8 -*-
"""

Train IF classifier with RF from existing segmentation
Created on Tue Aug 10 13:09:14 2021

@author: johan
"""


import tifffile as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

def sample_from_image(image, mask, size=256, n_tiles=32):
    
    x = np.random.randint(low=size//2, high=(image.shape[0]-size//2), size=n_tiles)
    y = np.random.randint(low=size//2, high=(image.shape[1]-size//2), size=n_tiles)
    
    sigma_min = 4
    sigma_max = 16
    
    features_func = partial(feature.multiscale_basic_features,
            intensity=True, edges=False, texture=True,
            sigma_min=sigma_min, sigma_max=sigma_max,
            multichannel=True)
    
    # create first tile from middle
    img_tile = image[image.shape[0]//2 - size//2: image.shape[0]//2 + size//2,
                     image.shape[1]//2 - size//2: image.shape[1]//2 + size//2,
                     :]
    labels = mask[image.shape[0]//2 - size//2: image.shape[0]//2 + size//2,
                  image.shape[1]//2 - size//2: image.shape[1]//2 + size//2]
    features = features_func(img_tile)
    
    for idx in range(n_tiles - 1):
        img_tile = image[x[idx] - size//2: x[idx] + size//2,
                         y[idx] - size//2: y[idx] + size//2,
                         :]
        label_tile = mask[x[idx] - size//2: x[idx] + size//2,
                          y[idx] - size//2: y[idx] + size//2]
        
        features = np.concatenate((features, features_func(img_tile)), 0)
        labels = np.concatenate((labels, label_tile), 0)
        
    return features, labels
    

if __name__ == "__main__":
    print('Here we go')
    plot = False

    root = r'C:\Users\johan\Desktop\IFsegmentation'
    
    image = os.path.join(root, '181212_N182f_SAS_29_1_IF_norm.tif')
    mask = os.path.join(root, '181212_N182f_SAS_29_1_IF_norm_Simple Segmentation.tif')
    
    image = tf.imread(image)
    mask = tf.imread(mask)
    
    features, labels = sample_from_image(image, mask)
    
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05, bootstrap=True)
    clf = future.fit_segmenter(labels, features, clf)
    result = future.predict_segmenter(features, clf)
    
    if plot:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(labels)
        ax[0].set_title('True Labels')
        ax[1].imshow(result)
        ax[1].set_title('Segmentation')
        fig.tight_layout()
    
    joblib.dump(clf, r"D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\models\IF\model_210810/sklearn_random_forest.joblib", compress=2)
    
    


    
    