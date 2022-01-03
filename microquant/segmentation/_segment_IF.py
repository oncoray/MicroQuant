# -*- coding: utf-8 -*-
import tifffile as tf
from skimage import feature, future
from functools import partial

import aicspylibczi
import aicsimageio

import numpy as np
from utils.utils import normalize

import joblib
import tqdm


class IFImageDataset():
    def __init__(self, filename, classifier, **kwargs):

        self.sigma_min = kwargs.get('sigma_min', 4)
        self.sigma_max = kwargs.get('sigma_max', 16)
        self.use_dask = kwargs.get('use_dask', True)

        self.filename = filename
        self.classifier = classifier

        self.image = self.read_czi(self.filename)  # read czi
        self.clf = joblib.load(self.classifier)  # load RF classifier file

        self.features_func = partial(feature.multiscale_basic_features,
                                     intensity=True, edges=True, texture=True,
                                     sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                     multichannel=True, num_workers=4)

    def read_czi(self, filename):

        try:
        # Read czi image
            self.AICS = aicsimageio.AICSImage(filename)

            if self.use_dask:
                self.dask_image = self.AICS.get_dask_image_data("YXC")
            else:
                self.dask_image = self.AICS.get_image_data("YXC")

        # use pylibczi in case of error
        except ValueError:
            Image = aicspylibczi.CziFile(filename)
            C0 = Image.read_mosaic(C=0)
            C1 = Image.read_mosaic(C=1)
            C2 = Image.read_mosaic(C=2)

            self.dask_image = np.vstack([C0, C1, C2]).transpose((1,2,0))

        self.dask_image = normalize(self.dask_image)
        self.prediction = np.zeros_like(self.dask_image[:,:,0])  # allocate output


    def predict(self, ts=512):

        shape = self.dask_image.shape
        X = shape[0]//ts + 1
        Y = shape[1]//ts + 1

        tk0 = tqdm(np.arange(0, X, 1), desc='\tRandom forest-based segmentation')
        for x in tk0:
            for y in np.arange(0, Y, 1):

                try:
                    img = self.dask_image[x*ts:x*ts + ts,
                                          y*ts:y*ts + ts,
                                          :]
                    features = self.features_func(img)
                    self.prediction[x*ts:x*ts + ts,
                                    y*ts:y*ts + ts] = future.predict_segmenter(features, self.clf)
                    tk0.set_postfix(Processed_tiles='{:d}/{:d}'.format(y, Y))
                except Exception:
                    pass

    def export(self, filename):

        tf.imwrite(filename, self.prediction)
        self.foutput = filename
