# -*- coding: utf-8 -*-
import numpy as np

import aicspylibczi
import aicsimageio
from skimage import transform
import tifffile as tf

import torch
from torch.utils.data import DataLoader

import tqdm


class HEImageDataset():
    def __init__(self, filename, n_classes, **kwargs):

        self.target_pixsize = kwargs.get('target_pixsize', 2.5)
        self.patch_size = kwargs.get('patch_size', 128)
        self.stride = kwargs.get('stride', 16)
        self.density = kwargs.get('density', 32)
        self.augmentation = kwargs.get('augmentation', None)
        self.batch_size = kwargs.get('batch_size', 20)
        self.device = kwargs.get('device', 'cuda')

        self.c_order =kwargs.get('c_order', 'RGB')

        print(f'\tPixel size: {self.target_pixsize}')
        print(f'\tPatch size: {self.patch_size}')
        print(f'\tbatch size: {self.batch_size}')

        self.filename = filename
        self.n_classes = n_classes

        self.read_czi(filename)

        # transpose if necessary
        if self.image.shape[-1] == 3:
            self.image = self.image.transpose((2, 0, 1))

        self.prediction = np.zeros((self.n_classes, self.image.shape[1], self.image.shape[2]),
                                   dtype='float32')

        # assign indeces on 2d grid and pad to prevent index errors
        self.create_samplelocations()

    def read_czi(self, filename):

        # Read czi image
        self.Image = aicsimageio.AICSImage(filename)
        self.czi = aicspylibczi.CziFile(filename)

        self.resolution = self.Image.physical_pixel_sizes.X/10

        if self.Image.dims.C == 1:
            self.image = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0]
            self.image = self.image.transpose((2,0,1))  # bring image to order CXY

        else:
            C0 = self.czi.read_mosaic(C = 0, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            C1 = self.czi.read_mosaic(C = 1, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            C2 = self.czi.read_mosaic(C = 2, scale_factor=self.resolution/self.target_pixsize)[0][None, :, :]
            self.image = np.vstack([C0, C1, C2])

        # re-arrange color axis if image is BGR and not RGB
        if self.c_order != 'BGR':
            self.image = self.image[::-1]
        elif self.c_order == 'RGB':
            pass

    def create_samplelocations(self):
        """
        Prepares a list of locations where the image will be fed forward
        through the network
        """

        # create sampling locations, omit half-tile sized margin at image edge
        X = np.arange(self.patch_size//2, self.image.shape[1] - self.patch_size//2, self.density)
        Y = np.arange(self.patch_size//2, self.image.shape[2] - self.patch_size//2, self.density)

        self.locations = []
        self.blacklisted_locations = []
        for x in tqdm.tqdm(X, desc='\tBrowsing image...'):
            for y in Y:
                patch = self.image[:,
                                   x - self.patch_size//2 : x + self.patch_size//2,
                                   y - self.patch_size//2 : y + self.patch_size//2]
                if np.sum(patch.sum(axis=0) == 0) > 150 or np.sum(patch.sum(axis=0) >= 3*np.iinfo(patch.dtype).max) >=150:
                    self.blacklisted_locations.append([x,y])
                    continue
                self.locations.append([x, y])

        np.random.shuffle(self.locations)


    def __len__(self):
        return len(self.locations)

    def __getitem__(self, key):

        loc = self.locations[key]
        x, y = loc[0], loc[1]

        patch = self.image[:,
                           x - self.patch_size//2 : x + self.patch_size//2,
                           y - self.patch_size//2 : y + self.patch_size//2].astype(float)

        return {'image': torch.from_numpy(patch)}

    def __setitem__(self, key, value):

        # crop center of processed tile
        patch = value[:,
                      self.stride : - self.stride,
                      self.stride : - self.stride]

        size = patch.shape[1]
        x, y = self.locations[key]

        self.prediction[:,
                        x - size//2 : x + size//2,
                        y - size//2 : y + size//2] += patch

    def predict(self, model):

        with torch.no_grad():
            dataloader = DataLoader(self, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)
            tk0 = tqdm.tqdm(dataloader, total=len(dataloader), desc='\tTilewise forward segmentation...')
            for b_idx, data in enumerate(tk0):

                data['image'] = data['image'].to(self.device).float()
                data['prediction'] = model(data['image'])
                out = torch.sigmoid(data['prediction']).detach().cpu().numpy()

                # iteratively write results from batches to prediction map
                for idx in range(out.shape[0]):
                    self[b_idx*self.batch_size + idx] = out[idx]

        # set blacklisted tiles to correct background
        size = self.patch_size
        for loc in self.blacklisted_locations:
            x, y = loc[0], loc[1]

            self.prediction[0,
                x - size//2 : x + size//2,
                y - size//2 : y + size//2] = np.ones((size, size))


    def export(self, filename, **kwargs):
        """
        Export prediction map to file with deflation compression
        """

        softmax = kwargs.get('softmax', True)
        upscale = kwargs.get('upscale', True)
        export = kwargs.get('export', True)

        if upscale:
            print('\t---> Upscaling')
            self.prediction = self.prediction.transpose((1,2,0))
            factor = self.Image.dims.X/self.prediction.shape[0]

            # prediction = torch.from_numpy(self.prediction)
            # prediction.to(self.device)
            # upsampler = torch.nn.Upsample(size=[self.Image.dims.Y,
            #                                     self.Image.dims.X,
            #                                     np.min(self.prediction.shape)])
            # self.prediction = upsampler(prediction).detach().cpu().numpy()
            self.prediction = transform.rescale(self.prediction, scale=(factor, factor, 1.0))

        if softmax:
            self.prediction = np.argmax(self.prediction, axis=-1)

        if export:
            tf.imwrite(filename, self.prediction.astype('uint8'))
            self.foutput = filename
