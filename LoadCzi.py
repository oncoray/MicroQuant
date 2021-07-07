
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:36:40 2021

@author: johan
"""



import napari
from aicsimageio import AICSImage
import os

HE = AICSImage(os.path.join(os.getcwd(), 'ImgData', 'N195e_SAS_28_200302-Scene-2-ScanRegion1_HE.czi'))
    

viewer = napari.Viewer()
viewer.add_image(HE.dask_data)

napari.run()