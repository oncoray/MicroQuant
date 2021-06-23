
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:36:40 2021

@author: johan
"""



import napari
from aicsimageio import AICSImage

with napari.gui_qt():
    
    img = AICSImage(r'D:\Documents\Promotion\Projects\2021_MicroQuant\ImgData\N195e_SAS_28_200302-Scene-2-ScanRegion1_HE.czi')
    napari.view_image(img.data)  # in memory