# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:13:34 2021

@author: johan
"""

import os


class model():
    
    def __init__(self, **kwargs):
        
        self.clas_type = kwargs.get('type', 'torch')
        self.path = kwargs.get('path', '')
        
        self.weights_file = None
        self.params_file = None
        self.ilp_classifier = None
        self.sklearn_IF_classifier = None
        
        self.load()
    
    def load(self):
        "FInd segmentation files (model and params) in a directory"
        
        files = os.listdir(self.path)
        
        for f in files:
            if f.endswith('.bin'):
                self.weights_file = os.path.join(self.path, f)
                
            elif f.endswith('.yaml'):
                self.params_file = os.path.join(self.path, f)
                
            elif f.endswith('ilp'):
                self.ilp_classifier = os.path.join(self.path, f)
                
            elif f.endswith('joblib'):
                self.sklearn_IF_classifier = os.path.join(self.path, f)
                
        # if self.params_file is None:
        #     raise FileNotFoundError('No parameter file found')
            
        # if self.weights_file is None:
        #     raise FileNotFoundError('No model file found')
            
            
                
    # def download()