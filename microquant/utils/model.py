# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:13:34 2021

@author: johan
"""

import os
import yaml
import pandas as pd

class MQ_segmentation_model():
    
    def __init__(self):
        
        self.file_param = ''
        self.file_model = ''
        
        
    def load(self, directory):
        """
        loads an MQ model parameter file and returns model object for further use
    
        Parameters
        ----------
        directory : string
            Path to model parameter file
    
        Returns
        -------
        MQ model object
    
        """
        
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f'Directory {directory} not found.')
        else:
            self.directory = directory
        
        
        for _file in os.listdir(directory):
            if _file.endswith('yaml'):
                file = _file
                break
            
        files = os.listdir(directory)
        files.remove(file)  # remove yml file from filelist
    
        if not len(files) == 1:
            raise FileExistsError(f'Too many/Not enough files found in model directory {os.path.dirname(file)}')
    
        self.file_model = os.path.join(self.directory, files[0])
        self.file_param = os.path.join(self.directory, file)
            
        # read config
        with open(self.file_param, "r") as yamlfile:
            self.params = yaml.load(yamlfile, Loader=yaml.FullLoader)
            
            self.labels = self.params['Labels']
            self.hyperparameters_training = self.params['Hyperparameters']
            self.model = self.params['Model']['model']
            
            assert os.path.basename(self.model) == os.path.basename(self.file_model)
            self.model = self.file_model
            
        

class model():
    
    def __init__(self, f_param, f_model, **kwargs):
        
        
        
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
    
if __name__ == '__main__':
    model = MQ_segmentation_model()
    model.load(r'D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\models\IF\model_210810')