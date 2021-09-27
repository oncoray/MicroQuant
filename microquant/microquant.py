"""Main module."""

from segmentation import segmentation as seg
from registration import registration as reg
from measurement import features

from utils import utils
from utils.model import MQ_segmentation_model

import pandas as pd
import os

class MQ_job():
    def __init__(self, **kwargs):
        
        self.sample = kwargs.get('sample', None)
        
        # Outputs
        self.HE_segmented = ''
        self.IF_segmented = ''
        
        # segmentation
        print('##### New MQ Job created #####')
        print('\t--> Directory: {:s}'.format(self.sample.dir))
        print('\t--> HE: {:s}'.format(self.sample.HE_img))
        print('\t--> IF: {:s}'.format(self.sample.IF_img))
        
    def segment(self, which, model):
        
        print('')
        print('\t----SEGMENTATION----')
        if which == 'HE':
        
            self.HE_segmented = seg.segment_he(
                ImgPath=os.path.join(self.sample.dir, self.sample.HE_img),
                MQmodel=model,
                batch_size=16,
                patch_size=128,
                c_order = 'BGR')
            
        elif which == 'IF':
            self.IF_segmented = seg.segment_CD31_Pimo_Hoechst(
                ImgPath=os.path.join(self.sample.dir, self.sample.IF_img),
                MQmodel=model,
                method='sklearn')
            
    
    def register(self):
        
        if self.HE_segmented == '' or self.IF_segmented == 'IF':
            raise FileNotFoundError('One of both necessary input files for registration missing!')
        
        print('')
        print('\t----REGISTRATION----')
        self.registered_IF, self.registered_HE = reg.register_and_transform(self.IF_segmented,
                                                                            self.HE_segmented)   

    def measure(self):
         
        print('')
        print('\t----MEASUREMENT----')
        features.measure(self.registered_IF, self.registered_HE)
        
        

def microquant(dataframe, overwrite=True, **kwargs):
    """

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    kwargs['overwrite'] = overwrite # overwrite previous results?
    
    # =============================================================================
    # Prep work
    # =============================================================================
    
    # if file location is passed to microquant
    if type(dataframe) == str:
        df = pd.read_csv(dataframe)
    
    # if dataframe is passed to microquant
    elif type(dataframe) == pd.core.frame.DataFrame:
        df = dataframe
        
    df = utils.check_data(df)
    utils.create_file_structure(df, **kwargs)
    
    HE_model = MQ_segmentation_model()
    IF_model = MQ_segmentation_model()
    
    HE_model.load(os.path.abspath(r'.\models\HE\model_210809'))
    IF_model.load(os.path.abspath(r'.\models\IF\model_210810'))
    
    for i, sample in df.iterrows():
        
        
        Job = MQ_job(sample=sample)
        
        # Segmentation tasks
        Job.segment(which='HE', model=HE_model)
        Job.segment(which='IF', model=IF_model)
        
        # Registration
        Job.register()
    
        # Measurement
        Job.measure()
        
    

if __name__ == "__main__":
    
    root = r'C:\Users\johan\Desktop\MQ\Test_dir\SampleB'
    df = utils.browse_data(root)
    
    out = microquant(df)
    