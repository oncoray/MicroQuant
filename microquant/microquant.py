"""Main module."""

from utils import segmentation as seg
from utils import registration as reg

from utils import utils
from utils.model import model

import pandas as pd
import os


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
    # utils.create_file_structure(df, **kwargs)
    
    HE_model = model(path = r'D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\models\HE\model_210809')
    IF_model = model(path = r'D:\Documents\Promotion\Projects\2021_MicroQuant\microquant\models\IF\model_210810')
    
    for i, sample in df.iterrows():
        
        # segmentation
        f_HE = seg.segment_he(
                ImgPath=os.path.join(sample.dir, sample.HE_img),
                model = HE_model,
                batch_size=16,
                patch_size=128,
                c_order = 'BGR'
                )
        
        f_IF = seg.segment_CD31_Pimo_Hoechst(
                ImgPath=os.path.join(sample.dir, sample.IF_img),
                model=IF_model,
                method='sklearn'
                )
        
        reg.register(f_IF, f_HE)
    

if __name__ == "__main__":
    
    root = r'C:\Users\johan\Desktop\Test_dir'
    df = utils.browse_data(root)
    
    out = microquant(df)
    