"""Main module."""

from utils import segmentation as seg
from utils import utils

import pandas as pd


def microquant(dataframe, **kwargs):
    """

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # if file location is passed to microquant
    if type(dataframe) == str:
        df = pd.read_csv(dataframe)
    
    # if dataframe is passed to microquant
    elif type(dataframe) == pd.core.frame.DataFrame:
        df = dataframe
        
    df = utils.check_data(df)
    
    

if __name__ == "__main__":
    
    root = r'E:\Promotion\Projects\2020_Radiomics\Data'
    df = utils.browse_data(root)
    
    
    microquant(df)
    