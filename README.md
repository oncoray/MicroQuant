![Logo](/Logo/Logo.png)

This repository contains a collection of scripts for the segmentation, registration and quantification of pairs of tumor tissue sections. Some of the functionality is available as both Python and ImageJ macro script.

The scripts and notebooks in this reppository refer to the publication:

*Radiomics-based tumor phenotype determination based on medical imaging and tumor microenvironment in a preclinical setting, Müller et al. (under review)* 

All used scripts for this publication are saved in this repository as follows:
- H&E image segmentation: Pytorch segmentation pipeline, which can be examplarily viewed in a [Jupyter notebook](notebooks/Python/Segmentation_HE_apply_model.ipynb)
- Immunofluorescent (IF) image segmentation: [ImageJ macro script](notebooks/ImageJ/Classify/Run_classification.ijm) relying on image segmentation with Ilastik. 
- Image registration of H&E and IF images: [ImageJ macro script](notebooks/ImageJ/Registration/Run_registration.ijm) relying on image registration with Elastix.
- Tumor microenvironent features (TME):  [ImageJ macro script](notebooks/ImageJ/Measure/Process_segmented_registered_data.ijm).

## Requirements:

- H&E Segmentation: Python (Framework: [Pytorch](https://pytorch.org))
- IF Segmentation: Python (Framework: [scikit learn ](https://scikit-learn.org/stable/index.html)/[Ilastik](https://www.ilastik.org/)
- Registration: [Elastix](https://elastix.lumc.nl/), can be callled from Python or ImageJ macro
- Measurement: ImageJ macro, requires the [CLIJ2 plugin](https://clij.github.io/) for Fiji

- File structure: The data are expected to be in a particular file structure to allow running analysis:
```bash
├── root
    ├── N1XX_Tumor_YY
        ├── Imaging
        └── Histology
            ├── 1_Sample_1
            └── 2_Sample_2
                ├── Filename_IF
                └── Filename_HE
```
where N1XX requires to the study number (in this case, could be N182, N183, N194 or N195) and YY refers to the tumor type, which could be either SAS or UT-SCC-14. The filenames `Filename_IF` and `Filename_HE` are required to carry the identifier strings `IF` and `HE` in their filenames, respectively.

## Installation

### Python:
Download and install [Anaconda3](https://www.anaconda.com/products/individual). Create a new environment and install pytorch:
```bash
conda create -n microquant Python=3.8 git
conda activate microquant
```

Install pytorch according to the configuration on the [Pytorch homepage](https://pytorch.org/get-started/locally/). For instance, for Windows OS and CUDA 11.3 driver this can be achieved by
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
``` 

You can now proceed to install microquant by cloning the repository and installing it:
```bash
git clone https://github.com/jo-mueller/MicroQuant.git
cd microquant
pip install -e .
```

### ImageJ macro:
As for the imageJ macro scripts, it is sufficient to download [Fiji](https://imagej.net/software/fiji/) and open the macro in the Fiji toolbar.

