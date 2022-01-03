![Logo](/Logo/Logo.png)

This repository contains a collection of scripts for the segmentation, registration and quantification of pairs of tumor tissue sections. Some of the functionality is available as both Python and ImageJ macro script:

## Requirements:

- H&E Segmentation: Python (Framework: [Pytorch](https://pytorch.org))
- IF Segmentation: Python (Framework: [scikit learn ](https://scikit-learn.org/stable/index.html)/[Ilastik](https://www.ilastik.org/)
- Registration: [Elastix](https://elastix.lumc.nl/), can be callled from Python or ImageJ macro
- Measurement: Available in Python and ImageJ macro

## Installation

### Python:
Download and install [Anaconda3](https://www.anaconda.com/products/individual). Create a new environment and install pytorch:
```bash
conda create -n microquant Python=3.8 git jupyter
conda activate microquant
```

Install pytorch according to the configuration on the [Pytorch homepage](https://pytorch.org/get-started/locally/). For instance, for Windows OS and CUDA 11.3 driver this can be achieved by
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
``` 

If you have an existing GPU on your computer, you can check that the installation worked by running the following commands, which should return `True`:
```
python
import torch
torch.cuda.is_available()
```

### ImageJ macro:
As for the imageJ macro scripts, it is sufficient to download [Fiji](https://imagej.net/software/fiji/) and open the macro in the Fiji toolbar.

## How to run
The source code for the Python scripts is provided [here](https://github.com/jo-mueller/MicroQuant/tree/main/microquant). You can find the ImageJ macro implementation as well as the Python demo notebooks here and here, respectively.

