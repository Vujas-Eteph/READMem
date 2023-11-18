# :blue_book: Installation

This document contains detailed instructions for installing the necessary dependencies needed to make this project work on your system.

:construction: To-Do List :construction::
- [ ] List the required packages.
- [ ] Provide a step-by-step guide.
- [ ] Add a bash file for automatic installation.

***Prerequisites:***
- :penguin: The installation has been tested on an Ubuntu 20.04 system.
- :snake: We used a (mini)Conda environment - version 4.13 - with Python version 3.10
- :fire: At least one Nvidia-GPU (We tested on a Nvidia GTX 1080-Ti) as we use PyTorch.


## ⬜ Essentials <a name="Essentials"></a>
- Create and activate a conda environment:
  ```bash
  conda create --name READMem python=3.10
  conda activate READMem
  ```
- Install PyTorch with CUDA (our version is 12.2 [or 11.7 check again])   
  ```bash
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  ```
- Install panda, matplotlib, PIL etc ...

## 🟥 READMem-MiVOS <a name="MiVOS"></a>
Download under MiVOS, the propagation branch from the official [MiVOS(Mask-Propagation) repository](https://github.com/hkchengrex/Mask-Propagation), by following their instructions.  
Replace the scripts (detailed the one), with the on in (path), through (command) / and delete the the needn't one

## 🟦 READMem-STCN <a name="STCN"></a>
:construction::construction::construction::construction:

## 🟧 READMem-QDMN <a name="QDMN"></a>
:construction::construction::construction::construction:

## :hotsprings: Download datasets <a name="dataset"></a>
We use the [DAVIS 17 dataset](https://davischallenge.org/) and [LV1 dataset](https://www.kaggle.com/datasets/gvclsu/long-videos).
Download the datasets and link them to a data folder in the main project:
```bash
mkdir data
ln -s [path/to/where/DAVIS/is/downloaded] ./data/DAVIS
ln -s [path/to/where/LV1/is/downloaded] ./data/long_video_set
```

## :white_check_mark: Check if everything is alright

In the end, the repository's tree should look like this - use ```tree -L 1``` in the main project folder:
```bash
├── MiVOS
├── READMem_API
├── data
├── dataset
├── docs
├── inference/data
├── model
├── scripts
├── util
├── .gitignore
├── LICENSE
├── README.md
├── READMem_MiVOS.py
├── TO-DO.md
├── inference_READMem_MIVOS.py
├── memory_configuration.yaml
├── requirements.txt
└── train.py
```




