# :blue_book: Installation

This document contains detailed instructions for installing the necessary dependencies needed to make this project work on your system.

:construction: To-Do List :construction::
- [ ] List the required packages
- [ ] Provide a step-by-step guide.
- [ ] Add a bash file for automatic installation.

***Prerequisites:***
- :penguin: The installation has been tested on an Ubuntu 20.04 system.
- :snake: We used a (mini)Conda environment - version 4.13 - with Python version 3.10
- :fire: At least one Nvidia-GPU (We tested on a Nvidia GTX 1080-Ti) as we use PyTorch.

## ⬜ Essentials
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

## 🟥 READMem-MiVOS Installation
Install MiVOS, by following instructions from the official [MiVOS(Mask-Propagation) repository](https://github.com/hkchengrex/Mask-Propagation).
Replace the scripts (detailed the one), with the on in (path), through (command)

At the end, the repository's tree should look like this - use ```tree -L 1```:
```bash
├── MiVOS
├── READMem_API
├── dataset
├── docs
├── img
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


## 🟦 READMem-STCN Installation


## 🟧: READMem-QDMN Installation



