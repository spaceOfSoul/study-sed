#!/bin/bash
conda create -y -n dcase2020 python=3.10
conda activate dcase2020
pip install pandas h5py scipy
pip install pytorch torchvision cudatoolkit # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
pip install pysoundfile librosa youtube-dl tqdm
pip install  ffmpeg

pip install dcase_util
pip install sed-eval
pip install --upgrade psds_eval
pip install scaper

# Should be done only if you did not already installed it to download the data
pip install --upgrade desed@git+https://github.com/turpaultn/DESED

# Source separation:
pip install tensorflow
