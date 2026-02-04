# Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection


This repository hosts the official implementation of Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection  based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). 



## Installation
1. Clone this project and create a conda environment:
    ```bash
    cd MonoDLGD

    conda create -n monodlgd python=3.8
    conda activate monodlgd
    ```
    
2. Install pytorch and torchvision matching your CUDA version:
    ```bash
    #install torch, torchvision, torchaudio
    ```
    
3. Install requirements and compile the deformable attention:
    ```bash
    pip install -r requirements.txt

    cd lib/models/monodlgd/ops/
    bash make.sh
    
    cd ../../../..
    ```
 
4. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```bash
    │MonoDLGD/
    ├──...
    │data/kitti/
    ├──ImageSets/
    ├──training/
    │   ├──image_2
    │   ├──label_2
    │   ├──calib
    ├──testing/
    │   ├──image_2
    │   ├──calib
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monodlgd.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monodlgd.yaml` and indicate the GPU in `train.sh`:
  ```bash
  bash train.sh configs/monodlgd.yaml > logs/monodlgd.log
  ```
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monodlgd.yaml`:
  ```bash
  bash test.sh configs/monodlgd.yaml
  ```
You can test the inference time on your own device:
  ```bash
  python tools/test_runtime.py
  ```


## Acknowlegment
This repo benefits from the excellent work [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), [MonoDGP](https://github.com/PuFanqi23/MonoDGP), [DINO](https://github.com/IDEA-Research/DINO) and [DN-DETR](https://github.com/IDEA-Research/DN-DETR)
