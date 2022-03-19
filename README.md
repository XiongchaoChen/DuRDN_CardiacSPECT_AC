# CT-free attenuation correction for dedicated cardiac SPECT using a 3D dual squeeze-and-excitation residual dense network

Xiongchao Chen, Bo Zhou, Luyao Shi, Hui Liu, Yulei Pang, Rui Wang, Edward J Miller, Albert J Sinusas, Chi Liu

Journal of Nuclear Cardiology. 2021 Jun 3:1-6.

![image](IMAGE/DuRDB.png)

[[Paper](https://link.springer.com/content/pdf/10.1007/s12350-021-02672-0.pdf)]

This repository contains the PyTorch implementation of DuRDN.

### Citation
If you use this code for your research or project, please cite:
    @article{chen2021ct,
      title={CT-free attenuation correction for dedicated cardiac SPECT using a 3D dual squeeze-and-excitation residual dense network},
      author={Chen, Xiongchao and Zhou, Bo and Shi, Luyao and Liu, Hui and Pang, Yulei and Wang, Rui and Miller, Edward J and Sinusas, Albert J and Liu, Chi},
      journal={Journal of Nuclear Cardiology},
      pages={1--16},
      year={2021},
      publisher={Springer}
    }
 
 ### Environment and Dependencies
 Requirements:
 * Python 3.6.10
 * Pytorch 1.2.0
 * numpy 1.19.2
 * scipy
 * scikit-image
 * h5py
 * tqdm
 
 Our code has been tested with Python 3.6.10, Pytorch 1.2.0, CUDA: 10.0.130 on Ubuntu 18.04.6.
 
 ### Dataset Setup
    .
    Data
    ├── train                # contain training files
    |   ├── data1.h5
    |       ├── AC.mat  
    |       ├── NC.mat
    |       ├── SC.mat
    |       ├── SC2.mat
    |       ├── SC3.mat
    |       ├── GD.mat
    |       ├── BMI.mat
    |       ├── STATE.mat
    |   └── ...  
    |
    ├── valid                # contain validation files
    |   ├── data1.h5
    |       ├── AC.mat  
    |       ├── NC.mat
    |       ├── SC.mat
    |       ├── SC2.mat
    |       ├── SC3.mat
    |       ├── GD.mat
    |       ├── BMI.mat
    |       ├── STATE.mat
    |   └── ... 
    |
    └── test                 # contain testing files
        ├── data1.h5
            ├── AC.mat  
            ├── NC.mat
            ├── SC.mat
            ├── SC2.mat
            ├── SC3.mat
            ├── GD.mat
            ├── BMI.mat
            ├── STATE.mat
        └── ... 

Each .mat should contain a H x W x H float value matrix. 
AC: Attenuation-corrected image
NC: Non-attenuation-corrected image
SC: The 1st Scatter-window image
SC2: The 2nd Scatter-window image
SC3: The 3rd Scatter-window image   
GD: Gender encoding volume (0/1)
BMI: BMI value volume
State: Stress/rest value volume (0/1)

### To Run the Code



 
