# HMNet
Our paper: https://arxiv.org/abs/2111.13324
## Setup：
The code was written in the following environment:  
- python 3.7.11  
- pytorch 1.10.0  
- cuda 11.3  
- cudnn 8.2.0  

## Preparation for data:
The raw data of Next Generation Simulation (NGSIM) is downloadable at https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm   
- Put the raw data into `./raw`
- Run `preprocess_data.m` to preprocess the data for HMNet  

## Using the code:
To use the pretrained models at `./trained_models` and evaluate the models performance run:  
```
python evaluate.py
```
