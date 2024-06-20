# Node Duplication Improves Cold-start Link Prediction

## Introduction
This repository contains the source code for the paper: [Node Duplication Improves Cold-start Link Prediction](https://arxiv.org/pdf/2402.09711) 

## Requirements
The code was developed and tested with Python 3.10.9. The enviromental requirements are listed as in `requirements.txt`. Please run the following code to install all the requirements:
```
pip install -r requirements.txt
```

## Prepare the datasets
IGB-100K dataset needs to be pre-downloaded and pre-processed. Please install IGB package and download IGB-100k datset as follows:
```
git clone https://github.com/IllinoisGraphBenchmark/IGB-Datasets.git
cd IGB-Datasets/
pip install .
cd IGB-Datasets
python
from igb import dataloader
from igb import download
download.download_dataset(path='../IGB-Datasets', dataset_type='homogeneous', dataset_size='tiny')
```
Then pre-process IGB-100K datasets with the following command:
```
cd data
python igb_process.py
```
Other datasets can be directly downloaded when running the experimental code. 

### Usage

#### Transductive Setting
Please run the following command to reproduce the results in Table 1:
```
bash scripts/transductive.sh
```

#### Production Setting
Please run the following command to reproduce the results in Table 2 and Table 7 in the paper:
```
bash scripts/inductive.sh
```
## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@article{guo2024node,
  title={Node Duplication Improves Cold-start Link Prediction},
  author={Guo, Zhichun and Zhao, Tong and Liu, Yozen and Dong, Kaiwen and Shiao, William and Shah, Neil and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2402.09711},
  year={2024}
}
```

