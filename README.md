# Can Abnormality be Detected by Graph Neural Networks?
This repository contains the code used to generate the results reported in the paper: [Can Abnormality be Detected by Graph Neural Networks?](https://github.com/Myrtus-c/APANet).

# Dependencies
This project uses the `python 3.7` environment. The code was tested on `torch 1.7` ,  `torch-geometric 1.6.3`  and `CUDA 11.2`

# Structure
The project is structured as follows: 
* `data`: contains the the datasets;
* `baseline`: contains the baseline model shown in the experiment;
* `module`: contains the module code of which the models are composed;
* `utils`: contains all the utility;.

# Usage
### Dataset Download
First you need to download the public dataset elliptic and configuring the environment by running the following commands:
```bash
pip install -r requirements.txt
```

### Data Pre-processing
We split the elliptic dataset following the setting of [Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://arxiv.org/pdf/1908.02591.pdf). For convenience, you could download the processed dataset from [Google Drive](https://drive.google.com/file/d/1NS37DJ0hKMcBl24K4BDsNNNMa60n83OO/view?usp=sharing) to the project root path:
```bash
mv elliptic.tar.gz data
cd data
tar -zxvf elliptic.tar.gz
```

### Model Training & Testing
In order to train & test the model use:
```bash
python run_elliptic.py --dataset [dataset] --logName [logFileName] --K [orders] --test 
```

# Licenze
MIT

