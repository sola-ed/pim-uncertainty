# Can a single neuron learn quantiles?
Source code for [Can a single neuron learn quantiles?](https://arxiv.org/abs/2106.03702) 

## Requirements

To run the scripts, please install the requirements from the provided conda environment file:

```setup
conda env create -f environment.yml
```

## Contents:
```
├── calib_cls.py        (For calibrating classifiers)
├── clean_cls_data.py   (For data cleaning of classification datasets)
├── cls_benchmarks.py   (To train classification model on given dataset)
├── data                (Datasets and metadata for regression, classification and time series)
├── environment.yml     (Anaconda requirements)
├── hetero_var.py       (To get PSE and P_sig in Table 2)
├── hp_optim_cwc.py     (For hyperparameter optimization with cwc)
├── hp_optim.py         (For hyperparameter optimization without cwc)
├── models              (Best hyperparameters after tuning without cwc)
├── models_cwc          (Best hyperparameters after tuning using cwc)
├── pim.py              (PIM classes)
├── pim_vs_qr.py        (To plot Fig. 2)
├── pim_vs_ranking.py   (To plot Fig. 1)
├── plot_ts.py          (To plot Fig. 3) 
├── README.md           (This file)
├── reg_benchmarks.py   (To train regression model on given dataset) 
├── run_cls.py          (To train on all classification datasets)
├── run_reg.py          (To train on all regression datasets)
└── sqr.py              (Simultaneous Quantiles Regression)
```

This source code is released under a Attribution-NonCommercial 4.0 International license, found [here](https://github.com/sola-ed/pim-uncertainty/blob/main/LICENSE.txt)