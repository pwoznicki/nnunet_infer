#!/bin/bash
python nnunet/inference/predict_simple.py -i deepcyst/data/test/ -o deepcyst/data/prediction/ -t Task00_DATASET -tr nnUNetTrainer -m 3d_fullres