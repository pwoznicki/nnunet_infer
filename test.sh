#!/bin/bash
python nnunet/inference/predict_simple.py -i /home/deepcyst/single_input/70100007\ Unnamed\ Series.nrrd -o /home/piotr/Documents/SlicerOutput -t Task00_DATASET -tr nnUNetTrainer -m 2d --tta 0 --num_threads_preprocessing 1 --num_threads_nifti_save 1