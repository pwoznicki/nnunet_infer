#!/bin/bash
python nnunet/inference/predict_simple.py -i /home/piotr/kidney_seg/example/single_input -o /home/deepcyst -t Task00_DATASET -tr nnUNetTrainer -m 2d --tta 0 --num_threads_preprocessing 1 --num_threads_nifti_save 1