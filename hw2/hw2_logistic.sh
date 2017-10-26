#!/bin/bash
python3 ./logistic.py --infer --train_data_path $3 --train_label_path $4 --test_data_path $5 --output_dir $6 --save_dir ./logistic_params/
exit 0
