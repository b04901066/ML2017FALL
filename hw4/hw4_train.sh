#!/bin/bash
python3 ./hw4_train.py $1 rnn.h5
python3 ./hw4_gene.py $2 ./gene.txt rnn.h5
python3 ./hw4_train.py $1 81703.h5 ./gene.txt
exit 0
