#!/bin/bash
wget 'https://www.dropbox.com/s/an43m7g42rmczsg/81703.h5?dl=1' -O './81703.h5'
python3 ./hw4_test.py $1 $2 ./81703.h5
exit 0
