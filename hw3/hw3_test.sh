#!/bin/bash
wget 'https://www.dropbox.com/s/e9n9ipmxn4quc3p/best2.h5?dl=1' -O './best2.h5'
wget 'https://www.dropbox.com/s/t4tkc408ykrcxp5/hw3_65.h5?dl=1' -O './hw3_65.h5'
wget 'https://www.dropbox.com/s/9w3i0s7ayuwngvf/hw3_Liu.h5?dl=1' -O './hw3_Liu.h5'
wget 'https://www.dropbox.com/s/or4j7gagrvoajcz/u.h5?dl=1' -O './u.h5'
python3 ./hw3_test.py $1 $2 ./best2.h5 ./hw3_65.h5 ./hw3_Liu.h5 ./u.h5
exit 0
