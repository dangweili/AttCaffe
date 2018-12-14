#!/bin/bash sh

Partation=1
Net=GoogLeNet
Model_type=trainval
Model_iter=30000
Iter_cnt=266
Dataset=rap2
Split=test
GPUID=2

if [ ! -d "lmdb/result_${Net}_fc8_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_fc8_lmdb
fi

extract_features ./temp_models/${Net}/${Dataset}_${Model_type}_part${Partation}_iter_${Model_iter}.caffemodel \
    ./prototxts/${Net}/${Dataset}_${Split}_${Partation}.prototxt \
    fine_att lmdb/result_${Net}_${Dataset}_${Split}_${Partation}_att_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 

python compute_accuracy_att.py ${Net} ${Dataset} ${Split} ${Partion}

