#!/usr/bin/env bash
Partation=1
GPUID=1
Net=GoogLeNet
Split=trainval
Dataset=rap2
LOG="logs/${Net}/${Dataset}_${Split}_part${Partation}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/${Net}/${Dataset}_solver_${Split}_${Partation}.prototxt \
    --weights=./pretrained/bvlc_googlenet.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
