#!/usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mjpro150/bin
export CHAINER_TYPE_CHECK=0
export OMP_NUM_THREADS=1

$@
