#!/usr/bin/env bash

python main.py --task tasd-cn \
            --dataset ctrip \
            --paradigm extraction \
            --n_gpu 0 \
            --do_direct_predict \