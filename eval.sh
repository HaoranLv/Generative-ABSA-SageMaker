#!/usr/bin/env bash

# python train_ctrip.py --task tasd-cn2-xtc \
#             --dataset xtc \
#             --paradigm extraction \
#             --n_gpu 0 \
#             --do_direct_eval \
#             --eval_batch_size 32 \

python train_ctrip.py --task tasd-cn \
            --dataset ctrip \
            --paradigm extraction \
            --n_gpu '0' \
            --do_direct_eval \
            --eval_batch_size 128 \

# python train_ctrip.py --task tasd-cn \
#             --dataset ctrip \
#             --paradigm annotation \
#             --n_gpu '2' \
#             --do_direct_eval \
#             --eval_batch_size 128 \
