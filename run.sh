#!/usr/bin/env bash

# python main.py --task aste \
#             --dataset rest16 \
#             --paradigm annotation \
#             --n_gpu 0 \
#             --do_train \
#             --do_direct_eval \
#             --train_batch_size 16 \
#             --gradient_accumulation_steps 2 \
#             --eval_batch_size 16 \
#             --learning_rate 3e-4 \
#             --num_train_epochs 20 

# python -u bart.py --task tasd-cn \
#             --dataset ctrip \
#             --paradigm extraction \
#             --n_gpu '0' \
#             --do_train \
#             --do_eval \
#             --train_batch_size 2 \
#             --gradient_accumulation_steps 2 \
#             --eval_batch_size 2 \
#             --learning_rate 3e-4 \
#             --num_train_epochs 15  > logs/noemj_bart_3e-4.log

python -u cpt.py --task tasd-cn \
            --dataset ctrip \
            --paradigm extraction \
            --n_gpu '0' \
            --do_train \
            --do_eval \
            --train_batch_size 2 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 2 \
            --learning_rate 3e-4 \
            --num_train_epochs 15  > logs/noemj_cpt_3e-4.log

# python -u main.py --task tasd-cn \
#             --dataset ctrip \
#             --paradigm extraction \
#             --n_gpu '2' \
#             --do_train \
#             --do_eval \
#             --train_batch_size 2 \
#             --gradient_accumulation_steps 2 \
#             --eval_batch_size 2 \
#             --learning_rate 5e-5 \
#             --num_train_epochs 25  > logs/emj1119_correct5e-5.log

# python -u main.py --task tasd-cn \
#             --dataset ctrip \
#             --paradigm annotation \
#             --n_gpu '2','3' \
#             --do_train \
#             --do_eval \
#             --train_batch_size 2 \
#             --gradient_accumulation_steps 2 \
#             --eval_batch_size 2 \
#             --learning_rate 5e-5 \
#             --num_train_epochs 15 #> logs/annotation_noemj1119.log
