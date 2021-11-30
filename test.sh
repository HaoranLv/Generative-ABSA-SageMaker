#!/usr/bin/env bash

# python main.py --task tasd-cn \
#             --dataset ctrip \
#             --paradigm extraction \
#             --n_gpu 0 \
#             --do_direct_predict \

python main.py --task tasd-cn \
            --dataset ctrip \
            --ckpoint_path outputs/tasd-cn/ctrip/extraction/cktepoch=1.ckpt \
            --text 早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧 \
            --paradigm extraction \
            --n_gpu 0 \
            --do_direct_predict \