
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from datasets_utils.data_utils import ABSADataset
from datasets_utils.data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import *

pre=np.load('predictions.npy',allow_pickle=True)
lab=np.load('labels_total.npy',allow_pickle=True)
print(lab.shape)
print(lab[0])
def cal_class_num(labels):
    num_dict=dict()
    for i in labels:
        for j in i:
            if j[0] not in num_dict.keys():
                num_dict[j[0]]=1
            else:
                num_dict[j[0]]+=1
    return num_dict
num_dict=cal_class_num(lab)
print(num_dict)
print(len(num_dict))


x = list(num_dict.keys())
y = list(num_dict.values())


import matplotlib.pyplot as plt
import random 

my_list = y

labels = x
plt.figure(dpi=1200)
plt.barh(range(len(my_list)), my_list, tick_label=labels,height=0.5)
plt.tick_params(labelsize=5)

# plt.ylabel('time(S)') #图像纵坐标的标记
# plt.figure(figsize=(600, 600))
plt.ylabel('class') #图像纵坐标
plt.savefig("/Users/lvhaoran/AWScode/Generative-ABSA/time_bar.jpg") #保存图像
