---
title: "抽取式ABSA模型训练"
date: 2021-11-05T14:52:27+08:00
weight: 0270
draft: false
---

paper - Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)。 这个方法的输入输出设计很符合TABSA任务，做法简单，适合作为此类任务的baseline.

我们现在会用sagemaker进行一个模型的本地训练，使用ml.p3.8xlarge机型。

## 数据准备 

首先下载代码
```
source activate pytorch_p37
cd SageMaker
git clone https://github.com/jackie930/ABSA-BERT-pair.git
```

将数据`data1119.csv`上传到`data/custom/data1119.csv`

## 模型训练

接下来我们运行训练，首先下载预训练模型并转换为torch版本
```
cd ABSA-BERT-pair
wget -P ./source/bert/pretrain_model/cn https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
cd ./source/bert/pretrain_model/cn
unzip chinese_L-12_H-768_A-12.zip 

pip install tensorflow==1.13.1
cd /home/ec2-user/SageMaker/ABSA-BERT-pair/
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./source/bert/pretrain_model/cn/chinese_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file ./source/bert/pretrain_model/cn/chinese_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path ./source/bert/pretrain_model/cn/pytorch_model.bin
```

数据准备
```
cd generate/
python generate_custom_NLI_M.py
```

然后进行模型训练，演示目的，只训练一个epoch

```
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA-v1.py \
--task_name custom_NLI_M \
--data_dir data/custom/bert-pair/  \
--vocab_file ./source/bert/pretrain_model/cn/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file ./source/bert/pretrain_model/cn/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint ./source/bert/pretrain_model/cn/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 48 \
--learning_rate 2e-5 \
--num_train_epochs 1.0 \
--do_save_model \
--output_dir results/custom/NLI_M \
--seed 42
```

训练完成后，模型评估
```
python evaluation.py --task_name custom_NLI_M --pred_data_dir results/custom/NLI_M/test_ep_1.txt
```

