---
title: "生成式的ABSA模型训练"
date: 2021-11-05T14:52:27+08:00
weight: 0300
draft: false
---

我们现在会用sagemaker进行一个生成式的ABSA模型的本地训练，使用ML.P3.2xlarge机型。


## 数据准备 

首先下载代码
```
cd SageMaker
git clone https://github.com/HaoranLv/Generative-ABSA-SageMaker.git
```


安装环境

```
source activate pytorch_p37
pip install -r requirements.txt
```

然后处理数据`data_prepare.ipynb`，进行数据清洗并切分train/test。
注意这里，为了快速产生结果，我们只要用800条数据训练，200条测试/验证
```
df=pd.read_csv('data/ctrip/data1119_part.csv')
write_txt(df,path='data/ctrip/total.txt')
write_train_test(train_path='data/ctrip/train.txt',test_path='data/ctrip/test.txt',root='data/ctrip/total1119.txt')
```

## 模型训练

接下来我们运行训练,这里我们使用huggingface上的公开的`lemon234071/t5-base-Chinese`作为训练起点，为了演示目的，我们只运行一个epoch，大约需要5min

```
python -u main.py --task tasd-cn \
            --dataset ctrip \
            --paradigm extraction \
            --n_gpu '0' \
            --model_name_or_path lemon234071/t5-base-Chinese \
            --do_train \
            --train_batch_size 2 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 2 \
            --learning_rate 3e-4 \
            --num_train_epochs 1  > logs/noemj_lr3e-4.log
```

训练完成后，会提示日志信息如下

```
Finish training and saving the model!
```

模型结果文件及相应的日志等信息会自动保存在`./outputs/tasd-cn/ctrip/extraction/`


## 结果本地验证
我们可以直接用这个产生的模型文件进行本地推理。注意这里的模型文件地址的指定为你刚刚训练产生的。
```
python main.py --task tasd-cn \
            --dataset ctrip \
            --ckpoint_path outputs/tasd-cn/ctrip/extraction/cktepoch=1.ckpt \
            --paradigm extraction \
            --n_gpu '0' \
            --do_direct_eval \
            --eval_batch_size 128 \
```
## 结果本地测试

我们可以直接用这个产生的模型文件进行本地推理。注意这里的模型文件地址的指定为你刚刚训练产生的。

```
python main.py --task tasd-cn \
            --dataset ctrip \
            --ckpoint_path outputs/tasd-cn/ctrip/extraction/cktepoch=1.ckpt \
            --text 早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧 \
            --paradigm extraction \
            --n_gpu 0 \
            --do_direct_predict \
```

输出如下

```
sents: 早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧
pred: 
0.5794262886047363

```
由于样本数据中空输出较多，且仅训练了一个epoch故模型倾向于输出空


到这里，就完成了一个模型的训练过程。