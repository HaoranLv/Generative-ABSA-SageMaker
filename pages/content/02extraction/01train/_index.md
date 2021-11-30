---
title: "抽取式ABSA模型训练1"
date: 2021-11-05T14:52:27+08:00
weight: 0270
draft: false
---

paper - A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction。 

SOTA for ate task： https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=a-multi-task-learning-model-for-chinese 
我们现在会用sagemaker进行一个模型的本地训练，使用ml.p3.8xlarge机型。

## 数据准备 

首先下载代码
```
source activate pytorch_p37
cd SageMaker
git clone https://github.com/jackie930/PyABSA.git
```

将数据`data1119.csv`上传到`PyABSA/data/data1109.csv`

## 模型训练

数据准备
```
cd PyABSA
pip install termcolor update_checker findfile jupyterlab-git torch==1.10.0 transformers==4.12.3 autocuda spacy googledrivedownloader seqeval emoji
python pyabsa/utils/preprocess.py --inpath './data/data1109.csv' --folder_name 'custom_atepc_1109' --task 'aptepc'
```

然后进行模型训练，演示目的，只训练一个epoch

```python
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ATEPCConfigManager

atepc_config_custom = ATEPCConfigManager.get_atepc_config_chinese()
atepc_config_custom.num_epoch = 2
atepc_config_custom.evaluate_begin = 1
atepc_config_custom.log_step = 100
atepc_config_custom.model = ATEPCModelList.LCF_ATEPC

aspect_extractor = ATEPCTrainer(config=atepc_config_custom, 
                                dataset='./custom_atepc_1109'
                                )
```

训练完成后，模型评估
```
python utils/metrics_cacl.py --data_path --checkppoint
```

