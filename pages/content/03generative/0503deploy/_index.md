---
title: "生成式ABSA模型部署"
date: 2021-11-05T14:52:27+08:00
weight: 0500
draft: false
---


首先打包镜像并推送

```
cd endpoint
sh build_and_push.sh generative-absa
```
然后部署一个预置的endpoint


```shell script
#注意修改：847380964353.dkr.ecr.ap-northeast-1.amazonaws.com/generative-absa为自己对应的

!python create_endpoint.py \
--endpoint_ecr_image_path "847380964353.dkr.ecr.ap-northeast-1.amazonaws.com/generative-absa" \
--endpoint_name 'generative-absa' \
--instance_type "ml.p3.2xlarge"
```
输出
```
model_name:  generative-absa
endpoint_ecr_image_path:  847380964353.dkr.ecr.ap-northeast-1.amazonaws.com/generative-absa
<<< Completed model endpoint deployment. generative-absa
```

![](../pics/02pegasus/14.png)

当状态变为`InService`即代表部署完成

在部署结束后，看到SageMaker控制台生成了对应的endpoint,可以使用如下客户端代码测试调用

```python
%%time 

from boto3.session import Session
import json
data={"data": '早餐一般般，勉勉强强填饱肚子，样式可选性不多，可能是疫情的影响吧。不过酒店的服务不错，五个小孩早餐都送了，点👍。由于酒店历史有点长，所以设施感觉一般般，整体还可以，三钻吧'}
session = Session()
    
runtime = session.client("runtime.sagemaker")
response = runtime.invoke_endpoint(
    EndpointName='absa',
    ContentType="application/json",
    Body=json.dumps(data),
)

result = json.loads(response["Body"].read())
print (result)
```

结果如下
```
{'result': '(小孩早餐, 儿童餐饮, 送了点👍, 正)', 'infer_time': '0:00:00.725859'}
```

