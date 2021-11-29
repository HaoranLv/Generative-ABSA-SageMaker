import pandas as pd
import re
import numpy as np
from collections import Counter
from imblearn import over_sampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MultiLabelBinarizer,LabelBinarizer
import emoji

sentiments=['negative','none','positive']
def emoji_idx(s):
    res=[]
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    it = co.finditer(s)
    for i in it:
        res.append(i.span())
    return res

def get_annotated_aste_targets(sents, labels):
    senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}

    annotated_targets = []
    num_sents = len(sents)
    for i in range(num_sents):
        tuples = labels[i]
        # tup: ([2], [5], 'NEG')
        for tup in tuples:
            ap, op, sent = tup[0], tup[1], tup[2]
            op = [sents[i][j] for j in op]
            # multiple OT for one AP
            if '[' in sents[i][ap[0]]:
                # print(i)
                if len(ap) == 1:
                    sents[i][ap[0]] = f"{sents[i][ap[0]][:-1]}, {' '.join(op)}]"
                else:
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}, {' '.join(op)}]"
            else:
                annotation = f"{senttag2word[sent]}|{' '.join(op)}"
                if len(ap) == 1:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
                else:
                    sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
                    sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
        annotated_targets.append(sents[i])
    return annotated_targets

def correct_idx(s,lab):
    emj=emoji_idx(s)   
    if len(emj)==0 or lab=='[]':
        return lab
    lab=eval(lab)
    # print(lab)
    for i in range(len(lab)):
        lab[i]=list(lab[i])
        lab[i][4]=list(lab[i][4])
        lab[i][5]=list(lab[i][5])
        for j in range(len(emj)):
            # print(1)
            # print(s)
            # print(emj)
            # break
            if emj[j][0]<=lab[i][4][0]:
                lab[i][4][0]-=1
                lab[i][4][1]-=1
            if emj[j][0]<=lab[i][5][0]:
                lab[i][5][0]-=1
                lab[i][5][1]-=1
        lab[i][4]=(lab[i][4][0],lab[i][4][1])
        lab[i][5]=(lab[i][5][0],lab[i][5][1])
        lab[i]=(lab[i][0], lab[i][1],lab[i][2],lab[i][3],lab[i][4],lab[i][5])
    lab=str(lab)
    return lab
def clean(path):
    df=pd.read_csv(path)
    false=[]
    false_idx=[]
    for i in range(len(df)):
        lab=eval(df['aspect_opinion_list'][i])
#     print(lab)
#     print(lab[4][0])
        if df['aspect_opinion_list'][i]=='[]':
            continue
        for j in range(len(lab)):
            if  lab[j][0]!=df['reviews'][i][lab[j][4][0]:lab[j][4][1]] or lab[j][2]!=df['reviews'][i][lab[j][5][0]:lab[j][5][1]]:
                false.append(df['reviews'][i])
                false_idx.append(i)
                break   
    return false_idx            

def pick_cn(content):
    content = str(content)
    REG_CN ="[\u4e00-\u9fa5]";#包含中文英文数字
    for i in content:
        if re.match(REG_CN,i) != 'none':
            return ''.join(re.findall(REG_CN,content))
        else:
            return 'none'
def replace_emoji(content):
    data = content.map(lambda x: emoji.demojize(x))
    res = re.sub('\:.*?\:','xx',data)
    return res

def del_blank(content):
    separators=[' ']
    for separator in separators:
        content=content.replace(separator, '')
    return content

def write_txt(df,path='data/ctrip/total.txt'):
    with open(path,'a')as f:
        for i in range(len(df)):
            if len(df.loc[i,'reviews'])>600:
                continue
            f.write("{} ####{}".format(df.loc[i,'reviews'],df.loc[i,'aspect_opinion_list']))
            f.write('\n')

def write_txt2(df,false,path='data/ctrip/total1119_clean.txt'):
    with open(path,'a')as f:
        for i in range(len(df)):
            if len(df.loc[i,'reviews'])>600 or i in false:
                continue
            f.write("{} ####{}".format(df.loc[i,'reviews'],df.loc[i,'aspect_opinion_list']))
            f.write('\n')
def read_txt(root):
    res=[]
    with open(root, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            res.append(line)
    return res
def filter_emoji(desstr,restr=''):
    #过滤表情
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)
def write_train_test(train_path,test_path,root='data/ctrip/total1109.txt'):
    data=read_txt(root)
    x_train,x_test=train_test_split(data,test_size=0.2, random_state=42)
    with open(train_path,'a')as f1:
        for i in range(len(x_train)):
            f1.write(x_train[i])
            f1.write('\n')
    with open(test_path,'a')as f2:
        for i in range(len(x_test)):
            f2.write(x_test[i])
            f2.write('\n')
def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels, total = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels, total = [], [], []
        for line in fp:
            line = line.strip()
            if line != '':
                total.append(line)
                words, tuples = line.split('####')
                sents.append(words.split())
                # sents.append(words)
                labels.append(eval(tuples))
    print(f"Total examples = {len(sents)}")
    return sents, labels, total
def cal_rate(labs):
    num_dict=dict()
    num_dict[' ']=0
    num=0
    for i in range(len(labs)):
        lab=list(set(labs[i]))
        num+=len(lab)
        if len(lab)==0:
            num+=1
            num_dict[' ']+=1
        for j in range(len(lab)):
            if (lab[j][1],lab[j][3]) not in num_dict.keys():
                num_dict[(lab[j][1],lab[j][3])]=1
            else:
                num_dict[(lab[j][1],lab[j][3])]+=1
    for key,val in num_dict.items():
        num_dict[key]=num_dict[key]/num
    return num_dict,num
def build_xy(total,lab):
    x,y=[],[]
    for i in range(len(lab)):
        if len(lab[i])==0:
            x.append(total[i])
            y.append(' ')
        for j in range(len(lab[i])):
            x.append(total[i])
            y.append((lab[i][j][1],lab[i][j][3]))
    return (x,y)