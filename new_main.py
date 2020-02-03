import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import cGAN_Model
from Onehot_embedding import Voc

#导入经过处理的label原始信息
f=open("train_tag_dict.txt","r")
data =eval(f.read())

#展示前5条
for i in range(5):
    print(data[])

voc = Voc("d_vector")
for key in data:
    voc.addSentence(data[key])
voc.trim(100)