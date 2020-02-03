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
import yaml



file=open('config.yml')
config = yaml.load(file)
file.close()




#导入经过处理的label原始信息
f=open("train_tag_dict.txt","r")
data =eval(f.read())


print("Showing the first 5 descriptions")
#展示前5条
for i in range(5):
    print("img_index",list(data.keys())[i],"description:",data[list(data.keys())[i]])


#生成label向量
voc = Voc("d_vector")
for key in data:
    voc.addSentence(data[key])
voc.trim(100)

#展示5条onehot编码的向量
print("Showing the one_hot vector for the first 5 descriptions")
for i in range(5):
    print("img_index",list(data.keys())[i],"description:",data[list(data.keys())[i]])
    print("onehot_vector:",voc.generate_final_vector(data[list(data.keys())[i]]))


batch_size=config['batch_size']
num_epoch=config['num_epoch']
z_dimension=config['z_dimension']


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


FaceDataset = datasets.ImageFolder('./data', transform=img_transform) # 数据路径
dataloader = torch.utils.data.DataLoader(FaceDataset,
                                     batch_size=batch_size, # 批量大小
                                     shuffle=True, # 乱序
                                     num_workers=8 # 多进程
                                     )



