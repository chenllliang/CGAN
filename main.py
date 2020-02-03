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



#展示前5条
for i in range(5):
    print(data[])

voc = Voc("d_vector")
for key in data:
    voc.addSentence(data[key])
voc.trim(100)






# def to_img(x):
#     out = 0.5 * (x + 1)
#     out = out.clamp(0, 1)
#     out = x.view(-1, 3, 96, 96)
#     return out


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


batch_size = 500
num_epoch = 5
z_dimension = 512

FaceDataset = datasets.ImageFolder('./datas', transform=img_transform) # 数据路径
dataloader = torch.utils.data.DataLoader(FaceDataset,
                                     batch_size=batch_size, # 批量大小
                                     shuffle=True, # 乱序
                                     num_workers=8 # 多进程
                                     )


# dataloader = torch.utils.data.DataLoader(
#     dataset=dataSet, batch_size=batch_size, shuffle=True,num_workers=8)

G = cGAN_Model.CNN_Generator(z_dimension,32,15*192*192)
D = cGAN_Model.CNN_Discriminator(32)

if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

for epoch in range(num_epoch):
    for i, (img,_) in enumerate(dataloader):
        num_img = img.size(0)
        #train discriminator
        img = img.view(num_img,3,96,96)
        real_img = Variable(img).to(device)
        real_label = Variable(torch.ones(num_img)).to(device)
        fake_label = Variable(torch.zeros(num_img)).to(device)

        # compute loss of real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


        print('Epoch [{}/{}], Batch {},d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                epoch, num_epoch,i,d_loss.data, g_loss.data,
                real_scores.data.mean(), fake_scores.data.mean()))
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')

        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(i+(epoch-1)*batch_size))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')



