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
import random

def to_img(x):
	out = 0.5 * (x + 1)
	out = out.clamp(0, 1)
	out = x.view(-1, 3, 96, 96)
	return out


file=open('config.yml')
config = yaml.load(file,Loader=yaml.Loader)
file.close()




#导入经过处理的label原始信息
f=open("final_train_tag_dict.txt","r")
data =eval(f.read())


# print("Showing the first 5 descriptions")
# #展示前5条
# for i in range(5):
#     print("img_index",list(data.keys())[i],"description:",data[list(data.keys())[i]])


#生成label向量
voc = Voc("d_vector")
for key in data:
	voc.addSentence(data[key])
voc.trim(100)

# #展示5条onehot编码的向量
# print("Showing the one_hot vector for the first 5 descriptions")
# for i in range(5):
#     print("img_index",list(data.keys())[i],"description:",data[list(data.keys())[i]])
#     print("onehot_vector:",voc.generate_final_vector(data[list(data.keys())[i]]))


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
									 shuffle=False, # 不要乱序
									 num_workers=8 # 多进程
									 )
label = list(data.values())
num_samples = len(label)

label_vectors=[]
for i in label:
	label_vectors.append(torch.from_numpy(voc.generate_final_vector(i)))



batch_vectors=torch.cat(label_vectors[0:10],0)
batch_vectors=batch_vectors.view(-1,32).float()


G = cGAN_Model.CNN_Generator(z_dimension,32,15*192*192)
D = cGAN_Model.CNN_Discriminator(32)

if torch.cuda.is_available():
	D = D.cuda()
	G = G.cuda()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)





for epoch in range(num_epoch):
	for i, (img,_) in enumerate(dataloader):

		if (i+1)*batch_size < num_samples:
			batch_vectors=torch.cat((label_vectors[i*batch_size:(i+1)*batch_size]),0)
			batch_vectors=batch_vectors.view(-1,32).float()
		else:
			batch_vectors=torch.cat((label_vectors[i*batch_size:-1]),0)
			batch_vectors=batch_vectors.view(-1,32).float()

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		num_img = img.size(0)
		#train discriminator
		# compute loss of real_matched_img
		img = img.view(num_img,3,96,96)
		real_img = Variable(img).to(device)
		real_label = Variable(torch.ones(num_img)).to(device)
		fake_label = Variable(torch.zeros(num_img)).to(device)
		batch_vectors = Variable(batch_vectors).to(device)
		matched_real_out = D(real_img,batch_vectors)
		d_loss_matched_real = criterion(matched_real_out, real_label)
		matched_real_scores = matched_real_out  # closer to 1 means better

		# compute loss of fake_matched_img
		z = Variable(torch.randn(num_img, z_dimension)).to(device)
		z = torch.cat((z,batch_vectors),axis=1).to(device)
		fake_img = G(z)
		matched_fake_out = D(fake_img,batch_vectors)
		d_loss_matched_fake = criterion(matched_fake_out, fake_label)
		matched_fake_out_scores = matched_fake_out  # closer to 0 means better

		# compute loss of real_unmatched_img

		rand_label_vectors=random.sample(label_vectors,num_img)
		rand_batch_vectors=torch.cat((rand_label_vectors[:]),0)
		rand_batch_vectors=rand_batch_vectors.view(-1,32).float().to(device)


		z = Variable(torch.randn(num_img, z_dimension)).to(device)
		z = torch.cat((z,rand_batch_vectors),axis=1).to(device)
		fake_img = G(z)
		unmatched_real_out = D(fake_img,batch_vectors)
		d_loss_unmatched_real = criterion(unmatched_real_out, fake_label)
		unmatched_real_out_scores = unmatched_real_out  # closer to 0 means better

		# bp and optimize
		d_loss = d_loss_matched_real + d_loss_matched_fake + d_loss_unmatched_real
		d_optimizer.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		# ===============train generator
		# compute loss of fake_img
		# compute loss of fake_matched_img
		z = Variable(torch.randn(num_img, z_dimension)).to(device)
		z = torch.cat((z,batch_vectors),axis=1).to(device)
		fake_img = G(z)
		matched_fake_out = D(fake_img,batch_vectors)
		matched_fake_out_scores = matched_fake_out

		g_loss = criterion(matched_fake_out,real_label)

		# bp and optimize
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()


		print('Epoch [{}/{}], Batch {},d_loss: {:.6f}, g_loss: {:.6f} '
				  .format(
				epoch, num_epoch,i,d_loss.data, g_loss.data,
				))
		if epoch == 0:
			real_images = to_img(real_img.cpu().data)
			save_image(real_images, './img/real_images.png')

		fake_images = to_img(fake_img.cpu().data)
		save_image(fake_images, './img/fake_images-{}.png'.format(i+(epoch-1)*batch_size))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')






