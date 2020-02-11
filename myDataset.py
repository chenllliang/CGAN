from torch.utils import data
import numpy as np
from PIL import Image


class face_dataset(data.Dataset):
	def __init__(self):
		self.file_path = './data/faces/'
		f=open("final_train_tag_dict.txt","r")
		self.label_dict=eval(f.read())
		f.close()

	def __getitem__(self,index):
		label = list(self.label_dict.values())[index-1]
		img_id = list(self.label_dict.keys())[index-1]
		img_path = self.file_path+str(img_id)+".jpg"
		img = np.array(Image.open(img_path))
		return img,label

	def __len__(self):
		return len(self.label_dict)



a = face_dataset()

img,label = a[1]

print(img)


