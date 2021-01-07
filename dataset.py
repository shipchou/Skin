import torchvision.transforms as trns
from PIL import Image
from scipy.io import loadmat

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from torchvision.datasets.folder import default_loader
import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import OneHotEncoder
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    
 
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    
    classes.sort()
    
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    tmp=list()
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
    	

        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    #print(class_to_idx[target])
                    tmp.append(class_to_idx[target])
                    item = (path, class_to_idx[target])
                    images.append(item)


    enc = OneHotEncoder().fit(tmp)
    for i in range(len(images)):
    	images[i]=list(images[i])
    	images_array=np.array(images[i][1])
    	images_array = (images_array[np.newaxis,: ])
     	images[i][1]=enc.transform(images_array).toarray()
        images[i][1]=np.squeeze(images[i][1])
        images[i][1]=torch.tensor(images[i][1])


    return images

class SkinFolder(Dataset):

    def __init__(self, root, transform=None, target_transform=None,loader=default_loader):
	classes, class_to_idx = find_classes(root)
        
    	multi=np.array(classes)
        multi = (multi[:, np.newaxis]).tolist()
        for i in range(len(multi)):
	    if multi[i][0]=='AK' or multi[i][0]=='BCC' or multi[i][0]=='DF' or multi[i][0]=='SCC' or multi[i][0]=='SK':
		multi[i].insert(0,'g')
	    else:
		multi[i].insert(0,'b')
		self.multi=multi
  	zip_iterator = zip(classes, multi)
  	mapping = dict(zip_iterator)
  	imgs = make_dataset(root, mapping)
        self.imgs=imgs
        
        if len(imgs) == 0:
	    raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"+"Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

	self.root = root

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)




