# load data and process
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .augmentation import MyAug
import random


class MyDataset(Dataset):
    def __init__(self, lstpaths, img_root, mappings, aug: MyAug, ratio=1.) -> None:
        """
        lstpaths: paths of the dataset list file
        img_root: root path of images
        """

        self.img_root = img_root
        self.aug = aug
        self.num_class = len(mappings)
        self.mappings = mappings
        self.ratio = ratio

        self.lstpaths = lstpaths
        self.lst = self.read_list(self.lstpaths)
        random.shuffle(self.lst)
    
    def read_list(self, lstpaths):
        total_lst = []
        for lstpath in lstpaths:
            with open(lstpath) as fin:
                lst = fin.readlines()[:]
                lst = list(map(lambda x: x.strip('\r\n').split('\t'), lst))
                # random.shuffle(lst)
                # lst = random.sample(lst, int(len(lst) * self.ratio))
                total_lst += lst
        random.shuffle(total_lst)
        total_lst = random.sample(total_lst, int(len(total_lst) * self.ratio))
        return total_lst

    def __getitem__(self, index):

        line = self.lst[index]

        dataset, img_name, labels = line 
        img_path = os.path.join(self.img_root, dataset, img_name)
        if not os.path.exists(img_path):
            raise Exception('image file missing', img_path)
        
        # load image
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]
        img, _ = self.aug.process(img)
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))
        
        # load labels
        labels = labels.split(',')
        gt = np.zeros(self.num_class)
        for label in labels:
            if label not in self.mappings:
                continue
            gt[self.mappings[label]] = 1
        
        return img_path, img, gt

    def __len__(self):
        return len(self.lst)

