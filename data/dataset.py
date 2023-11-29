# load data and process
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .augmentation import OurAug
import random


class MyDataset(Dataset):
    def __init__(self, lstpath, img_root, num_class, mappings, aug: OurAug, ratio=1., domain='source') -> None:
        """
        lstpath: path of the dataset list file
        img_root: root path of images
        num_classes: 
        """
        if domain not in ['source', 'target']:
            raise Exception('Invalid domain type {}'.format(domain))
        self.domain = domain

        self.img_root = img_root
        self.aug = aug
        self.num_class = num_class
        self.mappings = mappings
        self.ratio = ratio

        self.lstpath = lstpath
        self.lst = self.read_list(self.lstpath)
        random.shuffle(self.lst)
    
    def read_list(self, lstpath):
        with open(lstpath) as fin:
            lst = fin.readlines()[:]
            lst = list(map(lambda x: x.strip('\r\n').split('\t'), lst))
            random.shuffle(lst)
            lst = random.sample(lst, int(len(lst) * self.ratio))
        return lst

    def __getitem__(self, index):
        
        line = self.lst[index]
        dataset, img_name, labels = line 
        labels = labels.split(',')

        img_path = os.path.join(self.img_root, dataset, img_name)
        if not os.path.exists(img_path):
            raise Exception(img_path)
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]

        # the input size of source image and target image may different
        size_h = self.aug.aug_cfg['{}_size_h'.format(self.domain)]
        size_w = self.aug.aug_cfg['{}_size_w'.format(self.domain)]

        img, _ = self.aug.process(img, size_h=size_h, size_w=size_w)
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))

        gt = np.zeros(self.num_class)
        for label in labels:
            if label not in self.mappings:
                continue
            gt[self.mappings[label]] = 1
        return img_path, img, gt

    def __len__(self):
        return len(self.lst)


class UWFDatasetFolder(Dataset):
    def __init__(self, img_root, aug: OurAug, num_class, need_instance=True, WF=False, center_crop=False):
        self.img_root = img_root
        self.aug = aug
        self.num_class = num_class

        self.need_instance = need_instance
        self.WF = WF  # 区分是wf图像还是uwf图像
        self.center_crop = center_crop # 是否只切出中间区域

        self.lst = self.read_list(img_root=img_root)[:]
        random.shuffle(self.lst)
    
    def read_list(self, img_root):
        lst = []
        extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
        for root_path, dir_list, file_list in os.walk(img_root):
            for fname in file_list:
                if fname.split('.')[-1] not in extensions:
                    continue
                lst.append(os.path.join(root_path, fname))
        return lst

    def __getitem__(self, index):
        img_path = self.lst[index]

        img = cv2.imread(img_path)[:, :, [2, 1, 0]]
        if self.center_crop:
            if self.WF:
                img = ImgCrop.center_wf(img)
            else:
                img = ImgCrop.center_uwf(img)

        size_h = self.aug.aug_cfg['instance_size_h']
        size_w = self.aug.aug_cfg['instance_size_w']

        whole_size_h = self.aug.aug_cfg['whole_size_h']
        whole_size_w = self.aug.aug_cfg['whole_size_w']

        if self.need_instance:
            if self.WF:
                instances, _ = ImgCrop.crop_wf(img)
            else:
                instances, _ = ImgCrop.crop_uwf(img)
            for i in range(len(instances)):
                instances[i], _ = self.aug.process(instances[i], size_h=size_h, size_w=size_w)
                instances[i] = np.multiply(instances[i], 1 / 255.0)
                instances[i] = np.transpose(instances[i], (2, 0, 1))
            instances = np.array(instances)
        else:
            instances = -1

        img, _ = self.aug.process(img, size_h=whole_size_h, size_w=whole_size_w)
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))

        gt = np.zeros(self.num_class)

        return img_path, img, instances, gt, -1

    def __len__(self):
        return len(self.lst)

