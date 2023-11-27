# load data and process
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .augmentation import OurAug
import random


raw_dataset_mapping = {'zeiss_img_1200': 'zeiss_img',
                       'uwf_md_1536_1210': 'uwf_md',
                       'uwf25264/raw_size896': 'uwf25264/raw',
                       '0002_anno_uwf_dr_20230913_data/Images_size1210_1536': '0002_anno_uwf_dr_20230913_data/Images'}


class ImgCrop(object):
    @staticmethod
    def crop_wf(raw_img):
        # 用于切蔡司广角图像

        L = raw_img.shape[0]

        #alpha = 1 / ((133 / 30) ** 0.5)
        alpha = 0.45

        l = int(L * alpha / 2)

        c1 = (int(L * 0.5), int(l))
        c2 = (int(L * 0.5), int(L - l))
        c3 = (int(L - l), int(L * 0.5))
        c4 = (int(l), int(L * 0.5))

        dis = int((2 - 2 ** 0.5 + alpha * 2 ** 0.5) / 4 * L)
        c5 = (dis, dis)
        c6 = (dis, L - dis)
        c7 = (L - dis, dis)
        c8 = (L- dis, L - dis)

        c9 = (int(L / 2), int(L / 2))
        centers = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
        cropped_imgs = []
        for _, c in enumerate(centers):
            c_x, c_y = c
            cropped_im = raw_img[c_y - l: c_y + l, c_x - l: c_x + l, :].copy()
            mask = np.zeros([2 * l, 2 * l], dtype=np.uint8)
            cv2.circle(mask, (l, l), l, 255, cv2.FILLED)
            cropped_im[mask==0] = 0
            cropped_imgs.append(cropped_im)
        return cropped_imgs, centers
    
    @staticmethod
    def crop_uwf(raw_img):
        # 用于切欧堡超广角图像
        h, w = raw_img.shape[:2]
        L = w
        centers = []
        alpha = 1 / 5
        l = int(L * alpha)
        
        ys = []
       
        x_starts = [2. * l, 1.1 * l, l]
        x_ends = [1.2 * l, l, l * 1.1]
        x_steps = [2, 3, 3]
        
        y_step = 3
        
        dy = (h - 2 * l)/ (y_step - 1)
        for i in range(y_step):
            y = int(l + i * dy)
            ys.append(y)
            x_range = w - x_ends[i] - x_starts[i]
            x_step = x_steps[i]
            dx = int(x_range / (x_step - 1))
            for j in range(x_step):

                centers.append((int(x_starts[i]) + int(dx * j), y))
        centers = sorted(centers, key=lambda x:x[0])
        centers = sorted(centers, key=lambda x:x[1])
        cropped_imgs = []
        for c in centers:
            c_x, c_y = c
            cropped_im = raw_img[c_y - l: c_y + l, c_x - l: c_x + l, :].copy()
            mask = np.zeros([2 * l, 2 * l], dtype=np.uint8)
            cv2.circle(mask, (l, l), l, 255, cv2.FILLED)
            cropped_im[mask==0] = 0
            cropped_imgs.append(cropped_im)

        return cropped_imgs, centers

    @staticmethod
    def center_uwf(raw_img):
        h, w = raw_img.shape[:2]
        L = w
        alpha = 45 / 200
        l = int(L * alpha)

        c_x = int(w / 2)
        c_y = int(h / 2)
        cropped_img = raw_img[c_y - l: c_y + l, c_x - l: c_x + l, :].copy()
        mask = np.zeros([2 * l, 2 * l], dtype=np.uint8)
        cv2.circle(mask, (l, l), l, 255, cv2.FILLED)
        cropped_img[mask==0] = 0

        return cropped_img

    @staticmethod
    def center_wf(raw_img):
        h, w = raw_img.shape[:2]
        L = w
        alpha = 45 / 133
        l = int(L * alpha)

        c_x = int(w / 2)
        c_y = int(h / 2)
        cropped_img = raw_img[c_y - l: c_y + l, c_x - l: c_x + l, :].copy()
        mask = np.zeros([2 * l, 2 * l], dtype=np.uint8)
        cv2.circle(mask, (l, l), l, 255, cv2.FILLED)
        cropped_img[mask==0] = 0

        return cropped_img

class CFPDataset(Dataset):
    def __init__(self, lstpath, img_root, num_class, mappings, aug: OurAug, ratio=1.) -> None:
        self.img_root = img_root
        self.aug = aug
        self.num_class = num_class
        self.mappings = mappings
        self.ratio = ratio

        self.lstpath = lstpath
        self.lst = []
        for _lstpath in self.lstpath:
            lst = self.read_list(_lstpath)
            self.lst += lst
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
        if len(line) == 3:
            dataset, img_name, labels = line 
            mask_labels = list(self.mappings.keys())
        else:
            dataset, img_name, labels, mask_labels = line
            mask_labels = mask_labels.split(',')

        labels = labels.split(',')
        

        img_path = os.path.join(self.img_root, dataset, img_name)
        if not os.path.exists(img_path):
            raise Exception(img_path)
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]

        size_h = self.aug.aug_cfg['instance_size_h']
        size_w = self.aug.aug_cfg['instance_size_w']

        img, _ = self.aug.process(img, size_h=size_h, size_w=size_w)
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))

        gt = np.zeros(self.num_class)
        for label in labels:
            if label not in self.mappings:
                continue
            gt[self.mappings[label]] = 1
        
        mask = np.zeros(self.num_class)
        for label in mask_labels:
            if label not in self.mappings:
                continue
            mask[self.mappings[label]] = 1


        return img_path, img, gt, mask

    def __len__(self):
        # return int(1e7)
        return len(self.lst)

class UWFDataset(CFPDataset):
    def __init__(self, lstpath, img_root, num_class, mappings, aug: OurAug, need_instance=True, WF=False, center_crop=False, get_raw_img=False, ratio=1.):
        CFPDataset.__init__(self, lstpath=lstpath, img_root=img_root, num_class=num_class, mappings=mappings, aug=aug, ratio=ratio)
        self.need_instance = need_instance
        self.WF = WF  # 区分是wf图像还是uwf图像
        self.center_crop = center_crop # 是否只切出中间区域
        self.get_raw_img = get_raw_img # 读取原始图像还是读取缩小后的图像以加快IO
        if self.get_raw_img:
            print('load raw image')
            print(raw_dataset_mapping)


    def __getitem__(self, index):
        line = self.lst[index]

        if len(line) == 3:
            dataset, img_name, labels = line 
            mask_labels = list(self.mappings.keys())
        else:
            dataset, img_name, labels, mask_labels = line
            mask_labels = mask_labels.split(',')
        if self.get_raw_img:
            dataset = raw_dataset_mapping[dataset]
        labels = labels.split(',')

        img_path = os.path.join(self.img_root, dataset, img_name)
        if not os.path.exists(img_path):
            raise Exception(img_path)
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
        for label in labels:
            if label not in self.mappings:
                continue
            gt[self.mappings[label]] = 1
        
        mask = np.zeros(self.num_class)
        for label in mask_labels:
            if label not in self.mappings:
                continue
            mask[self.mappings[label]] = 1

        return img_path, img, instances, gt, mask


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

