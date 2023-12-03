# model training
import os
import time
import torch
from nets import build_model

from utils import Evaluater
from data import build_cfp_dataloader, build_uwf_dataloader
from predictor import Predictor
import numpy as np

class Trainer(object):
    def __init__(self, paths, training_params, augmentation_params):
        self.training_params = training_params
        self.paths = paths
        self.augmentation_params = augmentation_params

        self.imroot = paths['image_root']

        self.train_cfp_collection = paths['source_train_collection']
        self.train_uwf_collection = paths['target_train_collection']
        self.val_uwf_collection = paths['target_val_collection']

        self.collection_root = paths['collection_root']
        self.config_path = paths['config_path']
        self.config_name = self.config_path.split(os.sep)[-1]

        self.mapping_path = paths['mapping_path']

        self.num_workers = self.training_params['num_workers']

        # self.target_type = self.training_params['target_type']  # 用来区分输入的是uwf还是wf
        
        self.evaluater = Evaluater()
        # 数据集
        self.train_cfp_loader = build_dataloader(
            paths=self.paths, training_params=self.training_params, 
            augmentation_params=self.augmentation_params, 
            collection_name=self.train_cfp_collection, 
            mapping_path=self.mapping_path, train=True)
        self.train_uwf_loader = build_uwf_dataloader(
            paths=self.paths, training_params=self.training_params, 
            augmentation_params=self.augmentation_params, 
            collection_name=self.train_uwf_collection, 
            mapping_path=self.mapping_path, train=True, WF=self.wf_image)
        self.val_uwf_loader = build_uwf_dataloader(
            paths=self.paths, training_params=self.training_params, 
            augmentation_params=self.augmentation_params, 
            collection_name=self.val_uwf_collection, 
            mapping_path=self.mapping_path, train=False, WF=self.wf_image)
        

        self.inter_val = int(self.train_uwf_loader.dataset.__len__() / self.train_uwf_loader.batch_size) + 1
        self.training_params['inter_val'] = self.inter_val
        self.model = build_model(training_params['net'], training_params)
        self.model.cuda()
        print('finish model loading')
        # self.optimizer = Optimizer(self.model, self.training_params, method=training_params['optimizer'])

        # 输出目录
        self.run_num = 0
        self.out = os.path.join(self.collection_root + '_out', self.train_uwf_collection + '_' + self.train_cfp_collection, 'Models', self.val_uwf_collection, self.config_name, 'runs_{}'.format(self.run_num))
        while os.path.exists(self.out):
            self.run_num += 1
            self.out = os.path.join(self.collection_root + '_out', self.train_uwf_collection + '_' + self.train_cfp_collection, 'Models', self.val_uwf_collection, self.config_name, 'runs_{}'.format(self.run_num))
        os.makedirs(os.path.join(self.out, 'models'))

        # log表头
        self.log_headers = ['iteration', 'train/loss', 
        'train/ap', 'train/f1', 'train/precision', 'train/recall', 
        'valid/ap', 'valid/f1', 'valid/precision', 'valid/recall', 
        'total_time']

        with open(os.path.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

        
        self.iteration = 0
        self.ap = -1
        self.no_improve = 0
        self.start_time = time.time()
        self.end = False

        print('model: {}'.format(training_params['net']))
        print('dataset: ', self.train_cfp_collection, self.train_uwf_collection, self.val_uwf_collection)
        print('optimizer: {}'.format(training_params['optimizer']))

    def validate(self):
        print('validating...')
        predictor = Predictor(self.model, self.val_uwf_loader)
        _, scores, targets = predictor.predict()
        hist, precisions, recalls, fs, specificities, aps, iaps, aucs = self.evaluater.evaluate(scores, targets)
        precisions, recalls, fs, specificities, aps, iaps, aucs = np.nanmean(precisions), np.nanmean(recalls), np.nanmean(fs), np.nanmean(specificities), np.nanmean(aps), np.nanmean(iaps), np.nanmean(aucs)

        with open(os.path.join(self.out, 'log.csv'), 'a') as f:
            log_iter = [self.iteration, '']
            log_train = [''] * 4
            log_test = [aps, fs, precisions, recalls]
            total_time = time.time() - self.start_time

            log_iter = ','.join(list(map(str, log_iter)))
            log_train = ','.join(log_train)
            log_test = ','.join(list(map(lambda x: '{:.4f}'.format(x), log_test)))

            log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
            f.write(log)
        
        is_best = aps > self.ap
        if is_best:
            self.ap = aps
            self.no_improve = 0
            torch.save(self.model.state_dict(), os.path.join(self.out, 'best_model.pkl'))
            print('model saved')
        else:
            self.no_improve += 1

        if self.no_improve >= 10:
            self.end = True

    def train(self):
        self.iter_cfp_train_loader = iter(self.train_cfp_loader)
        self.iter_uwf_train_loader = iter(self.train_uwf_loader)
        while True:
            try:      
                data_in_uwf = next(self.iter_uwf_train_loader)
            except StopIteration:
                print('new_batch, 1')
                self.iter_uwf_train_loader = iter(self.train_uwf_loader)
                print(2)
                data_in_uwf = next(self.iter_uwf_train_loader)
                print(3)

            try:      
                data_in_cfp = next(self.iter_cfp_train_loader)
            except StopIteration:
                self.iter_cfp_train_loader = iter(self.train_cfp_loader)
                data_in_cfp = next(self.iter_cfp_train_loader)

            if self.iteration % self.inter_val  == 0:
                self.model.eval()
                self.validate()
                self.model.train()
                if self.end:
                    break
            self.iteration += 1

            _, cfp_img, cfp_target, cfp_mask = data_in_cfp
            _, uwf_whole_img, uwf_crop_img, uwf_target, uwf_mask = data_in_uwf

            cfp_img = cfp_img.type(torch.FloatTensor).cuda()
            cfp_target = cfp_target.type(torch.FloatTensor).cuda()

            uwf_whole_img = uwf_whole_img.type(torch.FloatTensor).cuda()
            uwf_crop_img = uwf_crop_img.type(torch.FloatTensor).cuda()
            uwf_target = uwf_target.type(torch.FloatTensor).cuda()

            uwf_score, train_loss, detailed_losses, detailed_scores = \
            self.model.train_model(cfp=cfp_img, clarus_whole=uwf_whole_img, clarus_split=uwf_crop_img, gt_cfp=cfp_target, gt_clarus=uwf_target, iter_num=self.iteration)
            
            uwf_score = uwf_score.data.cpu().numpy()
            uwf_target = uwf_target.data.cpu().numpy()

            if len(uwf_target) > len(uwf_score):
                uwf_target = uwf_target[:len(uwf_score)]
            hist, precisions, recalls, fs, specificities, aps, iaps, aucs = self.evaluater.evaluate(uwf_score, uwf_target)
            precisions, recalls, fs, specificities, aps, iaps, aucs = np.nanmean(precisions), np.nanmean(recalls), np.nanmean(fs), np.nanmean(specificities), np.nanmean(aps), np.nanmean(iaps), np.nanmean(aucs)

            total_time = time.time() - self.start_time
            print('iteration {:d}, loss={:.3f}, lr={:.3e}, ap={:.3f}, max_ap={:.3f}, no_improve:{:d}'.format(
                self.iteration, train_loss.data.item(), self.model.opt.get_lr(), aps, self.ap, self.no_improve))
           
            with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                log_iter = [self.iteration, train_loss.data.item()]
                log_train = [aps, fs, precisions, recalls]
                log_test = [''] * 4

                log_iter = ','.join(list(map(str, log_iter)))
                log_test = ','.join(log_test)
                log_train = ','.join(list(map(lambda x: '{:.4f}'.format(x), log_train)))

                log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
                f.write(log)
            

