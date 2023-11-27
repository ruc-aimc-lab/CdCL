# model training
import os
import time
import torch
import torch.optim as optim

from utils import Evaluater
from data import build_cfp_dataloader, build_uwf_dataloader
from predictor import Predictor
import numpy as np


class Optimizer(object):
    def __init__(self, model, training_params, method='SGD', inter_val=0):
        self.lr = training_params['lr']
        self.weight_decay = training_params['weight_decay']
        if method == 'SGD':
            self.momentum = training_params['momentum']
            self.optim = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise Exception('')

        schedule_name = training_params['lr_schedule']
        schedule_params = training_params['schedule_params']
        if schedule_name == 'CosineAnnealingLR':
            schedule_params['T_max'] = inter_val * 4
        self.lr_schedule = getattr(torch.optim.lr_scheduler, schedule_name)(self.optim, **schedule_params)
        
    def update_lr(self):
        self.lr_schedule.step()
    
    def z_grad(self):
        self.optim.zero_grad()

    def g_step(self):
        self.optim.step()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

class Trainer(object):
    def __init__(self, model, paths, training_params, augmentation_params):
        
        
        self.training_params = training_params
        self.paths = paths
        self.augmentation_params = augmentation_params

        self.cfp_imroot = paths['cfp_image_root']
        self.uwf_imroot = paths['uwf_image_root']

        self.train_collection = paths['train_collection']
        self.val_collection = paths['val_collection']

        self.collection_root = paths['collection_root']
        self.config_path = paths['config_path']
        self.config_name = self.config_path.split(os.sep)[-1]

        self.mapping_path = paths['mapping_path']

        self.num_workers = self.training_params['num_workers']

        self.domain = self.training_params['domain'] # domain: {'cfp', 'uwf', 'wf}
        self.im_in = self.training_params['im_in'] # im_in: {'whole', 'instances', 'whole_instances'}

        self.evaluater = Evaluater()

        # 数据集
        if self.domain == 'cfp':
            self.train_loader = build_cfp_dataloader(
                paths=self.paths, training_params=self.training_params, 
                augmentation_params=self.augmentation_params, 
                collection_name=self.train_collection, 
                mapping_path=self.mapping_path, train=True)
            self.val_loader = build_cfp_dataloader(
                paths=self.paths, training_params=self.training_params, 
                augmentation_params=self.augmentation_params, 
                collection_name=self.val_collection, 
                mapping_path=self.mapping_path, train=False)

        elif self.domain in ['uwf', 'wf']:
            if self.domain == 'uwf':
                self.wf_image = False
            else:
                self.wf_image = True
            self.train_loader = build_uwf_dataloader(
                paths=self.paths, training_params=self.training_params, 
                augmentation_params=self.augmentation_params, 
                collection_name=self.train_collection, 
                mapping_path=self.mapping_path, train=True, WF=self.wf_image)

            self.val_loader = build_uwf_dataloader(
                paths=self.paths, training_params=self.training_params, 
                augmentation_params=self.augmentation_params, 
                collection_name=self.val_collection, 
                mapping_path=self.mapping_path, train=False, WF=self.wf_image)
        else:
            raise Exception('invalid domain: ', self.domain)
        
        self.model = model 
        self.inter_val = int(self.train_loader.dataset.__len__() / self.train_loader.batch_size) + 1

        self.optimizer = Optimizer(self.model, self.training_params, method=training_params['optimizer'], inter_val=self.inter_val)

        # 输出目录
        self.run_num = 0
        self.out = os.path.join(self.collection_root + '_out', self.train_collection , 'Models', self.val_collection, self.config_name, 'runs_{}'.format(self.run_num))
        while os.path.exists(self.out):
            self.run_num += 1
            self.out = os.path.join(self.collection_root + '_out', self.train_collection, 'Models', self.val_collection, self.config_name, 'runs_{}'.format(self.run_num))
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
        print('domain', self.domain)
        print('im_in', self.im_in)
        print('dataset: ', self.train_collection, self.val_collection)
        print('optimizer: {}'.format(training_params['optimizer']))

    def validate(self):
        print('validating...')
        predictor = Predictor(self.model, self.val_loader)
        _, scores, targets = predictor.predict_single_domain(self.domain, self.im_in)
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
        while True:
            if self.end:
                break
            for data_in in self.train_loader:
                if self.iteration % self.inter_val  == 0:
                    self.model.eval()
                    self.validate()
                    self.model.train()
                    if self.end:
                        break
                self.iteration += 1
                if self.domain == 'cfp':
                    _, img, target, mask = data_in
                elif self.domain in ['uwf', 'wf']:
                    _, img, crop_img, target, mask = data_in
                
                target = target.type(torch.FloatTensor).cuda()
                mask = mask.cuda()

                if self.im_in == 'whole':
                    img = img.type(torch.FloatTensor).cuda()
                    din = [img]
                elif self.im_in == 'instances':
                    crop_img = crop_img.type(torch.FloatTensor).cuda()
                    din = [crop_img]
                elif self.im_in == 'whole_instances':
                    img = img.type(torch.FloatTensor).cuda()
                    crop_img = crop_img.type(torch.FloatTensor).cuda()
                    din = [img, crop_img]
                
    
                self.optimizer.z_grad()
                if self.training_params.get('label_mask', False):
                    score, train_loss = self.model.train_model(din=din, gt=target, mask=mask)
                else:
                    score, train_loss = self.model.train_model(din=din, gt=target)
                
                train_loss.backward()
                self.optimizer.g_step()
                self.optimizer.z_grad()
                
                score = score.data.cpu().numpy()
                target = target.data.cpu().numpy()

                hist, precisions, recalls, fs, specificities, aps, iaps, aucs = self.evaluater.evaluate(score, target)
                precisions, recalls, fs, specificities, aps, iaps, aucs = np.nanmean(precisions), np.nanmean(recalls), np.nanmean(fs), np.nanmean(specificities), np.nanmean(aps), np.nanmean(iaps), np.nanmean(aucs)

                total_time = time.time() - self.start_time
                print('iteration {:d}, loss={:.3f}, lr={:.3e}, ap={:.3f}, max_ap={:.3f}, no_improve:{:d}'.format(
                    self.iteration, train_loss.data.item(), self.optimizer.get_lr(), aps, self.ap, self.no_improve))
            
                with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                    log_iter = [self.iteration, train_loss.data.item()]
                    log_train = [aps, fs, precisions, recalls]
                    log_test = [''] * 4

                    log_iter = ','.join(list(map(str, log_iter)))
                    log_test = ','.join(log_test)
                    log_train = ','.join(list(map(lambda x: '{:.4f}'.format(x), log_train)))

                    log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
                    f.write(log)
                self.optimizer.update_lr()
