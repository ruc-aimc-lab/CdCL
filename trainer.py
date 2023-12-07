# model training
import os
import time
import torch
import numpy as np

from models import build_model
from utils import Evaluater, list2str
from dataloader import build_dataloader
from predictor import Predictor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Trainer(object):
    def __init__(self, paths, training_params, augmentation_params):
        self.training_params = training_params
        self.paths = paths
        self.augmentation_params = augmentation_params

        self.imroot = paths['image_root']

        self.train_source_collection = paths['source_train_collection']
        self.train_target_collection = paths['target_train_collection']
        self.val_target_collection = paths['target_val_collection']

        self.collection_root = paths['collection_root']
        self.config_path = paths['config_path']
        self.config_name = self.config_path.split(os.sep)[-1]

        self.mapping_path = paths['mapping_path']

        self.num_workers = self.training_params['num_workers']

        self.evaluater = Evaluater()

        self.train_source_loader = build_dataloader(
            paths=self.paths, collection_names=self.train_source_collection, 
            training_params=self.training_params, mapping_path=self.mapping_path,
            augmentation_params=self.augmentation_params, 
            domain='source', train=True)
        self.train_target_loader = build_dataloader(
            paths=self.paths, collection_names=self.train_target_collection, 
            training_params=self.training_params, mapping_path=self.mapping_path, 
            augmentation_params=self.augmentation_params, 
            domain='target', train=True)
        self.val_target_loader = build_dataloader(
            paths=self.paths, collection_names=self.val_target_collection, 
            training_params=self.training_params, mapping_path=self.mapping_path, 
            augmentation_params=self.augmentation_params, 
            domain='target', train=False)
        
        self.inter_val = int(self.train_target_loader.dataset.__len__() / self.train_target_loader.batch_size + 0.5)
        self.training_params['inter_val'] = self.inter_val
        
        self.model = build_model(training_params['net'], training_params, training=True)
        self.model.set_device('cuda')
        print('finish model loading')

        # output folder
        self.run_num = 0  # We run multiple times with the same settings.
        self.out = os.path.join('./out', self.train_target_collection + '_' + self.train_source_collection, 
                                'Models', self.val_target_collection, self.config_name, 
                                'runs_{}'.format(self.run_num))
        while os.path.exists(self.out):
            self.run_num += 1
            self.out = os.path.join('./out', self.train_target_collection + '_' + self.train_source_collection, 
                                    'Models', self.val_target_collection, self.config_name, 
                                    'runs_{}'.format(self.run_num))
        os.makedirs(self.out)

        # log head
        self.log_headers = ['iteration', 'train/loss', 
                            'train/ap', 'train/f1', 'train/precision', 'train/recall', 
                            'valid/ap', 'valid/f1', 'valid/precision', 'valid/recall',
                            'total_time']

        with open(os.path.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

        self.iteration = 0
        self.best_ap = -1
        self.no_improve = 0
        self.start_time = time.time()
        self.end = False  # whether to stop training

        print('model: {}'.format(training_params['net']))
        print('dataset: ', self.train_source_collection, self.train_target_collection, self.val_target_collection)
        print('optimizer: {}'.format(training_params['optimizer']))

    def validate(self):
        print('validating...')
        predictor = Predictor(self.model, self.val_target_loader)
        _, scores, labels = predictor.predict()
        _, precisions, recalls, fs, specificities, aps, iaps, aucs = self.evaluater.evaluate(scores, labels)
        precisions, recalls, fs, specificities, aps, iaps, aucs = np.nanmean(precisions), np.nanmean(recalls), np.nanmean(fs), np.nanmean(specificities), np.nanmean(aps), np.nanmean(iaps), np.nanmean(aucs)

        with open(os.path.join(self.out, 'log.csv'), 'a') as f:
            log_iter = [self.iteration, '']
            log_train = [''] * 4
            log_test = [aps, fs, precisions, recalls]
            total_time = time.time() - self.start_time

            log_iter = ','.join(map(str, log_iter))
            log_train = ','.join(log_train)
            log_test = list2str(lst=log_test, decimals=4, separator=',')

            log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
            f.write(log)
        
        if aps > self.best_ap:
            self.best_ap = aps
            self.no_improve = 0
            self.model.save_model(os.path.join(self.out, 'best_model.pkl'))
            print('model saved')
        else:
            self.no_improve += 1

        if self.no_improve >= 10:
            self.end = True

    def train(self):
        self.iter_source_train_loader = iter(self.train_source_loader)
        self.iter_target_train_loader = iter(self.train_target_loader)
        while True:
            try:      
                data_in_target = next(self.iter_target_train_loader)
            except StopIteration:
                self.iter_target_train_loader = iter(self.train_target_loader)
                data_in_target = next(self.iter_target_train_loader)

            try:      
                data_in_source = next(self.iter_source_train_loader)
            except StopIteration:
                self.iter_source_train_loader = iter(self.train_source_loader)
                data_in_source = next(self.iter_source_train_loader)

            if self.iteration % self.inter_val  == 0:
                self.model.set_model_mode('eval')
                self.validate()
                self.model.set_model_mode('train')
                
                if self.end:
                    break
            self.iteration += 1

            _, source_x, source_y = data_in_source
            _, target_x, target_y = data_in_target
            source_x = source_x.type(torch.FloatTensor).cuda()
            source_y = source_y.type(torch.FloatTensor).cuda()
            target_x = target_x.type(torch.FloatTensor).cuda()
            target_y = target_y.type(torch.FloatTensor).cuda()

            target_score, train_loss = self.model.fit(source_x=source_x, target_x=target_x, 
                                                      source_y=source_y, target_y=target_y)
            
            target_score = target_score.data.cpu().numpy()
            target_y = target_y.data.cpu().numpy()

            if len(target_y) > len(target_score):
                target_y = target_y[:len(target_score)]
            _, precisions, recalls, fs, specificities, aps, iaps, aucs = self.evaluater.evaluate(target_score, target_y)
            precisions, recalls, fs, specificities, aps, iaps, aucs = \
                np.nanmean(precisions), np.nanmean(recalls), np.nanmean(fs), np.nanmean(specificities), np.nanmean(aps), np.nanmean(iaps), np.nanmean(aucs)

            total_time = time.time() - self.start_time

            """
            It is ok to receive aps=nan here. When all the labels in a batch are the same, AP cannot be calculated.
            """
            print('iteration {:d}, loss={:.3f}, lr={:.3e}, ap={:.3f}, max_ap={:.3f}, no_improve:{:d}'.format(
                self.iteration, train_loss.data.item(), self.model.opt.get_lr(), aps, self.best_ap, self.no_improve))

            with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                log_iter = [self.iteration, train_loss.data.item()]
                log_train = [aps, fs, precisions, recalls]
                log_test = [''] * 4

                log_iter = ','.join(map(str, log_iter))
                log_test = ','.join(log_test)
                log_train = list2str(lst=log_train, decimals=4, separator=',')

                log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
                f.write(log)
            

