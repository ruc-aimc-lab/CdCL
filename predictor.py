import numpy as np

import torch
from tqdm import tqdm


class Predictor(object):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def predict(self):
        self.model.eval()
        eval_im_names = []
        eval_scores = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class))
        eval_targets = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class), dtype=int)

        p = 0
        for i, data_in in tqdm(enumerate(self.dataloader)):
            img_name, whole_img, crop_img, targets, _ = data_in

            whole_img = whole_img.type(torch.FloatTensor).cuda()
            crop_img = crop_img.type(torch.FloatTensor).cuda()

            targets = targets.numpy().astype(int)

            batch_size = len(img_name)
            
            scores = self.model.predict_result(whole_img, crop_img)
            scores = scores.data.cpu().numpy()

            eval_scores[p:p+batch_size, :] = scores

            eval_targets[p:p+batch_size] = targets
            eval_im_names += img_name
            p += batch_size

        return eval_im_names, eval_scores, eval_targets

    def predict_single_domain(self, domain, im_in):
        self.model.eval()
        eval_im_names = []
        eval_scores = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class))
        eval_targets = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class), dtype=int)

        p = 0
        for i, data_in in tqdm(enumerate(self.dataloader)):
            if domain == 'cfp':
                img_name, img, targets, mask = data_in
            elif domain in ['uwf', 'wf']:
                img_name, img, crop_img, targets, mask = data_in
            
            if im_in == 'whole':
                img = img.type(torch.FloatTensor).cuda()
                din = [img]
            elif im_in == 'instances':
                crop_img = crop_img.type(torch.FloatTensor).cuda()
                din = [crop_img]
            elif im_in == 'whole_instances':
                img = img.type(torch.FloatTensor).cuda()
                crop_img = crop_img.type(torch.FloatTensor).cuda()
                din = [img, crop_img]

            targets = targets.numpy().astype(int)

            batch_size = len(img_name)
            
            scores = self.model(din)
            scores = scores.data.cpu().numpy()

            eval_scores[p:p+batch_size, :] = scores

            eval_targets[p:p+batch_size] = targets
            eval_im_names += img_name
            p += batch_size

        return eval_im_names, eval_scores, eval_targets

