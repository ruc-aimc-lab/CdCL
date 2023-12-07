import numpy as np

import torch
from tqdm import tqdm


class Predictor(object):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def predict(self):
        self.model.set_model_mode('eval')
        eval_img_paths = []
        eval_scores = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class))
        eval_labels = np.zeros((self.dataloader.dataset.__len__(), self.dataloader.dataset.num_class), dtype=int)

        p = 0
        for data_in in tqdm(self.dataloader):
            img_paths, imgs, labels = data_in
            imgs = imgs.type(torch.FloatTensor).cuda()
            labels = labels.numpy().astype(int)

            batch_size = len(img_paths)
            
            scores = self.model.predict(imgs)
            scores = scores.data.cpu().numpy()

            eval_scores[p:p+batch_size, :] = scores

            eval_labels[p:p+batch_size] = labels
            eval_img_paths += list(img_paths)
            p += batch_size

        return eval_img_paths, eval_scores, eval_labels

