import os
import cv2
import json
import shutil
import random

from nets import VITFOVNetTraining, build_backbone, VITMixupWholeTraining
from data import build_uwf_dataloader, build_cfp_dataloader
import numpy as np
import torch
import torch.nn as nn

def mix(cfp, uwf):
    ratio = 0.7
    w_cfp = cfp.shape[1]
    w_uwf = uwf.shape[1]
    dw = int((w_uwf - w_cfp) / 2)
    if w_uwf != w_cfp:
        new_cfp = np.zeros_like(uwf)
        new_cfp[:, dw:dw+w_cfp, :] = cfp
    else:
        new_cfp = cfp

    mix = (1 - ratio) * new_cfp + ratio * uwf
    return mix

def pad_cfp(cfp, clarus_whole):
    # 超广角图像可能不是正方形的，所以把cfp padding以下，以便mix up
    min_n = min(cfp.size(0), clarus_whole.size(0))
    cfp = cfp[:min_n]
    clarus_whole = clarus_whole[:min_n]

    w_cfp = cfp.size(3)
    w_clarus_whole = clarus_whole.size(3)
    dw = int((w_clarus_whole - w_cfp) / 2)
    if w_clarus_whole != w_cfp:
        new_cfp = torch.zeros_like(clarus_whole)
        new_cfp[:, :, :, dw:dw+w_cfp] = cfp
    else:
        new_cfp = cfp
    return new_cfp

def load_gt(collection, mappings):
    label_idx = {}
    with open(os.path.join('VisualSearch', collection, 'Annotations', 'anno.txt')) as fin:
        lines = fin.readlines()
    for line in lines:
        folder, img_name, labels = line.strip('\r\n').split('\t')
        labels = labels.split(',')
        for l in labels:
            if l not in mappings:
                continue
            i = mappings[l]
            label_idx[i] = label_idx.get(i, []) + [(folder, img_name)]
    return label_idx

def load_img(img_lst, h, w):
    img_root = '../imgdata'
    imgs = []
    paths = []
    for folder, img_name in img_lst:
        im_path = os.path.join(img_root, folder, img_name)

        img = cv2.imread(im_path)[:, :, [2, 1, 0]]
        img = cv2.resize(img, (w, h))
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
        paths.append(im_path)
    imgs = torch.tensor(np.array(imgs))
    imgs =  imgs.type(torch.FloatTensor).cuda()
    return imgs, paths

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# config_path = 'configs/uwf_fovnet_vit_mix_mean_whole_sfusion_mhsa2_ratio4_effi_b3_p.json'
# config_path = 'configs/wf_fovnet_vit_mix_mean_whole_sfusion_mhsa2_ratio3_effi_b3_p.json'
config_path = 'configs_public/public_uwf_fovnet_vit_mix_mean_mhsa2_ratio4_effi_b3_p_r01.json'

# mapping_path = 'VisualSearch/mapping_private_dataset.json'
mapping_path = 'VisualSearch/mapping_public_dataset.json'

uwf_collection = 'test_uwf_TOP'
# model_path = 'VisualSearch_out/train_uwf_md_train_cfp_md/Models/val_uwf_md/uwf_fovnet_vit_mix_mean_whole_sfusion_mhsa2_ratio4_effi_b3_p.json/runs_3/best_model.pkl'
# model_path = 'VisualSearch_out/train_wf_zeiss_train_cfp_md/Models/val_wf_zeiss/wf_fovnet_vit_mix_mean_whole_sfusion_mhsa2_ratio3_effi_b3_p.json/runs_0/best_model.pkl'
model_path = 'VisualSearch_out/train_uwf_TOP_train_cfp_RFMiD/Models/val_uwf_TOP/public_uwf_fovnet_vit_mix_mean_whole_sfusion_mhsa2_ratio4_effi_b3_p_r01.json/runs_6/best_model.pkl'

w=650


out_root = './scores__{}'.format(uwf_collection) 


with open(config_path, 'r') as fin:
    config = json.load(fin)
config['training_params']['batch_size_cfp'] = 1
config['training_params']['batch_size_uwf'] = 1

training_params = config['training_params']
paths = config['paths']
augmentation_params = config['augmentation_params']
collection_root = paths['collection_root']
mapping_path = paths['mapping_path']
wf_image = training_params['wf_image']
training_params['inter_val'] = 100000

backbone = build_backbone(model_name='effi_b3_p', pretrained=False)
model = VITMixupWholeTraining(backbone=backbone, n_class=3, channels=1536, 
                            crit_sup=nn.BCEWithLogitsLoss(),
                            weights=[],
                            mhsa_nums=2,
                            mil_ratio=4,
                            training_params=training_params,
                            use_mean=True,
                            mix_ratio=None,
                            score_fusion='single',
                            over_lap=True)
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.cuda()
model.load_state_dict(torch.load(model_path))


with open(mapping_path, 'r') as fin:
    mappings = fin.read()
    mappings = json.loads(mappings)

idx_uwf = load_gt(uwf_collection, mappings)

for i in idx_uwf:
    
    # uwf_lst = random.sample(idx_uwf[i], 20)
    uwf_lst = idx_uwf[i]
    
    for k in range(len(uwf_lst)):
        uwfs, uwf_paths = load_img(uwf_lst[k:k+1], h=512, w=w)
        uwf = uwfs

        score, score_vit, score_whole, weight_fusion, attns = model.draw_heat(uwf)
        
        attns1 = attns[0].data.cpu().numpy()[0]
        attns2 = attns[1].data.cpu().numpy()[0]
        score = score.data.cpu().numpy()[0]
        score_vit = score_vit.data.cpu().numpy()[0]
        score_whole = score_whole.data.cpu().numpy()[0]
        weight_fusion = weight_fusion.data.cpu().numpy()[0][0]

        os.makedirs(os.path.join(out_root, '{}'.format(i), '{}'.format(k), 'heat'), exist_ok=True)

        #shutil.copy(uwf_paths[k], os.path.join(out_root, '{}'.format(i), '{}'.format(k), uwf_paths[k].split(os.sep)[-1]))
        #np.save(os.path.join(out_root, '{}'.format(i), '{}'.format(k), 'heat', 'att1.npy'), attns1)
        #np.save(os.path.join(out_root, '{}'.format(i), '{}'.format(k), 'heat', 'att2.npy'), attns2)

        with open(os.path.join(out_root, '{}'.format(i), '{}'.format(k), 'scores_weights.txt'), 'w') as fout:
            fout.write('total_score\tscore_vit\tscore_whole\tweight_fusion\n')
            for mat in [score, score_vit, score_whole, weight_fusion]:
                line = ','.join(list(map(str, mat))) + '\n'
                fout.write(line)
