import os
import json

from models import build_model
from dataloader import build_dataloader
import numpy as np
from predictor import Predictor
from evaluate import load_gt
from utils import Evaluater

model_path = 'cdcl_effi_b3_p-90a78584.pkl'  # The path of your checkpoint
config_path = 'configs/cdcl_effi_b3_p.json'
collection_name = 'TOP_test'

with open(config_path, 'r') as fin:
    config = json.load(fin)

training_params = config['training_params']
paths = config['paths']
augmentation_params = config['augmentation_params']
mapping_path = paths['mapping_path']

test_target_loader = build_dataloader(
    paths=paths, collection_names=collection_name, 
    training_params=training_params, mapping_path=mapping_path, 
    augmentation_params=augmentation_params, 
    domain='target', train=False)

model = build_model(training_params['net'], training_params, training=False)
model.requires_grad_false()
model.set_model_mode('eval')
model.load_model(model_path)
model.set_device('cuda')

predictor = Predictor(model, test_target_loader)
img_paths, scores, _ = predictor.predict()
# scores = weighted_sigmoid(scores)
mappings = test_target_loader.dataset.mappings
reserve_mappings = {}
for k in mappings:
    reserve_mappings[mappings[k]] = k

gt_dic = load_gt(gt_path=test_target_loader.dataset.lstpaths[0], mapping=mappings)

gts = []
preds = []
for im_path, score_arr in zip(img_paths, scores):
    im_name = im_path.split(os.sep)[-1]
    gts.append(gt_dic[im_name])
    preds.append(score_arr)
gts = np.array(gts)
preds = np.array(preds)

evaluater = Evaluater()

_, precisions, recalls, fs, specificities, aps, iaps, aucs = evaluater.evaluate(preds, gts, thre=0.)

for i in range(len(aps)):
    print('Disease: {}, AP: {:.4f}'.format(reserve_mappings[i], aps[i]))    
print('Mean AP: {:.4f}'.format(np.mean(aps)))  
    
