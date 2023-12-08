

config_path = 'configs/cdcl_effi_b3_p.json'
model_path = 'cdcl_effi_b3_p-90a78584.pkl'

import os
import json

from models import build_model
from dataloader import build_dataloader
import numpy as np
from predictor import Predictor
from evaluate import load_gt
from utils import Evaluater

def main(model_path, config_path, collection_path):
    with open(config_path, 'r') as fin:
        config = json.load(fin)

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']
    mapping_path = paths['mapping_path']

    test_target_loader = build_dataloader(
        paths=paths, collection_names=collection_path, 
        training_params=training_params, mapping_path=mapping_path, 
        augmentation_params=augmentation_params, 
        domain='target', train=False)

    model = build_model(training_params['net'], training_params, training=False)
    model.requires_grad_false()
    model.set_model_mode('eval')
    model.load_model(model_path)
    model.set_device('cuda')
   
    predictor = Predictor(model, test_target_loader)
    img_names, scores, _ = predictor.predict()
    # scores = weighted_sigmoid(scores)
    
    gt_dic = load_gt(gt_path=collection_path, mappings=test_target_loader.dataset.mappings)


    
