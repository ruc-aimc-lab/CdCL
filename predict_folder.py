import os
import sys
import json

from nets import build_model
from data import build_uwf_dataloader_folder
import numpy as np
import torch
from predictor import Predictor


def folder2collection(folder_path):
    lst = os.listdir()


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


def main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, img_root, run_num):
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']

    collection_root = paths['collection_root']

    wf_image = training_params['wf_image']

    test_uwf_loader = build_uwf_dataloader_folder(
            img_root=img_root, training_params=training_params, 
            augmentation_params=augmentation_params, WF=wf_image)

    inter_val = int(test_uwf_loader.dataset.__len__() / test_uwf_loader.batch_size) + 1
    training_params['inter_val'] = inter_val

    model = build_model(training_params['net'], training_params)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model_path = os.path.join(collection_root + '_out', train_uwf_collection + '_' + train_cfp_collection, 'Models', val_uwf_collection, config_name, 'runs_{}'.format(run_num), 'best_model.pkl')
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.cuda()

    out_root = os.path.join(img_root + '_out', train_uwf_collection + '_' + train_cfp_collection, val_uwf_collection, config_name, 'runs_{}'.format(run_num))
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    else:
        pass
        #raise Exception('prediction \n{}\nalready exists'.format(out_root))

    predictor = Predictor(model, test_uwf_loader)
    img_names, scores, targets = predictor.predict()
    scores = weighted_sigmoid(scores)
    with open(os.path.join(out_root, 'results.csv'), 'w') as fout:
        for img_name, score in zip(img_names, scores):
            score = ','.join(list(map(lambda x: '{:.4f}'.format(x), score)))
            line = '{},{}\n'.format(img_name, score)
            fout.write(line)
    
if __name__ == '__main__':
    train_cfp_collection = sys.argv[1]
    train_uwf_collection = sys.argv[2]
    val_uwf_collection = sys.argv[3]
    config_path = sys.argv[4]
    img_root = sys.argv[5]
    run_num = sys.argv[6]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[7]
    main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, img_root, run_num)
   