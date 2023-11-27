import os
import sys
import json

from nets import single_domain_build_model
from data import build_uwf_dataloader, build_cfp_dataloader
import numpy as np
import torch
from predictor import Predictor


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


def main(train_collection, val_collection, config_path, test_collection, run_num):
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']

    collection_root = paths['collection_root']
    mapping_path = paths['mapping_path']

    domain = training_params['domain'] # domain: {'cfp', 'uwf', 'wf}
    im_in = training_params['im_in'] # im_in: {'whole', 'instances', 'whole_instances'}

    if domain == 'cfp':
        test_loader = build_cfp_dataloader(
                paths=paths, training_params=training_params, 
                augmentation_params=augmentation_params, 
                collection_name=test_collection, 
                mapping_path=mapping_path, train=False)
    elif domain in ['uwf', 'wf']:
        if domain == 'uwf':
            wf_image = False
        else:
            wf_image = True
        test_loader = build_uwf_dataloader(
            paths=paths, training_params=training_params, 
            augmentation_params=augmentation_params, 
            collection_name=test_collection, 
            mapping_path=mapping_path, train=False, WF=wf_image)
    else:
        raise Exception('invalid domain: ', domain)
    
    inter_val = int(test_loader.dataset.__len__() / test_loader.batch_size) + 1
    training_params['inter_val'] = inter_val

    model = single_domain_build_model(training_params['net'], training_params)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if '__' in config_name:
        model_config_name = config_name.split('__')[0] + '.json'
    else:
        model_config_name = config_name
    model_path = os.path.join(collection_root + '_out', train_collection , 'Models', val_collection, model_config_name, 'runs_{}'.format(run_num), 'best_model.pkl')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.cuda()


    out_root = os.path.join(collection_root + '_out', test_collection, 'Predictions', train_collection,
                            val_collection, config_name, 'runs_{}'.format(run_num))
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    else:
        raise Exception('prediction \n{}\nalready exists'.format(out_root))

    predictor = Predictor(model, test_loader)
    img_names, scores, targets = predictor.predict_single_domain(domain, im_in)
    scores = weighted_sigmoid(scores)
    with open(os.path.join(out_root, 'results.csv'), 'w') as fout:
        for img_name, score in zip(img_names, scores):
            score = ','.join(list(map(lambda x: '{:.4f}'.format(x), score)))
            line = '{},{}\n'.format(img_name, score)
            fout.write(line)
    
if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    test_collection = sys.argv[4]
    run_num = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]
    main(train_collection, val_collection, config_path, test_collection, run_num)
   