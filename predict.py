import os
import sys
import json

from models import build_model
from dataloader import build_dataloader
import numpy as np
from predictor import Predictor
from utils import list2str


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


def main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num):
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']

    mapping_path = paths['mapping_path']

    test_target_loader = build_dataloader(
        paths=paths, collection_names=test_target_collection, 
        training_params=training_params, mapping_path=mapping_path, 
        augmentation_params=augmentation_params, 
        domain='target', train=False)

    model = build_model(training_params['net'], training_params, training=False)
    model.requires_grad_false()
    model.set_model_mode('eval')
    model_path = os.path.join('out', train_target_collection + '_' + train_source_collection, 'Models', val_target_collection, config_name, 'runs_{}'.format(run_num), 'best_model.pkl')
    model.load_model(model_path)
    model.set_device('cuda')

    out_root = os.path.join('out', test_target_collection, 'Predictions', train_target_collection + '_' + train_source_collection,
                            val_target_collection, config_name, 'runs_{}'.format(run_num))
    results_path = os.path.join(out_root, 'results.csv')
    if os.path.exists(results_path):
        raise Exception('Prediction \n{}\nalready exists'.format(results_path))
    else:
        os.makedirs(out_root, exist_ok=True)
   
    predictor = Predictor(model, test_target_loader)
    img_names, scores, _ = predictor.predict()
    scores = weighted_sigmoid(scores)
    
    with open(results_path, 'w') as fout:
        for img_name, score_arr in zip(img_names, scores):
            score_str = list2str(lst=score_arr, decimals=4, separator=',')
            line = '{},{}\n'.format(img_name, score_str)
            fout.write(line)
    print('Predictions saved to {}'.format(results_path))
    return results_path
    
if __name__ == '__main__':
    train_source_collection = sys.argv[1]
    train_target_collection = sys.argv[2]
    val_target_collection = sys.argv[3]
    config_path = sys.argv[4]
    test_target_collection = sys.argv[5]
    run_num = sys.argv[6]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[7]
    main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num)
   