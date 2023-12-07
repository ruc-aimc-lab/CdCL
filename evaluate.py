import sys
import os
import json
import numpy as np
from utils import Evaluater, list2str


def load_pred(pred_path):
    with open(pred_path) as fin:
        lines = fin.readlines()
    pred_dic = {}
    for line in lines:
        line = line.strip('\r\n').split(',')
        im_name = line[0]
        im_name = im_name.split(os.sep)[-1]
        score = line[1:]
        score = list(map(float, score))
        pred_dic[im_name] = score
    return pred_dic


def load_gt(gt_path, mapping):
    with open(gt_path) as fin:
        lines = fin.readlines()
    n_class = len(mapping)
    gt_dic = {}
    for line in lines:
        line = line.strip('\r\n').split('\t')
        im_name = line[1]

        labels = np.zeros(n_class, dtype=int)
        
        for l in line[2].split(','):
            if l not in mapping:
                continue
            labels[mapping[l]] = 1
        gt_dic[im_name] = labels
    return gt_dic


def main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num):
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    paths = config['paths']
    collection_root = paths['collection_root']

    pred_path = os.path.join('./out', test_target_collection, 'Predictions', train_target_collection + '_' + train_source_collection,
                            val_target_collection, config_name, 'runs_{}'.format(run_num), 'results.csv')
    gt_path = os.path.join(collection_root, test_target_collection, 'Annotations', 'anno.txt')
    
    mapping_path = paths['mapping_path']
    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)
        reserve_mappings = {}
        for k in mappings:
            reserve_mappings[mappings[k]] = k

    gt_dic = load_gt(gt_path, mappings)
    pred_dic = load_pred(pred_path)

    gts = []
    preds = []
    for im_name in gt_dic:
        gts.append(gt_dic[im_name])
        preds.append(pred_dic[im_name])
    gts = np.array(gts)
    preds = np.array(preds)

    evaluater = Evaluater()

    _, precisions, recalls, fs, specificities, aps, iaps, aucs = evaluater.evaluate(preds, gts, thre=0.5)
    row_heads = ['precision', 'recall', 'f1', 'specificity', 'auc', 'ap']
    column_head = ['mean'] + [reserve_mappings[i] for i in range(len(precisions))]
    results = [precisions, recalls, fs, specificities, aucs, aps]

    out_path = os.path.join('./out', test_target_collection, 'Predictions', train_target_collection + '_' + train_source_collection,
                            val_target_collection, config_name, 'runs_{}'.format(run_num), 'eval_results.csv')
    with open(out_path, 'w') as fout:
        column_head = ','.join(column_head)
        fout.write(',{}\n'.format(column_head))

        for i in range(len(row_heads)):
            row_head = row_heads[i]
            result = results[i]
            result_mean = np.nanmean(result)
            result = list2str(lst=result, decimals=4, separator=',')
            fout.write('{},{:.4f},{}\n'.format(row_head, result_mean, result))
    print('Evaluation results saved to {}'.format(out_path))
    return out_path


if __name__ == '__main__':
    train_source_collection = sys.argv[1]
    train_target_collection = sys.argv[2]
    val_target_collection = sys.argv[3]
    config_path = sys.argv[4]
    test_target_collection = sys.argv[5]
    run_num = sys.argv[6]
    main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num)
    

