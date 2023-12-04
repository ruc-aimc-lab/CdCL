import os
from .dataset import MyDataset
from torch.utils.data import DataLoader
from .augmentation import MyAug
import json

def build_dataloader(paths, collection_names, training_params, mapping_path, augmentation_params, domain, train):
    lstpaths = []
    for collection in collection_names.split('+'):
        lstpath = os.path.join(paths['collection_root'], collection, 'Annotations', 'anno.txt')
        lstpaths.append(lstpath)
    im_root = paths['image_root']

    assert domain in ['source', 'target']
    augmentation_params['size_h'] = augmentation_params['{}_size_h'.format(domain)]
    augmentation_params['size_w'] = augmentation_params['{}_size_w'.format(domain)]

    if train:
        aug = MyAug(augmentation_params)
    else:
        aug = MyAug({'size_h': augmentation_params['size_h'],
                     'size_w': augmentation_params['size_w']})
    
    batch_size = training_params['batch_size_{}'.format(domain)]
    num_workers = training_params['num_workers']
    ratio = training_params.get('ratio', 1.)
    if ratio < 1:
        print('Use {}% of the data'.format(ratio * 100))

    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)
    
    assert training_params['model_params']['n_class'] == len(mappings)

    dataset = MyDataset(lstpaths=lstpaths, img_root=im_root, aug=aug, mappings=mappings, ratio=ratio)
    dataloader = DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=num_workers)

    return dataloader
