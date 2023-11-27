import os
from .dataset import CFPDataset, UWFDataset, UWFDatasetFolder
from torch.utils.data import DataLoader
from .augmentation import OurAug
import json

def build_cfp_dataloader(paths, training_params, augmentation_params, collection_name, mapping_path, train):
    lstpaths = []
    for collection in collection_name.split('+'):
        lstpath = os.path.join(paths['collection_root'], collection, 'ImageSet', 'idx.txt')
        lstpaths.append(lstpath)
    im_root = paths['cfp_image_root']

    augmentation_params['whole_size_h'] = augmentation_params['source_size_h']
    augmentation_params['whole_size_w'] = augmentation_params['source_size_w']

    if train:
        aug = OurAug(augmentation_params)
    else:
        aug = OurAug({'instance_size_h': augmentation_params['instance_size_h'],
                      'instance_size_w':augmentation_params['instance_size_w'],
                      'whole_size_h': augmentation_params['whole_size_h'],
                      'whole_size_w': augmentation_params['whole_size_w']})
    num_class = training_params['n_class']
    batch_size = training_params['batch_size_cfp']
    num_workers = training_params['num_workers']

    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)

    dataset = CFPDataset(lstpath=lstpaths, img_root=im_root, aug=aug, num_class=num_class, mappings=mappings)
    dataloader = DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=num_workers)

    return dataloader



def build_uwf_dataloader(paths, training_params, augmentation_params, collection_name, mapping_path, train, WF=False):
    lstpaths = []
    for collection in collection_name.split('+'):
        lstpath = os.path.join(paths['collection_root'], collection, 'ImageSet', 'idx.txt')
        lstpaths.append(lstpath)
    im_root = paths['uwf_image_root']
    augmentation_params['whole_size_h'] = augmentation_params['target_size_h']
    augmentation_params['whole_size_w'] = augmentation_params['target_size_w']
    ratio = training_params.get('dataset_ratio', 1.)

    if train:
        aug = OurAug(augmentation_params)
    else:
        aug = OurAug({'instance_size_h': augmentation_params['instance_size_h'],
                      'instance_size_w':augmentation_params['instance_size_w'],
                      'whole_size_h': augmentation_params['whole_size_h'],
                      'whole_size_w': augmentation_params['whole_size_w']})
        ratio = 1
        
    num_class = training_params['n_class']
    batch_size = training_params['batch_size_uwf']
    num_workers = training_params['num_workers']
    need_instance = training_params['need_instance']
    center_crop = training_params.get('center_crop', False)
    get_raw_img = training_params.get('get_raw_img', False)
    
    if ratio != 1:
        print('ratio', ratio)

    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)
    if WF:
        print('dataloader for WF')
    else:
        print('dataloader for UWF')

    print('center_crop', center_crop)

    dataset = UWFDataset(lstpath=lstpaths, img_root=im_root, aug=aug, num_class=num_class, mappings=mappings, need_instance=need_instance, WF=WF, center_crop=center_crop, get_raw_img=get_raw_img, ratio=ratio)
    dataloader = DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=num_workers)

    return dataloader



def build_uwf_dataloader_folder(img_root, training_params, augmentation_params, WF=False):

    augmentation_params['whole_size_h'] = augmentation_params['target_size_h']
    augmentation_params['whole_size_w'] = augmentation_params['target_size_w']
    num_class = training_params['n_class']
    
    aug = OurAug({'instance_size_h': augmentation_params['instance_size_h'],
                    'instance_size_w':augmentation_params['instance_size_w'],
                    'whole_size_h': augmentation_params['whole_size_h'],
                    'whole_size_w': augmentation_params['whole_size_w']})
        
    batch_size = training_params['batch_size_uwf']
    num_workers = training_params['num_workers']
    need_instance = training_params['need_instance']
    center_crop = training_params.get('center_crop', False)
    
    if WF:
        print('dataloader for WF')
    else:
        print('dataloader for UWF')

    dataset = UWFDatasetFolder(img_root=img_root, aug=aug, num_class=num_class, need_instance=need_instance, WF=WF, center_crop=center_crop)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return dataloader

