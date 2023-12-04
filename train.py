import os
import sys
import json
from trainer import Trainer



def main(source_train_collection, target_train_collection, target_val_collection, config_path):
    f = open(config_path, 'r').read()
    config = json.loads(f)

    paths = config['paths']
    training_params = config['training_params']
    augmentation_params = config['augmentation_params']
    paths['source_train_collection'] = source_train_collection
    paths['target_train_collection'] = target_train_collection
    paths['target_val_collection'] = target_val_collection
    paths['config_path'] = config_path

    trainer = Trainer(paths=paths,
                      training_params=training_params,
                      augmentation_params=augmentation_params,
                      )
    trainer.train()
    return trainer.run_num


if __name__ == '__main__':
    train_cfp_collection = sys.argv[1]
    train_uwf_collection = sys.argv[2]
    val_collection = sys.argv[3]
    config_path = sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

    main(train_cfp_collection, train_uwf_collection, val_collection, config_path)

    #python train_zeiss.py mcs_train_8 final_zeiss_train_8 final_zeiss_val_8 configs/noatten_mil_cla12_5e4_L1consistency_38_coscycle4.json 0
    #
    #python train_zeiss.py mcs_train_8 zeiss_train_8 zeiss_val_8 configs/mil_1e3_cla12_1e4_L1consistency_softmax_38_coscycle4.json 0
