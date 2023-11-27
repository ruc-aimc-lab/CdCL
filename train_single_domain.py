import os
import sys
import json
from nets import single_domain_build_model
from trainer_single_domain import Trainer
import warnings
warnings.filterwarnings("ignore")


def main(train_collection, val_collection, config_path):
    f = open(config_path, 'r').read()
    config = json.loads(f)

    paths = config['paths']
    training_params = config['training_params']
    augmentation_params = config['augmentation_params']
    paths['train_collection'] = train_collection
    paths['val_collection'] = val_collection
    paths['config_path'] = config_path

    model = single_domain_build_model(training_params['net'], training_params)
    model.cuda()
    print('finish model loading')

    trainer = Trainer(model=model,
                      paths=paths,
                      training_params=training_params,
                      augmentation_params=augmentation_params,
                      )
    trainer.train()
    return trainer.run_num


if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[4]

    main(train_collection, val_collection, config_path)

    #python train_zeiss.py mcs_train_8 final_zeiss_train_8 final_zeiss_val_8 configs/noatten_mil_cla12_5e4_L1consistency_38_coscycle4.json 0
    #
    #python train_zeiss.py mcs_train_8 zeiss_train_8 zeiss_val_8 configs/mil_1e3_cla12_1e4_L1consistency_softmax_38_coscycle4.json 0
