import os
import sys
import json
from trainer import Trainer


def main(train_source_collection, train_target_collection, val_target_collection, config_path):
    f = open(config_path, 'r').read()
    config = json.loads(f)

    paths = config['paths']
    training_params = config['training_params']
    augmentation_params = config['augmentation_params']
    paths['train_source_collection'] = train_source_collection
    paths['train_target_collection'] = train_target_collection
    paths['val_target_collection'] = val_target_collection
    paths['config_path'] = config_path

    trainer = Trainer(paths=paths,
                      training_params=training_params,
                      augmentation_params=augmentation_params,
                      )
    trainer.train()
    return trainer.run_num


if __name__ == '__main__':
    train_source_collection = sys.argv[1]
    train_target_collection = sys.argv[2]
    val_target_collection = sys.argv[3]
    config_path = sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

    main(train_source_collection, train_target_collection, val_target_collection, config_path)
