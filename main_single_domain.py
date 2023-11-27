from train_single_domain import main as train_main
from predict_single_domain import main as predict_main
from evaluate_single_domain import main as evaluate_main
import sys
import os


def main(train_collection, val_collection, config_path, test_collection):
    run_num = train_main(train_collection, val_collection, config_path)
    predict_main(train_collection, val_collection, config_path, test_collection, run_num)
    evaluate_main(train_collection, val_collection, config_path, test_collection, run_num)


if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    test_collection = sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
    main(train_collection, val_collection, config_path, test_collection)
    # python main_single_domain.py train_uwf_md val_uwf_md configs_single_domain/basic_inception_uwf_whole_imagenet.json test_uwf_md 1
