from train import main as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main
import sys
import os


def main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, test_uwf_collection):
    run_num = train_main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path)
    predict_main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, test_uwf_collection, run_num)
    evaluate_main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, test_uwf_collection, run_num)


if __name__ == '__main__':
    train_cfp_collection = sys.argv[1]
    train_uwf_collection = sys.argv[2]
    val_uwf_collection = sys.argv[3]
    config_path = sys.argv[4]
    test_uwf_collection = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]
    main(train_cfp_collection, train_uwf_collection, val_uwf_collection, config_path, test_uwf_collection)
    # python main.py train val configs/inception_v3_WeightedBCEWithLogitsLoss_05.json test 0
