import sys
import os
from train import main as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main



def main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection):
    run_num = train_main(train_source_collection, train_target_collection, val_target_collection, config_path)
    predict_main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num)
    evaluate_main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection, run_num)


if __name__ == '__main__':
    train_source_collection = sys.argv[1]  # The training set of source domain
    train_target_collection = sys.argv[2]  # The training set of target domain
    val_target_collection = sys.argv[3]    # The validation set of target domain
    config_path = sys.argv[4]              
    test_target_collection = sys.argv[5]   # The test set of target domain
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]
    main(train_source_collection, train_target_collection, val_target_collection, config_path, test_target_collection)
