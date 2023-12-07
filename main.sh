# train_RFMiD:          the training set of source domain
# train_TOP_10percent:  the training set of target domain
# val_TOP:              the validation set of target domain
# test_TOP:             the test set of target domain
# 0:                    the gpu id
python main.py train_RFMiD train_TOP_10percent val_TOP configs/cdcl_effi_b3_p.json test_TOP 0