# RFMiD_train:          the training set of source domain
# TOP_train_10percent:  the training set of target domain
# TOP_val:              the validation set of target domain
# TOP_test:             the test set of target domain
# 0:                    the gpu id
python main.py RFMiD_train TOP_train_10percent TOP_val configs/cdcl_effi_b3_p.json TOP_test 0