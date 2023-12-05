# Crop the RFMiD images into squares.
# The parameters are pre-calculated based on a circular fitting algorithm.
import cv2
import numpy as np
import os
from tqdm import tqdm

img_root = '../A. RFMiD_All_Classes_Dataset/1. Original Images'  # your downloaded RFMiD dataset
out_root = './imagedata/RFMiD'
params_path = 'tools/RFMiD_dataset/RFMiD_square_params.csv'

os.makedirs(out_root, exist_ok=True)    

dataset_mapping = {'a. Training Set': 'train', 'b. Validation Set': 'val', 'c. Testing Set': 'test'}
with open(params_path) as fin:
    lines = fin.readlines()[1:]
for line in tqdm(lines):

    """
    h, w: raw height and width of the image
    X, Y: the coordinate of the top left corner of the square
    D: the edge length of the square
    """
    fname, h, w, X, Y, D = line.strip('\r\n').split(',')
    h, w, X, Y, D = int(h), int(w), int(X), int(Y), int(D)

    path_in = os.path.join(img_root, '{}.png'.format(fname))
    
    dataset, img_name = fname.split('/')
    img_name = '{}_{}.jpg'.format(dataset_mapping[dataset], img_name)
    
    path_out = os.path.join(out_root, img_name)

    im = cv2.imread(path_in)
    assert list(im.shape[:2]) == [h, w]

    square_im = np.zeros((D, D, 3), dtype=np.uint8)

    square_h_start = max(0, -Y)
    square_h_end = min(D, square_h_start + h)
    square_w_start = max(0, -X)
    square_w_end = min(D, square_w_start + w)

    raw_h_start = max(0, Y)
    raw_h_end = min(h, raw_h_start + D)
    raw_w_start = max(0, X)
    raw_w_end = min(w, raw_w_start + D)

    square_im[square_h_start:square_h_end, square_w_start:square_w_end, :] = im[raw_h_start:raw_h_end, raw_w_start:raw_w_end, :]
    
    circle_mask = np.zeros((D, D), dtype=np.uint8)
    cv2.circle(circle_mask, (int(D/2), int(D/2)), int(D/2), 255, -1)
    square_im[circle_mask==0] = 0
    cv2.imwrite(path_out, square_im)

