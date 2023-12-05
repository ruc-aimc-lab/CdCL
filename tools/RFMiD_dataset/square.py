# Crop the RFMiD images into squares.
# The parameters are pre-calculated based on a circular fitting algorithm.
import cv2
import numpy as np
import os
from tqdm import tqdm

img_root = './imagedata/RFMiD'  # put all images here
out_root = img_root.rstrip(os.sep) + '_masked'
params_path = './tools/mask_RFMiD_dataset/RFMiD_square_params.csv'

os.makedirs(out_root, exist_ok=True)    

with open(params_path) as fin:
    lines = fin.readlines()
for line in tqdm(lines):

    """
    h, w: raw height and width of the image
    X, Y: the coordinate of the top left corner of the square
    D: the edge length of the square
    """
    img_name, h, w, X, Y, D = line.strip('\r\n').split(',')
    h, w, X, Y, D = int(h), int(w), int(X), int(Y), int(D)

    path_in = os.path.join(img_root, '{}.png'.format(img_name))
    path_out = os.path.join(out_root, '{}.jpg'.format(img_name))

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
    cv2.imwrite(path_out, square_im)

