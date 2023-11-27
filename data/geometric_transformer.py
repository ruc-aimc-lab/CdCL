# 对数据进行几何变换
import numpy as np
import cv2


class GeometricTransformer(object):
    @staticmethod
    def resize(im, size=(512, 512)):
        e = 1e-15
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        im_min, im_max = np.min(im), np.max(im)
        im_std = (im - im_min) / (im_max - im_min + e)
        resized_std = cv2.resize(im_std, size)
        resized_im = resized_std * (im_max - im_min) + im_min
        return resized_im

    @staticmethod
    def rotate(im, rotation_param, keep_aspect_ratio=True):
        h, w = im.shape[:2]
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), -rotation_param, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        if not keep_aspect_ratio:
            M[0, 2] += (w // 2) - cX
            M[1, 2] += (h // 2) - cY
            im_new = cv2.warpAffine(np.array(im, dtype=np.float32), M, (w, h))
            return im_new

        nW = int(h * sin + w * cos)
        nH = int(h * cos + w * sin)
        M[0, 2] += (nW // 2) - cX
        M[1, 2] += (nH // 2) - cY
        im_new = cv2.warpAffine(np.array(im, dtype=np.float32), M, (nW, nH))

        x0 = int(max(0, (nW - w) / 2))
        x1 = int(min((nW + w) / 2, nW))
        y0 = int(max(0, (nH - h) / 2))
        y1 = int(min((nH + h) / 2, nH))
        return im_new[y0:y1, x0:x1]

    @staticmethod
    def flip(im):
        return cv2.flip(im, 1)

    @staticmethod
    def zoom(im, w_dev):
        h, w = im.shape[:2]
        return im[w_dev:h - w_dev, w_dev:w - w_dev]

    @staticmethod
    def crop(im, w0=0, w1=0, h0=0, h1=0):
        h, w = im.shape[:2]
        return im[h0:h - h1, w0:w - w1]
