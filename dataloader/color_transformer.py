# 对数据进行颜色变换
import numpy as np

class ColorTransformer(object):
    @staticmethod
    def clip(im):
        im[im > 255] = 255
        im[im < 0] = 0
        return im

    @staticmethod
    def brightness(im, alpha):
        im *= alpha
        return im

    @staticmethod
    def contrast(im, alpha):
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = im * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        im *= alpha
        im += gray
        return im

    @staticmethod
    def saturation(im, alpha):
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = im * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        im *= alpha
        im += gray
        return im

    @staticmethod
    def gamma_trans(im, gamma):
        im /= 255.
        im_trans = im ** gamma
        return im_trans * 255.

    
