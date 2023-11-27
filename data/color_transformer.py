# 对数据进行颜色变换
import numpy as np
import cv2
from skimage.draw import ellipse, disk


def get_ellipse(x, y, a, b, theta=0, shape=None):
    rr, cc = ellipse(y, x, b, a, rotation=np.deg2rad(theta), shape=shape)
    return rr, cc


def get_circle(x, y, R, shape=None):
    rr, cc = disk((y, x), R, shape=shape)
    return rr, cc


def random_pick_regions(height, width, region_num_max=5, axis_rate=0.3):
    rrs, ccs = np.array([], dtype=np.int), np.array([], dtype=np.int)
    for i in range(region_num_max):
        x = np.random.randint(width)
        y = np.random.randint(height)
        a = np.random.randint(min(height, width) * axis_rate)
        b = np.random.randint(min(height, width) * axis_rate)
        theta = np.random.randint(360)
        rr, cc = ellipse(y, x, b, a, rotation=np.deg2rad(theta), shape=(height, width))

        rrs = np.concatenate((rrs, rr))
        ccs = np.concatenate((ccs, cc))
    return rrs, ccs


class ColorTransformer(object):
    @staticmethod
    def enhance(im):
        lab = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 30))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        im_new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return im_new

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
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        im /= 255
        im_trans = im ** gamma
        return im_trans * 255

    @staticmethod
    def multiple_rgb(im, alphas):
        for i in range(len(alphas)):
            im[:, :, [i]] *= alphas[i]
        return im

    @staticmethod
    def region_gaussian_blur(im, rr, cc, ksize, sigma=0):
        im = im.copy()
        if sigma == 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        im_blur = cv2.GaussianBlur(im, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        if len(rr) == 0:
            im = im_blur
        else:
            im[rr, cc] = im_blur[rr, cc]
        return im

    @staticmethod
    def region_gaussian_noise(im, rr, cc, var):
        im = im.copy().astype(np.float32) / 255.
        noise = np.random.normal(0, var ** 0.5, im.shape)
        if len(rr) == 0:
            im += noise
        else:
            im[rr, cc] += noise[rr, cc]
        im = np.clip(im, 0, 1.0)
        im = im * 255
        return im

    @staticmethod
    def region_salt_pepper_noise(im, rr, cc, amount=0.01, salt_vs_pepper=0.5):
        im = im.copy().astype(np.float32) / 255.

        im_noise = im.copy()

        flipped = np.random.choice([True, False], size=im.shape, p=[amount, 1 - amount])
        salted = np.random.choice([True, False], size=im.shape, p=[salt_vs_pepper, 1 - salt_vs_pepper])
        peppered = ~salted
        im_noise[flipped & salted] = 1
        im_noise[flipped & peppered] = 0
        if len(rr) == 0:
            im = im_noise
        else:
            im[rr, cc] = im_noise[rr, cc]
        im = im * 255
        return im

    @staticmethod
    def round_spot_noise(im, x, y, R, mean=15, var=1):
        im = im.copy().astype(np.float32)
        noise = np.random.normal(mean, var ** 0.5, im.shape)
        noise = im + noise

        ksize = 11
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        noise = cv2.GaussianBlur(noise, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        rr, cc = get_circle(x, y, R, im.shape[:2])
        im[rr, cc] = noise[rr, cc]
        im = np.clip(im, 0, 255)
        return im
