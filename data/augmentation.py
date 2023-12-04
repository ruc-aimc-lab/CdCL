# data augmentation
# can be replaced by torchvision.transforms
import numpy as np
import cv2
from .color_transformer import ColorTransformer
from .geometric_transformer import GeometricTransformer


class MyAug(ColorTransformer, GeometricTransformer):
   
    def __init__(self, params):
        self.aug_cfg = params

    def process(self, img, rand_values=None):
        chosen_value = {}

        im = np.copy(img)
        if self.aug_cfg.get('rotation', False):
            if rand_values:
                rotate_params = rand_values['rotate_params']
            else:
                rotate_params = np.random.randint(self.aug_cfg['rotation_range'][0], self.aug_cfg['rotation_range'][1])
                chosen_value['rotate_params'] = rotate_params
            im = self.rotate(im, rotate_params)

        if self.aug_cfg.get('crop', False):
            if rand_values:
                do_crop = rand_values['do_crop']
            else:
                do_crop = self.aug_cfg['crop_prob'] > np.random.rand()
                chosen_value['do_crop'] = do_crop

            if do_crop:

                if rand_values:
                    w0, w1 = rand_values['w0'], rand_values['w1']
                    h0, h1 = rand_values['h0'], rand_values['h1']
                else:
                    h, w = im.shape[:2]
                    w_dev = int(self.aug_cfg['crop_w'] * w)
                    h_dev = int(self.aug_cfg['crop_h'] * h)
                
                    w0 = np.random.randint(0, w_dev + 1)
                    w1 = np.random.randint(0, w_dev + 1)
                    h0 = np.random.randint(0, h_dev + 1)
                    h1 = np.random.randint(0, h_dev + 1)

                    chosen_value['w0'] = w0
                    chosen_value['w1'] = w1
                    chosen_value['h0'] = h0
                    chosen_value['h1'] = h1
                im = self.crop(im, w0, w1, h0, h1)

        if self.aug_cfg.get('flip', False):
            if rand_values:
                do_flip = rand_values['do_flip']
            else:
                do_flip = self.aug_cfg['flip_prob'] > np.random.rand()
                chosen_value['do_flip'] = do_flip
           
            if do_flip:
                im = self.flip(im)
               
        if self.aug_cfg.get('zoom', False):
            if rand_values:
                do_zoom = rand_values['do_zoom']
            else:
                do_zoom = self.aug_cfg['zoom_prob'] > np.random.rand()
                chosen_value['do_zoom'] = do_zoom
           
            if do_zoom:
                if rand_values:
                    w_dev = rand_values['w_dev']
                else:
                    zoom_min, zoom_max = self.aug_cfg['zoom_range']
                    h, w = im.shape[:2]
                    w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * w)
                    chosen_value['w_dev'] = w_dev
                im = self.zoom(im, w_dev)
        
        
        output_shape = (self.aug_cfg['size_w'], self.aug_cfg['size_h'])
        if tuple(im.shape[:2]) != output_shape:
            im = cv2.resize(im, output_shape)

        # start color augmentation
        if self.aug_cfg.get('gamma', False):
            if rand_values:
                pass
            else:
                gamma_options = self.aug_cfg['gamma_options']
                rand = np.random.randint(0, len(gamma_options))
                gamma_param = gamma_options[rand]
                chosen_value['gamma_param'] = gamma_param
                im = self.gamma_trans(im, gamma_param)

        if self.aug_cfg.get('contrast', False):
            if rand_values:
                pass
            else:
                contrast_param = np.random.uniform(self.aug_cfg['contrast_range'][0], self.aug_cfg['contrast_range'][1])
                chosen_value['contrast_param'] = contrast_param
                im = self.contrast(im, contrast_param)

        if self.aug_cfg.get('brightness', False):
            if rand_values:
                pass
            else:
                brightness_param = np.random.uniform(self.aug_cfg['brightness_range'][0], self.aug_cfg[
                    'brightness_range'][1])
                chosen_value['brightness_param'] = brightness_param
                im = self.brightness(im, brightness_param)

        if self.aug_cfg.get('saturation', False):
            if rand_values:
                pass
            else:
                saturation_param = np.random.uniform(self.aug_cfg['saturation_range'][0], self.aug_cfg[
                    'saturation_range'][1])
                chosen_value['saturation_param'] = saturation_param
                im = self.saturation(im, saturation_param)

        im = self.clip(im)
        return im, chosen_value
