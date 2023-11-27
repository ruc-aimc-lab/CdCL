import os
import sys
import json

from nets import build_model, single_domain_build_model
from data import build_uwf_dataloader
import numpy as np
import torch



model = single_domain_build_model(model_name='basic_uwf_resnet50', training_params={'n_class': 3, 'custom_pretrained': None})
din = torch.zeros((2, 3, 515, 512))
dout = model([din])
print(dout.size())
