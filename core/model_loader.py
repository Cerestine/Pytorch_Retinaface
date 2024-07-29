"""Module to load pytorch RetinaFace models"""
from collections import OrderedDict
import torch
from models.retinaface import RetinaFace

class ModelLoader():
    """Class to load pytorch model for RetinaFace"""
    def __init__(self, cfg):
        self.net = RetinaFace(cfg=cfg)
        print("Printing net...")
        print(self.net)

    def _load_resume_model(self, state_dict):
        print('Loading network to resume training...')
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def load_model(self, model_name, num_gpu, gpu_train, resume=None):
        """Load checkpoint file"""
        state_dict = torch.load(model_name)
        if resume is not None:
            self._load_resume_model(state_dict)
        if num_gpu > 1 and gpu_train:
            self.net = torch.nn.DataParallel(self.net).cuda()
        else:
            self.net = self.net.cuda()
    