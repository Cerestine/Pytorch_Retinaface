"""Module to load pytorch RetinaFace models"""
from collections import OrderedDict
import torch
from models.retinaface import RetinaFace

class ModelLoader():
    """Class to load pytorch model for RetinaFace"""
    def __init__(self, cfg, train=True):
        if train:
            self.mode = "train"
        else:
            torch.set_grad_enabled(False)
            self.mode = "test"
        self.net = RetinaFace(cfg=cfg, phase=self.mode)

    def _load_resume_model(self, state_dict):
        print("Loading network to resume training...")
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == "module.":
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

    def _load_pretrained_dict(self, pretrained_path, load_to_cpu):
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        return pretrained_dict

    def _remove_prefix(self, state_dict, prefix):
        """ Old style model is stored with all names of parameters sharing common prefix "module." """
        print("remove prefix \"{}\"".format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def load_train_model(self, model_name, num_gpu, gpu_train, resume=None):
        """Load checkpoint file"""
        print("Printing net...")
        print(self.net)
        state_dict = torch.load(model_name)
        if resume is not None:
            self._load_resume_model(state_dict)
        if num_gpu > 1 and gpu_train:
            self.net = torch.nn.DataParallel(self.net).cuda()
        else:
            self.net = self.net.cuda()

    def load_detection_model(self, pretrained_path, load_to_cpu):
        """Load pretrained model"""
        print("Loading pretrained model from {}".format(pretrained_path))
        pretrained_dict = self._load_pretrained_dict(pretrained_path, load_to_cpu)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict["state_dict"], "module.")
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, "module.")
        self._check_keys(self.net, pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()
        print('Finished loading model!')
        print(self.net)
    