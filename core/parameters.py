"""Module to parse and generate parameters for RetinaFace"""
import sys
import json
from data import cfg_mnet, cfg_re50

class RetinaFaceParameters():
    """RetinaFace Parameter Genaration class"""
    def __init__(self):
        # bgr order
        self.config_dict = {
            "rgb_mean": (104, 117, 123),
            "num_classes": 2
        }

    def _parse_config(self):
        """Parse model configs"""
        if self.config_dict["network"] == "mobilenet0.25":
            net_cfg = cfg_mnet
        elif self.config_dict["network"] == "Resnet50":
            net_cfg = cfg_re50
        else:
            print("No config for backbone. Exiting app")
            sys.exit()
        self.config_dict["name"] = net_cfg["name"]
        self.config_dict["img_dim"] = net_cfg["image_size"]
        self.config_dict["num_gpu"] = net_cfg["ngpu"]
        self.config_dict["batch_size"] = net_cfg["batch_size"]
        self.config_dict["max_epoch"] = net_cfg["epoch"]
        self.config_dict["gpu_train"] = net_cfg["gpu_train"]
        self.config_dict["step_dacay"] = {"decay_1": net_cfg["decay1"],
                           "decay_2": net_cfg["decay2"]}
        self.config_dict["loc_weight"] = net_cfg["loc_weight"]

    def parse_json(self, config_file):
        """Parse json config file"""
        with open(config_file) as json_file:
            config_data = json.load(json_file)
        for k, v in config_data.items():
            self.config_dict[k] = v

    def get_params(self, config_file):
        """Parse parameters ffor RetinaFace"""
        self.parse_json(config_file)
        self._parse_config()
        return self.config_dict
