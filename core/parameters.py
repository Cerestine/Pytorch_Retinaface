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
        for k, v in net_cfg.items():
            if k in ["decay1", "decay2"]:
                if "step_dacay" not in self.config_dict.keys():
                    self.config_dict["step_dacay"] = {}
                self.config_dict["step_dacay"][k] = v
            self.config_dict[k]  = v

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
